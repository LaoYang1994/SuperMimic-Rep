"""Appendix E filters for BONES-SEED: name filter + feasibility filter.

The paper keeps 142,220 -> 129,667 after the name filter -> 129,514 after the feasibility filter. Its name
filter is SONIC's keyword list "plus a few terms it misses"; the extra terms are decided by matching those two
counts (change 2026-10-24-paper-faithful-reproduction, T47/T48). So the per-batch pass applies only SONIC's
list + feasibility, and the extra terms are applied afterwards on the merged manifest:

  # per batch of converted npz clips (50 Hz, 30 body link states): writes <npz_dir>/manifest_filtered.csv
  python scripts/data/filter_bones_seed.py --npz-dir <batch npz dir> --workers 16
  # name-filter count only, from the release metadata (matches on its `filename` column = CSV/npz stem)
  python scripts/data/filter_bones_seed.py --metadata seed_metadata_v004.csv --extra-keywords box_ vault
  # shards of stream_bones_seed.py: filter + pack each shard (outputs read by merge_library_parts.py)
  python scripts/data/filter_bones_seed.py --shards $SM_BANK/bones_seed/shards --parts $SM_BANK/bones_seed/parts --workers 4
  # final selection on the merged manifest (+ optional clip list output)
  python scripts/data/filter_bones_seed.py --metadata seed_metadata_v004.csv --merged-manifest merged.csv --out keep.txt
"""

from __future__ import annotations

import csv
import multiprocessing as mp
from collections import Counter
from pathlib import Path

import numpy as np
import tyro

# SONIC's complete default list, verbatim (NVlabs/GR00T-WholeBodyControl,
# gear_sonic/data_process/filter_and_copy_bones_data.py), matched case-insensitively as substrings.
SONIC_KEYWORDS = [
    "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on", "seat", "table", "_sit_", "sit_",
    "ladder", "crutch", "_bed_", "_ride_", "scooter", "stepdown", "acrobatics_", "box_HSPU", "cartwheel",
    "50cm_box_", "on_box", "fall_from", "handstand_ff_", "on_1m", "form_box", "off_1m", "230m",
    "jump_over_obstacle_", "lift_crate_come_up_", "jump_to_shoulder_roll", "kozak_dance", "stair", "handstand",
    "box_jump", "monkey_jump", "safety_roll", "box_dips", "walking_on_edge", "push_obstacle",
]  # fmt: skip
# The paper's "few terms it misses". Chosen by the counts (change 2026-10-24-paper-faithful-reproduction, T47/T48):
# 12 small sets give exactly 129,667 after the name filter, none also gives exactly 129,514 after our feasibility
# filter (whose standing/stationary gates the paper leaves open). This set is closest (129,511, -3) and uses the
# paper's own examples ("box", "jump off 50cm"). Others at 129,667: box_+jump_off_50cm+off_50cm+vault (129,511),
# box_+off_50cm+vault (129,511), box_+fence/look_over(_fence)+vault (129,487), fence/look_over(_fence)+50cm+vault+_sit (129,557).
EXTRA_KEYWORDS: list[str] = ["box_", "jump_off_50cm", "vault"]
KEYWORDS = [k.lower() for k in SONIC_KEYWORDS]

# Feasibility thresholds (paper Appendix E + plan D16 choices).
ELEVATED_Z = 0.20
ELEVATED_S = 0.6
PENETRATE_Z_HARD = -0.12
PENETRATE_Z_SOFT = -0.08
PENETRATE_FRAC = 0.80
FLOAT_Z = 0.02
FLOAT_S = 0.5
UPRIGHT_GZ = 0.9
STAND_Z = (0.60, 0.85)
STILL_V = 0.1
STILL_QD = 0.5
FPS = 50


def name_filter(name: str, keywords: list[str] = KEYWORDS) -> str | None:
    n = name.lower()
    for k in keywords:
        if k.lower() in n:
            return f"name:{k}"
    return None


def feasibility_filter(npz_path: str) -> str | None:
    with np.load(npz_path) as d:
        names = [str(x) for x in d["body_names"]]
        arrays = {k: d[k] for k in ("body_pos_w", "body_quat_w", "joint_vel", "joint_pos", "body_lin_vel_w")}
    return feasibility(names, arrays)


def feasibility(names: list[str], d: dict) -> str | None:
    """Appendix E feasibility reason for one clip's 50 Hz arrays (None = feasible)."""
    pos = d["body_pos_w"]  # (T, 30, 3)
    quat = d["body_quat_w"]  # (T, 30, 4) wxyz
    jv = d["joint_vel"]
    lin = d["body_lin_vel_w"]
    T = pos.shape[0]
    zmin = pos[:, :, 2].min(1)
    lf, rf, pel = names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link"), names.index("pelvis")
    # elevated support: zmin > 0.20 for >= 0.6 s
    if _longest_run(zmin > ELEVATED_Z) >= ELEVATED_S * FPS:
        return "elevated_support"
    if (zmin < PENETRATE_Z_HARD).any() or (zmin < PENETRATE_Z_SOFT).mean() > PENETRATE_FRAC:
        return "ground_penetration"
    q = quat[:, pel]
    # gravity z-component in pelvis frame: -(R^T e_z)_z = -(1 - 2(x^2+y^2)) for wxyz quaternion
    gz = -(1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2))
    upright = gz < -UPRIGHT_GZ
    standing = (pos[:, pel, 2] > STAND_Z[0]) & (pos[:, pel, 2] < STAND_Z[1])
    still = (np.linalg.norm(lin[:, pel], axis=1) < STILL_V) & (np.sqrt((jv**2).mean(1)) < STILL_QD)
    gate = upright & standing & still
    if _longest_run(gate) >= FLOAT_S * FPS:
        # Exact lowest point of the foot collision capsules (paper: mj_geomDistance to the plane).
        foot_z = _foot_sole_height(d["body_pos_w"][:, pel], d["body_quat_w"][:, pel], d["joint_pos"], gate)
        floating = gate.copy()
        floating[gate] = foot_z > FLOAT_Z
        if _longest_run(floating) >= FLOAT_S * FPS:
            return "floating_feet"
    return None


_MODEL = None
_FOOT_GEOMS = None


def _foot_sole_height(root_pos: np.ndarray, root_quat: np.ndarray, jp: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Lowest point over both feet's collision capsules, for the gated frames (returns len(gate.sum()),)."""
    global _MODEL, _FOOT_GEOMS
    import mujoco
    import re

    if _MODEL is None:
        from supermimic.data.motion_convert import load_g1_model

        _MODEL = load_g1_model()
        _FOOT_GEOMS = [g for g in range(_MODEL.ngeom) if re.match(r"^(left|right)_foot[1-7]_collision$", mujoco.mj_id2name(_MODEL, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
        assert len(_FOOT_GEOMS) == 14 and all(_MODEL.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE for g in _FOOT_GEOMS)
    m = _MODEL
    d = mujoco.MjData(m)
    idx = np.nonzero(gate)[0]
    out = np.empty(len(idx))
    radius = m.geom_size[_FOOT_GEOMS, 0]
    half = m.geom_size[_FOOT_GEOMS, 1]
    for k, t in enumerate(idx):
        d.qpos[:3] = root_pos[t]
        d.qpos[3:7] = root_quat[t]
        d.qpos[7:] = jp[t]
        mujoco.mj_kinematics(m, d)
        cz = d.geom_xpos[_FOOT_GEOMS, 2]
        axis_z = d.geom_xmat[_FOOT_GEOMS].reshape(-1, 3, 3)[:, 2, 2]  # world z of the capsule axis
        out[k] = (cz - half * np.abs(axis_z) - radius).min()
    return out


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def _one(path: str) -> tuple[str, int, str | None]:
    name = Path(path).stem
    r = name_filter(name)
    frames = 0
    if r is None:
        r = feasibility_filter(path)
    with np.load(path) as d:
        frames = int(d["joint_pos"].shape[0])
    return name, frames, r


def _model_body_names() -> list[str]:
    """Body order of the converter's FK output (= every converted clip's body axis)."""
    import mujoco

    from supermimic.data.motion_convert import load_g1_model

    m = load_g1_model()
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, m.nbody)]


def _one_shard(args: tuple[str, str]) -> str:
    """Filter one stream_bones_seed.py shard and pack its feasible clips: writes <parts>/<shard>.manifest.csv
    (every clip: name, frames, keep, reason) and <parts>/<shard>.part.pt (fp16 71-d reference of kept clips)."""
    import torch

    shard, parts = Path(args[0]), Path(args[1])
    names = _model_body_names()
    pel = names.index("pelvis")
    clips = [l.strip() for l in open(shard.with_suffix(".txt")) if l.strip()]
    rows, refs, kept, lengths = [], [], [], []
    with np.load(shard) as z:
        for clip in clips:
            d = {k: z[f"{clip}/{k}"] for k in ("body_pos_w", "body_quat_w", "joint_vel", "joint_pos", "body_lin_vel_w", "body_ang_vel_w")}
            frames = int(d["joint_pos"].shape[0])
            r = name_filter(clip) or feasibility(names, d)
            rows.append((clip, frames, r))
            if r is None:
                ref = np.concatenate([d["body_pos_w"][:, pel], d["body_quat_w"][:, pel], d["joint_pos"], d["body_lin_vel_w"][:, pel], d["body_ang_vel_w"][:, pel], d["joint_vel"]], axis=1)
                refs.append(torch.from_numpy(ref.astype(np.float16)))
                kept.append(clip)
                lengths.append(frames)
    with open(parts / f"{shard.stem}.manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "frames", "keep", "reason"])
        for name, frames, r in rows:
            w.writerow([name, frames, int(r is None), r or ""])
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    tmp = parts / f"{shard.stem}.part.pt.tmp"
    torch.save({"ref": torch.cat(refs, 0) if refs else torch.zeros(0, 71, dtype=torch.float16), "names": kept, "starts": starts, "lengths": lengths}, tmp)
    tmp.rename(parts / f"{shard.stem}.part.pt")
    return f"{shard.stem}: {len(clips)} clips, kept {len(kept)}"


def filter_shards(shards: Path, parts: Path, workers: int) -> None:
    parts.mkdir(parents=True, exist_ok=True)
    todo = [(str(p), str(parts)) for p in sorted(shards.glob("shard_[0-9][0-9][0-9][0-9].npz")) if not (parts / f"{p.stem}.part.pt").exists()]
    print(f"{len(todo)} shards to filter", flush=True)
    with mp.Pool(workers) as pool:
        for msg in pool.imap_unordered(_one_shard, todo):
            print(msg, flush=True)


def _metadata_names(metadata: Path) -> list[str]:
    csv.field_size_limit(1 << 30)
    with open(metadata, newline="") as f:
        return [row["filename"] for row in csv.DictReader(f)]


def final_selection(metadata: Path, merged_manifest: Path | None, extra_keywords: list[str], out: Path | None = None) -> None:
    """Name-filter count from the metadata; with a merged manifest, also the feasibility-filtered selection."""
    names = _metadata_names(metadata)
    kw = KEYWORDS + [k.lower() for k in extra_keywords]
    after_name = [n for n in names if name_filter(n, kw) is None]
    print(f"raw {len(names)} -> after name filter {len(after_name)} (paper 129,667); extra keywords {extra_keywords}")
    if merged_manifest is None:
        return
    with open(merged_manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    keep = [r for r in rows if r["keep"] == "1" and name_filter(r["name"], kw) is None]
    hours = sum(int(r["frames"]) for r in keep) / FPS / 3600
    print(f"after feasibility {len(keep)} (paper 129,514); {hours:.2f} h (paper 264.56)")
    if out is not None:
        out.write_text("".join(r["name"] + "\n" for r in keep))
        print("saved", out)


def main(
    npz_dir: Path | None = None,
    workers: int = 16,
    out: Path | None = None,
    metadata: Path | None = None,
    merged_manifest: Path | None = None,
    extra_keywords: list[str] | None = None,
    shards: Path | None = None,
    parts: Path | None = None,
) -> None:
    if shards is not None:
        if parts is None:
            raise SystemExit("--shards needs --parts (output directory for manifests and library parts)")
        filter_shards(shards, parts, workers)
        return
    if npz_dir is None:
        if metadata is None:
            raise SystemExit("give --npz-dir (per-batch filtering) or --metadata (name count / final selection)")
        final_selection(metadata, merged_manifest, EXTRA_KEYWORDS if extra_keywords is None else extra_keywords, out)
        return
    files = sorted(str(p) for p in npz_dir.glob("*.npz"))
    print(f"{len(files)} clips")
    with mp.Pool(workers) as pool:
        rows = pool.map(_one, files, chunksize=64)
    out = out or npz_dir / "manifest_filtered.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "frames", "keep", "reason"])
        for name, frames, r in rows:
            w.writerow([name, frames, int(r is None), r or ""])
    reasons = Counter(r for _, _, r in rows)
    kept = sum(1 for _, _, r in rows if r is None)
    name_removed = sum(1 for _, _, r in rows if r and r.startswith("name:"))
    feas_removed = sum(1 for _, _, r in rows if r and not r.startswith("name:"))
    hours = sum(fr for _, fr, r in rows if r is None) / FPS / 3600
    print(f"{len(rows)} clips -> after SONIC name filter {len(rows) - name_removed} -> after feasibility {kept} ({feas_removed} removed); {hours:.2f} h kept")
    print("by reason:", dict(reasons.most_common(40)))
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)
