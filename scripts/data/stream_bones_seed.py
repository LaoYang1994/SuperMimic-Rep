"""Convert BONES-SEED G1 CSVs straight from g1.tar.gz into 50 Hz motion shards, without extracting the archive.

The GPU box's PFS creates small files at ~3/s and its system disk must hold no data, so every CSV is read from
the tar stream, converted in memory (same functions as csv_to_npz_batch.py), and every `shard_size` clips are
written as ONE npz shard (keys `<clip>/<field>`) to the output directory. Resumable: finished shards are kept.

  python scripts/data/stream_bones_seed.py --archive $SM_BANK/bones_seed/g1.tar.gz --out $SM_BANK/bones_seed/shards
"""

from __future__ import annotations

import io
import multiprocessing as mp
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import tyro

_MODEL = None


def _init() -> None:
    global _MODEL
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from supermimic.data import motion_convert as mc

    _MODEL = mc.load_g1_model()


def _convert(item: tuple[str, bytes]) -> tuple[str, dict | None, str]:
    from supermimic.data import motion_convert as mc

    name, data = item
    try:
        mc.csv_format = lambda _p: "bones"  # every BONES-SEED CSV has the header; load_csv then reads a buffer
        raw = mc.load_csv(io.BytesIO(data))
        motion = mc.resample_motion(raw, 120.0, 50.0)
        b = mc.forward_kinematics(_MODEL, motion)
        out = {
            "joint_pos": motion.joint_pos.astype(np.float32),
            "joint_vel": motion.joint_vel.astype(np.float32),
            "body_pos_w": b.pos_w.astype(np.float32),
            "body_quat_w": b.quat_w.astype(np.float32),
            "body_lin_vel_w": b.lin_vel_w.astype(np.float32),
            "body_ang_vel_w": b.ang_vel_w.astype(np.float32),
        }
        return name, out, ""
    except Exception as e:  # noqa: BLE001
        return name, None, repr(e)


def _members(archive: Path, skip: set[str]):
    with tarfile.open(archive, "r|gz") as tf:
        for m in tf:
            if m.isfile() and m.name.endswith(".csv"):
                name = Path(m.name).stem
                if name in skip:
                    continue
                yield name, tf.extractfile(m).read()


def _batches(it, n: int):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def main(archive: Path, out: Path, shard_size: int = 2000, workers: int = 8, read_ahead: int = 128) -> None:
    """``read_ahead`` CSVs are read from the archive per pool round, which bounds memory (Pool.imap would
    drain the whole archive into its task queue)."""
    out.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    for idx in sorted(out.glob("shard_*.txt")):
        done |= {l.strip() for l in open(idx) if l.strip()}
    n_shard = len(list(out.glob("shard_[0-9][0-9][0-9][0-9].npz")))
    print(f"{len(done)} clips already in {n_shard} shards", flush=True)
    buf: dict[str, np.ndarray] = {}
    names: list[str] = []
    errors = open(out / "errors.txt", "a")
    t0 = time.time()

    def flush():
        nonlocal n_shard, buf, names
        if not names:
            return
        p = out / f"shard_{n_shard:04d}.npz"
        np.savez(str(p) + ".tmp.npz", **buf)
        os.replace(str(p) + ".tmp.npz", p)
        (out / f"shard_{n_shard:04d}.txt").write_text("".join(n + "\n" for n in names))
        n_shard += 1
        print(f"[{(time.time() - t0) / 60:.1f} min] shard {n_shard - 1}: {len(names)} clips, total {len(done)}", flush=True)
        buf, names = {}, []

    with mp.Pool(workers, initializer=_init) as pool:
        for batch in _batches(_members(archive, done), read_ahead):
            for name, arrays, err in pool.map(_convert, batch, chunksize=4):
                if arrays is None:
                    errors.write(f"{name}\t{err}\n")
                    errors.flush()
                    continue
                for k, v in arrays.items():
                    buf[f"{name}/{k}"] = v
                names.append(name)
                done.add(name)
                if len(names) >= shard_size:
                    flush()
    flush()
    print(f"done: {len(done)} clips in {n_shard} shards", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
