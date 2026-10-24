"""Evaluate a checkpoint on every clip of a motion set (paper Appendix F protocol).

  uv run python scripts/eval_benchmark.py --ckpt logs/e2/ckpt_latest.pt --npz-dir data/npz/lafan1
  uv run python scripts/eval_benchmark.py --ckpt logs/bones_main/ckpt_h3.pt --library $SM_BANK/bones_seed/library/bones_seed_compact.pt
"""

from __future__ import annotations

from pathlib import Path

import torch
import tyro

from supermimic.config import obs_mode_from_config
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark, save_results, summarize


def main(ckpt: Path, npz_dir: Path = Path("data/npz/lafan1"), clips: list[str] | None = None, clip_list: Path | None = None, manifest: Path | None = None, library: Path | None = None, num_envs: int = 1024, device: str = "cuda:0", out: Path | None = None) -> None:
    """``--library`` evaluates every clip of a compact library (build a subset library for subsets)."""
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    common = dict(device=device, num_envs=num_envs, future_frames=cfg["env"]["future_frames"], obs_mode=obs_mode_from_config(cfg))
    if library is not None:
        n_clips = len(torch.load(library, map_location="cpu", weights_only=False)["names"])
        results = run_benchmark(policy, [str(library)], compact_library=True, num_clips=n_clips, **common)
    else:
        if clip_list is not None:
            clips = [l.strip() for l in open(clip_list) if l.strip()]
        files = [str(f) for f in discover_npz(npz_dir, names=clips, manifest=manifest)]
        results = run_benchmark(policy, files, **common)
    s = summarize(results)
    print(
        f"\n{s['clips']} clips | failed {s['failed']} | norm return (paper §4) mean {s['mean_norm_return']:.4f} length-weighted {s['length_weighted_norm_return']:.4f}"
        f" | per-executed-step (upstream) mean {s['mean_norm_return_per_step']:.4f} step-weighted {s['step_weighted_norm_return_per_step']:.4f}"
    )
    if "terms" in s:
        print("per-step term means (max 0.02 each for the 1.0-weight terms, 0.01 for root terms):")
        for k, v in s["terms"].items():
            print(f"  {k:28s} {v:.4f}")
    worst = sorted(results, key=lambda r: r.norm_return)[:10]
    for r in worst:
        print(f"  {r.name:32s} ret {r.norm_return:.3f} steps {r.steps}/{r.frames} {'FAIL@' + str(r.fail_step) if r.failed else ''}")
    out = out or ckpt.parent / f"benchmark_{ckpt.stem}.json"
    save_results(results, out)
    print("saved", out)


if __name__ == "__main__":
    tyro.cli(main)
