"""Where the return-based reset sampler of a checkpoint sends its resets, per motion group (CPU only).

Rebuilds the motion-library bins from the compact library's metadata (no GPU, the 6.7 GB reference is mmapped)
and reads ``bin_return`` / ``bin_memory`` from the checkpoint. Old checkpoints (raw discounted returns, before
change 2026-10-25-sampler-attainable-return) are converted as training does (``bin_return / g_max``).

  python scripts/sampler_report.py --ckpt logs/bones_seq_fast@B/ckpt_0044000.pt \
      --library $SM_BANK/bones_seed/library/bones_seed_compact.pt
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_groups import TABLE_A8, group_of  # noqa: E402

EXTRA_GROUPS = {
    "high / reach jumps": r"^(high_jump|reach_jump)_r",
    "floor (inchworm, faint, ...)": r"inchworm|spread_out|dozing_cross|fetal_position|meditate|faint|balled_up|spider_crawl",
    "walking": r"^walk_",
}


def main(ckpt: Path, library: Path, bin_frames: int = 50, softmax_temp: float = 25.0, g_max: float = 10.0,
         uniform_ratio: float = 0.5, hard_ratio: float = 0.4, memory_ratio: float = 0.1, top: int = 2000) -> None:
    from supermimic.envs.mdp.sampler_math import bin_return_from_checkpoint

    meta = torch.load(library, map_location="cpu", mmap=True, weights_only=False)
    names = [n.lower() for n in meta["names"]]
    lengths = torch.tensor(meta["lengths"])
    bins_per_clip = (lengths + bin_frames - 1) // bin_frames
    nb = int(bins_per_clip.sum())
    bin_clip = torch.repeat_interleave(torch.arange(len(names)), bins_per_clip)
    clip_end = torch.zeros(nb, dtype=torch.bool)
    clip_end[torch.cumsum(bins_per_clip, 0) - 1] = True

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    G = bin_return_from_checkpoint(ck, g_max).float()
    M = ck["bin_memory"].float()
    if G.numel() != nb:
        raise SystemExit(f"checkpoint has {G.numel()} bins, library {nb}: different library or bin_frames")
    hard = torch.softmax(-softmax_temp * G, 0)
    mem = M / M.sum()
    p = uniform_ratio / nb + hard_ratio * hard + memory_ratio * mem
    p = p / p.sum()

    print(f"{ckpt}: {'normalized' if ck.get('bin_return_normalized') else 'legacy (converted)'} bin returns, iteration {ck.get('iteration')}")
    print(f"bins {nb}; clip-end bins {int(clip_end.sum())} ({100 * clip_end.float().mean():.1f}%) hold {100 * hard[clip_end].sum():.1f}% of the hard mass")
    print(f"G_b (completion 0-1): mean {G.mean():.3f}, clip-end bins {G[clip_end].mean():.3f}, other bins {G[~clip_end].mean():.3f}")
    groups = {g: pat for g, pat, *_ in TABLE_A8} | EXTRA_GROUPS
    print(f"{'group':30s} {'bins':>7s} {'uniform%':>9s} {'p%':>9s} {'hard%':>9s} {'memory%':>9s} {'G mean':>7s} {'G min':>7s}")
    for g, pat in groups.items():
        cm = torch.tensor([bool(re.search(pat, n)) for n in names])
        bm = cm[bin_clip]
        if not bm.any():
            continue
        print(f"{g:30s} {int(bm.sum()):7d} {100 * bm.float().mean():9.4f} {100 * p[bm].sum():9.4f} {100 * hard[bm].sum():9.4f} {100 * mem[bm].sum():9.4f} {G[bm].mean():7.3f} {G[bm].min():7.3f}")
    other = torch.tensor([group_of(n) == "all other motions" for n in names])[bin_clip]
    print(f"{'(Table A.8 all other motions)':30s} {int(other.sum()):7d} {100 * other.float().mean():9.4f} {100 * p[other].sum():9.4f} {100 * hard[other].sum():9.4f} {100 * mem[other].sum():9.4f} {G[other].mean():7.3f} {G[other].min():7.3f}")
    idx = torch.topk(hard, min(top, nb)).indices
    fam = collections.Counter(re.sub(r"_\d+(__a\d+(_m)?)?$", "", names[int(bin_clip[i])]) for i in idx)
    print(f"top {len(idx)} hard bins: {100 * hard[idx].sum():.1f}% of the hard mass, {100 * clip_end[idx].float().mean():.1f}% are clip-end bins")
    print("  by motion family:", ", ".join(f"{f} {k}" for f, k in fam.most_common(12)))


if __name__ == "__main__":
    tyro.cli(main)
