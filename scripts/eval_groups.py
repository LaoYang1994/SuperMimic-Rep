"""Failed motions per motion group, as paper Table A.8, from a benchmark JSON (or group sizes of a clip list).

  python scripts/eval_groups.py --benchmark logs/bones_main/benchmark_ckpt_h3.json
  python scripts/eval_groups.py --clip-list keep.txt          # sizes only (T57 check against Table A.8)

Groups are assigned by clip name, first match wins. The patterns are our reading of the paper's group names;
the size check against Table A.8 tells whether that reading (and the data filtering) matches the paper.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import tyro

# (group, name regex, paper size, SONIC failures, SuperMimic 1 GPU x 3 h failures)
TABLE_A8 = [
    ("side rolls", r"side_roll", 22, 22, 2),
    ("flips: 360°, 180°, 90°", r"^flip_(090|180|360)_", 44, 44, 1),
    ("fire-in-the-hole turns", r"fire_in_the_hole", 20, 3, 3),
    ("crossed legs", r"crossed_legs", 10, 9, 0),
    ("lying, resting and standing up", r"lying|^rest_|stand_up", 99, 28, 8),
    ("broad jumps", r"broad_jump", 24, 15, 0),
    ("dance routine", r"^dancing_routine_1_", 8, 8, 0),
]
OTHER = ("all other motions", 129_287, 50, 11)


def group_of(name: str) -> str:
    n = name.lower()
    for g, pat, *_ in TABLE_A8:
        if re.search(pat, n):
            return g
    return OTHER[0]


def main(benchmark: Path | None = None, clip_list: Path | None = None) -> None:
    if benchmark is not None:
        clips = json.loads(benchmark.read_text())["clips"]
        rows = [(c["name"], bool(c["failed"])) for c in clips]
    elif clip_list is not None:
        rows = [(l.strip(), False) for l in open(clip_list) if l.strip()]
    else:
        raise SystemExit("give --benchmark or --clip-list")
    size: dict[str, int] = {}
    failed: dict[str, int] = {}
    for name, f in rows:
        g = group_of(name)
        size[g] = size.get(g, 0) + 1
        failed[g] = failed.get(g, 0) + int(f)
    print(f"{'group':34s} {'size':>7s} {'paper':>7s} {'ok':>3s} | {'failed':>6s} {'SONIC':>6s} {'paper 1x3h':>10s}")
    for g, _, n_paper, sonic, sm in TABLE_A8 + [(OTHER[0], None, *OTHER[1:])]:
        n = size.get(g, 0)
        print(f"{g:34s} {n:7d} {n_paper:7d} {'✓' if n == n_paper else '✗':>3s} | {failed.get(g, 0):6d} {sonic:6d} {sm:10d}")
    print(f"{'total':34s} {len(rows):7d} {129_514:7d} {'✓' if len(rows) == 129_514 else '✗':>3s} | {sum(failed.values()):6d} {179:6d} {25:10d}")


if __name__ == "__main__":
    tyro.cli(main)
