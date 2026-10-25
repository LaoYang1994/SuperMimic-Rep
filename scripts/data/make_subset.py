"""Fixed random clip subset (for cheap checkpoint tracking on BONES-SEED; not a substitute for the full benchmark).

  python scripts/data/make_subset.py --clip-list keep.txt --n 2000 --out subset2000.txt
"""

from __future__ import annotations

import random
from pathlib import Path

import tyro


def main(clip_list: Path, out: Path, n: int = 2000, seed: int = 0) -> None:
    names = sorted(l.strip() for l in open(clip_list) if l.strip())
    subset = sorted(random.Random(seed).sample(names, n))
    out.write_text("".join(s + "\n" for s in subset))
    print(f"saved {out}: {n} of {len(names)} clips (seed {seed})")


if __name__ == "__main__":
    tyro.cli(main)
