"""Merge the per-batch outputs of prepare_bones_seed.sh: all manifests into one, and the library parts into one
compact library holding exactly the clips of a list (the final selection, or a subset).

  python scripts/data/merge_library_parts.py --parts <part dir> --merged-manifest merged.csv
  python scripts/data/merge_library_parts.py --parts <part dir> --clip-list keep.txt --out bones_seed_compact.pt
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import tyro


def main(parts: Path, merged_manifest: Path | None = None, clip_list: Path | None = None, out: Path | None = None) -> None:
    manifests = sorted(parts.glob("*.manifest.csv"))
    if merged_manifest is not None:
        with open(merged_manifest, "w", newline="") as f:
            w = None
            for m in manifests:
                with open(m, newline="") as g:
                    for row in csv.DictReader(g):
                        if w is None:
                            w = csv.DictWriter(f, fieldnames=list(row))
                            w.writeheader()
                        w.writerow(row)
        print(f"merged {len(manifests)} manifests -> {merged_manifest}")
    if out is None:
        return
    wanted = None if clip_list is None else {l.strip() for l in open(clip_list) if l.strip()}
    refs, names, lengths = [], [], []
    for p in sorted(parts.glob("*.part.pt")):
        pack = torch.load(p, map_location="cpu", weights_only=False)
        for name, start, length in zip(pack["names"], pack["starts"], pack["lengths"]):
            if wanted is None or name in wanted:
                refs.append(pack["ref"][start : start + length])
                names.append(name)
                lengths.append(int(length))
    if wanted is not None and len(names) != len(wanted):
        missing = sorted(wanted - set(names))
        raise SystemExit(f"{len(missing)} listed clips are not in any part, e.g. {missing[:5]}")
    ref = torch.cat(refs, 0)
    starts = torch.cumsum(torch.tensor([0] + lengths[:-1]), 0).tolist()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ref": ref, "names": names, "starts": starts, "lengths": lengths}, out)
    print(f"saved {out}: {len(names)} clips, {ref.shape[0]} frames ({ref.shape[0] / 50 / 3600:.2f} h), {ref.numel() * 2 / 1e9:.2f} GB fp16")


if __name__ == "__main__":
    tyro.cli(main)
