#!/usr/bin/env python3
"""Deterministically pick an evenly-strided subset of a portable file list.

Used to carve a smaller, representative sample out of the ~10k heavy file list
for the Step 1 / Step 3 worker-scaling sweep, without paying the full-corpus
cost at every worker count.

Why an even stride (not head, not random)
  The heavy file list is grouped by category (bearing/, bolt/, ...) and, within
  a category, spans the full file-size range. Taking every k-th line therefore
  preserves BOTH the category proportions and the size distribution, so the
  scaling curve has the same shape as the full run. `head` would bias to the
  first categories; a random sample is not reproducible without pinning a seed.

The selection is a pure function of (input list, target count), so the same
subset is reproducible on any machine.

Usage
  python make_subset_filelist.py --in filelists/mech_heavy_aws.txt \
      --out filelists/mech_2k.txt --count 2000
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def stride_subset(lines: list[str], count: int) -> list[str]:
    """Return `count` evenly-spaced entries (order preserved, de-duplicated)."""
    n = len(lines)
    if count >= n:
        return list(lines)
    idx = sorted(set(round(i * n / count) for i in range(count)))
    return [lines[i] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="source file list")
    ap.add_argument("--out", required=True, help="destination file list")
    ap.add_argument("--count", type=int, default=2000, help="target number of files")
    args = ap.parse_args()

    src = Path(args.inp)
    if not src.is_file():
        raise SystemExit(f"input not found: {src}")
    # utf-8-sig tolerates the BOM PowerShell 5.1 writes with -Encoding UTF8.
    lines = [l.strip().lstrip("\ufeff")
             for l in src.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    sub = stride_subset(lines, args.count)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sub) + "\n", encoding="utf-8")

    cats = Counter(l.replace("\\", "/").split("/")[0] for l in sub)
    print(f"wrote {len(sub)} files -> {out}  (from {len(lines)})")
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    main()
