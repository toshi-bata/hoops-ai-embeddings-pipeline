"""Build balanced CAD file lists from the mechcad corpus.

The mechcad tree has 10 part categories (bearing, bolt, bracket, coupling,
flange, gear, nut, pulley, screw, shaft), each with ~1000-1500 STEP files named
0.stp, 1.stp, ... plus a macOS .DS_Store that must be skipped.

"Balanced" here means: take the SAME number of parts from every category, and
within each category pick a size-stratified sample so the selection is
representative of that category's size range rather than the first N files.
Categories are then round-robin interleaved so that heavy and light parts are
spread evenly across the list -- otherwise a max_workers sweep would see all the
big files bunched at the end.

Outputs (benchmark/filelists/):
  mech500.txt   - <per-category> x 10 balanced set                 <- primary
  mech100.txt   - size-stratified sweep subset of mech500
  mech_smoke.txt- 1 per category (10 files) for the pre-flight smoke test
  manifest_mech.json - what was picked, with size stats per category
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bench_common import BENCH_ROOT

CATEGORIES = ["bearing", "bolt", "bracket", "coupling", "flange",
              "gear", "nut", "pulley", "screw", "shaft"]
CAD_SUFFIXES = (".stp", ".step", ".igs", ".iges")


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def stratified_by_size(files: list[Path], k: int) -> list[Path]:
    """Pick k files spread evenly across the whole size range (incl. the max)."""
    by_size = sorted(files, key=lambda p: (p.stat().st_size, natural_key(p)))
    m = min(k, len(by_size))
    if m <= 1:
        return by_size[:m]
    idx = sorted({round(i * (len(by_size) - 1) / (m - 1)) for i in range(m)})
    return [by_size[i] for i in idx]


def round_robin(groups: list[list[Path]]) -> list[Path]:
    """Interleave category groups so sizes are spread across the final order."""
    out: list[Path] = []
    for i in range(max((len(g) for g in groups), default=0)):
        for g in groups:
            if i < len(g):
                out.append(g[i])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=r"C:\SDK\HOOPS_AI\mechcad",
                    help="mechcad root holding the category sub-folders")
    ap.add_argument("--per-category", type=int, default=50,
                    help="parts to take from each category (default 50 -> 500 total)")
    ap.add_argument("--sweep-per-category", type=int, default=10,
                    help="parts per category for the sweep subset (default 10 -> 100)")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.is_dir():
        raise SystemExit(f"mechcad source not found: {src}")

    out_dir = BENCH_ROOT / "filelists"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_cat_full: dict[str, list[Path]] = {}
    stats: dict[str, dict] = {}
    missing_cats = []
    for cat in CATEGORIES:
        cdir = src / cat
        if not cdir.is_dir():
            missing_cats.append(cat)
            continue
        files = [p for p in cdir.iterdir()
                 if p.is_file() and p.suffix.lower() in CAD_SUFFIXES
                 and not p.name.startswith(".")]
        if not files:
            missing_cats.append(cat)
            continue
        picked = stratified_by_size(files, args.per_category)
        picked.sort(key=natural_key)
        per_cat_full[cat] = picked
        sizes_mb = [p.stat().st_size / 1024 ** 2 for p in picked]
        stats[cat] = {
            "available": len(files),
            "picked": len(picked),
            "min_mb": round(min(sizes_mb), 3),
            "mean_mb": round(sum(sizes_mb) / len(sizes_mb), 3),
            "max_mb": round(max(sizes_mb), 3),
        }
    if missing_cats:
        print(f"[mech] WARNING categories missing/empty: {missing_cats}")
    if not per_cat_full:
        raise SystemExit("no categories found under mechcad")

    # Primary set: round-robin interleave the categories.
    mech_full = round_robin([per_cat_full[c] for c in CATEGORIES if c in per_cat_full])

    # Sweep subset: size-stratified across the WHOLE pooled set (matches the
    # full-set size distribution, like the screw harness). Per-category
    # stratification over-weights each category's heavy tail and makes the sweep
    # subset ~2.5x heavier than the full set, which would let a few monster
    # parts dominate a num_workers comparison.
    per_cat_sweep = {}  # kept for manifest compatibility (unused)
    mech_sweep = stratified_by_size(list(mech_full), args.sweep_per_category * len(per_cat_full))
    mech_sweep.sort(key=natural_key)

    # Smoke: 1 (median-size) per category.
    smoke = []
    for c in CATEGORIES:
        if c in per_cat_full:
            fs = sorted(per_cat_full[c], key=lambda p: p.stat().st_size)
            smoke.append(fs[len(fs) // 2])

    def dump(name: str, files: list[Path]) -> None:
        path = out_dir / name
        path.write_text("\n".join(str(f) for f in files) + "\n", encoding="utf-8", newline="\n")
        total_mb = sum(f.stat().st_size for f in files) / 1024 ** 2
        print(f"[mech] {name:16s} {len(files):4d} files  {total_mb:8.1f} MB  -> {path}")

    dump(f"mech{len(mech_full)}.txt", mech_full)
    dump(f"mech{len(mech_sweep)}.txt", mech_sweep)
    dump("mech_smoke.txt", smoke)

    manifest = {
        "source": str(src),
        "per_category": args.per_category,
        "categories": [c for c in CATEGORIES if c in per_cat_full],
        "missing": missing_cats,
        "n_full": len(mech_full),
        "n_sweep": len(mech_sweep),
        "n_smoke": len(smoke),
        "per_category_stats": stats,
        "full_name": f"mech{len(mech_full)}.txt",
        "sweep_name": f"mech{len(mech_sweep)}.txt",
    }
    (out_dir / "manifest_mech.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[mech] manifest -> {out_dir / 'manifest_mech.json'}")


if __name__ == "__main__":
    main()
