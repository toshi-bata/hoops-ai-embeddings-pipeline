"""Build the fixed CAD file lists used by every benchmark run.

Why a fixed list instead of "point the flow at the folder":
  * 15 of the 500 screw files fail to encode (9 x 120 s worker timeout,
    6 x data-extraction error). The timeouts alone inject up to ~18 minutes of
    worker stall, which swamps a max_workers comparison.
  * A frozen, ordered list makes every configuration process *exactly* the same
    parts, so wall-clock differences are attributable to the knob under test.

Outputs (benchmark/filelists/):
  all500.txt    - every .stp in the source dir, sorted naturally
  clean485.txt  - all500 minus the 15 known-bad files      <- primary set
  sub100.txt    - deterministic stratified-by-size sample of clean485
  bad15.txt     - the excluded files, for the record
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bench_common import BENCH_ROOT, REPO_ROOT

# From GPU1.1/embeddings_pipeline/out/flows/HOOPS_Embedding_Training_GPU/error_summary.json
# (500 files, max_workers=12, HOOPS AI 1.1) -- 2026-07-29 baseline run.
KNOWN_BAD = {
    # 120 s cumulative worker timeout
    "490.stp", "141.stp", "82.stp", "459.stp", "361.stp",
    "306.stp", "84.stp", "224.stp", "388.stp",
    # push_face_attributes requires 'face_indices'
    "131.stp", "458.stp", "183.stp", "149.stp",
    # division by zero
    "297.stp", "293.stp",
}


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(REPO_ROOT / "screw"),
                    help="directory holding the 500 STEP files")
    ap.add_argument("--subset-size", type=int, default=100,
                    help="size of the sweep subset (default 100)")
    ap.add_argument("--error-summary", default=None,
                    help="optional error_summary.json to derive the bad list from "
                         "instead of the hard-coded one")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.is_dir():
        raise SystemExit(f"source dir not found: {src}")

    out_dir = BENCH_ROOT / "filelists"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(
        [p for p in src.iterdir()
         if p.suffix.lower() in (".stp", ".step", ".igs", ".iges")],
        key=natural_key)
    if not all_files:
        raise SystemExit(f"no CAD files found in {src}")

    bad = set(KNOWN_BAD)
    if args.error_summary:
        entries = json.loads(Path(args.error_summary).read_text(encoding="utf-8"))
        bad = {Path(e["item"]).name for e in entries}
        print(f"[filelist] bad list taken from {args.error_summary}: {len(bad)} files")

    clean = [p for p in all_files if p.name not in bad]
    excluded = [p for p in all_files if p.name in bad]

    # Deterministic subset: sort clean files by size, then pick indices spread
    # evenly across the WHOLE range (including the heaviest file). Slicing with
    # a fixed stride would truncate the tail and make the subset systematically
    # lighter than the full set, which would flatter the worker sweep.
    by_size = sorted(clean, key=lambda p: (p.stat().st_size, natural_key(p)))
    m = min(args.subset_size, len(by_size))
    if m <= 1:
        idx = [0]
    else:
        idx = sorted({round(i * (len(by_size) - 1) / (m - 1)) for i in range(m)})
    subset = [by_size[i] for i in idx]
    subset.sort(key=natural_key)

    def dump(name: str, files: list[Path]) -> Path:
        path = out_dir / name
        path.write_text("\n".join(str(f) for f in files) + "\n", encoding="utf-8")
        total_mb = sum(f.stat().st_size for f in files) / 1024 ** 2
        print(f"[filelist] {name:14s} {len(files):4d} files  {total_mb:8.1f} MB  -> {path}")
        return path

    dump("all500.txt", all_files)
    dump("clean485.txt", clean)
    dump(f"sub{len(subset)}.txt", subset)
    dump("bad15.txt", excluded)

    manifest = {
        "source_dir": str(src),
        "n_all": len(all_files),
        "n_clean": len(clean),
        "n_subset": len(subset),
        "excluded": sorted(p.name for p in excluded),
        "subset_files": [p.name for p in subset],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[filelist] manifest -> {out_dir / 'manifest.json'}")

    if len(clean) != 485:
        print(f"[filelist] NOTE: expected 485 clean files, got {len(clean)}. "
              "The primary set is still self-consistent across runs.")


if __name__ == "__main__":
    main()
