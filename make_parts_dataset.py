"""Build the standardized 500-part benchmark dataset.

New test methodology (2024 redesign):

  * From each of the 10 mechcad categories, pick 10 parts whose file size is
    closest to that category's MEDIAN, subject to two hard filters:
      1. single body   (HOOPSModel.get_body_count() == 1)
      2. encodes cleanly through the real Step-1 flow (a graph .pt is produced)
  * Copy every selected part 5x under a deterministic name, giving
      10 categories x 10 parts x 5 copies = 500 files
    in bench_parts/ (git-ignored). Because the copies are byte-identical, every
    configuration encodes exactly the same geometry, so wall-time differences
    are purely a function of the parameter under test.
  * Emit filelists/parts500.txt (round-robin interleave across categories so the
    worker pool sees a balanced mix) and filelists/parts_smoke.txt (1 per cat).

Run with the GPU venv python (it can load + encode); no GPU is actually used
for selection.

    C:\\SDK\\HOOPS_AI\\GPU1.1\\.venv\\Scripts\\python.exe make_parts_dataset.py

Selection is deterministic and idempotent: re-running reproduces the same 100
parts and overwrites bench_parts/ + the filelists.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from bench_common import BENCH_ROOT, apply_license, require_license

CATEGORIES = ["bearing", "bolt", "bracket", "coupling", "flange",
              "gear", "nut", "pulley", "screw", "shaft"]


def log(msg: str) -> None:
    print(msg, flush=True)


def pick_single_body_candidates(cat_dir: Path, loader, want: int,
                                scan_cap: int) -> list[Path]:
    """Return up to `want` single-body parts nearest this category's median size.

    Files are sorted by size; we walk outward from the median index and load
    each candidate to check the body count, stopping once we have `want`
    single-body parts or exhaust `scan_cap` load attempts.
    """
    files = sorted((p for p in cat_dir.glob("*.stp") if p.is_file()),
                   key=lambda p: p.stat().st_size)
    if not files:
        return []
    mid = len(files) // 2
    # median-first ordering: mid, mid+1, mid-1, mid+2, mid-2, ...
    order = [mid]
    step = 1
    while len(order) < len(files):
        for nxt in (mid + step, mid - step):
            if 0 <= nxt < len(files):
                order.append(nxt)
        step += 1

    picked: list[Path] = []
    scanned = 0
    for idx in order:
        if len(picked) >= want or scanned >= scan_cap:
            break
        f = files[idx]
        scanned += 1
        try:
            model = loader.create_from_file(str(f))
            if model.get_body_count() == 1:
                picked.append(f)
        except Exception as exc:  # noqa: BLE001 - a bad file is simply skipped
            log(f"    skip (load error) {f.name}: {exc!r}"[:160])
    log(f"  {cat_dir.name}: {len(picked)} single-body candidates "
        f"(scanned {scanned}, median size "
        f"{files[mid].stat().st_size/1e6:.3f} MB)")
    return picked


def validate_encoding(candidates: list[Path], out_dir: Path,
                      max_workers: int) -> set[str]:
    """Run the real Step-1 flow over `candidates`; return the set of paths (str)
    that produced a graph .pt (i.e. encoded without error)."""
    from hoops_ai.storage.helpers import generate_unique_id_from_path

    flow_name = "SELECT_validate_encode"
    cand_list = out_dir / "_candidates.txt"
    cand_list.write_text("\n".join(str(p) for p in candidates) + "\n",
                         encoding="utf-8")

    # bench_tasks reads these at import time.
    os.environ["BENCH_FILELIST"] = str(cand_list)
    os.environ["BENCH_OUT_DIR"] = str(out_dir)
    os.environ["BENCH_FLOW_NAME"] = flow_name

    import hoops_ai
    from bench_tasks import (encode_data_for_ml_training, flows_outputdir,
                             gather_cad_files)

    graph_dir = flows_outputdir / "flows" / flow_name / "graph_data"
    log(f"  validating {len(candidates)} candidates via real encode "
        f"(max_workers={max_workers}) ...")
    flow = hoops_ai.create_flow(
        name=flow_name,
        tasks=[gather_cad_files, encode_data_for_ml_training],
        max_workers=max_workers,
        flows_outputdir=str(flows_outputdir),
        ml_task="Private HOOPS Embedings Model",
        export_visualization=False,
    )
    t = time.time()
    flow.process(inputs={"cad_datasources": [str(BENCH_ROOT)]},
                 clean_ouput_dir=True)
    log(f"  encode validation finished in {time.time()-t:.1f}s")

    ok: set[str] = set()
    for p in candidates:
        hid = generate_unique_id_from_path(str(p.with_suffix("")))
        # single-body parts emit exactly one graph, named "<hid>_0.pt"
        # (the trailing index is the per-body number).
        if list(graph_dir.glob(f"{hid}_*.pt")) or (graph_dir / f"{hid}.pt").is_file():
            ok.add(str(p))
    log(f"  {len(ok)}/{len(candidates)} candidates encoded cleanly")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechcad", default=r"C:\SDK\HOOPS_AI\mechcad",
                    help="root dir holding the 10 category subfolders")
    ap.add_argument("--per-cat", type=int, default=10,
                    help="final single-body parts kept per category")
    ap.add_argument("--copies", type=int, default=5)
    ap.add_argument("--candidate-slack", type=int, default=4,
                    help="extra candidates per category to survive encode "
                         "failures (want = per-cat + slack)")
    ap.add_argument("--scan-cap", type=int, default=60,
                    help="max files to load per category while hunting for "
                         "single-body candidates near the median")
    ap.add_argument("--validate-workers", type=int, default=8)
    ap.add_argument("--out-parts", default=str(BENCH_ROOT / "bench_parts"))
    args = ap.parse_args()

    apply_license()

    mechcad = Path(args.mechcad)
    if not mechcad.is_dir():
        sys.exit(f"mechcad root not found: {mechcad}")

    parts_dir = Path(args.out_parts)
    work_dir = BENCH_ROOT / "out" / "_select"
    work_dir.mkdir(parents=True, exist_ok=True)

    from hoops_ai.cadaccess import HOOPSLoader
    loader = HOOPSLoader()

    # ---- phase A: single-body candidates near the median (per category) -----
    want = args.per_cat + args.candidate_slack
    candidates: dict[str, list[Path]] = {}
    all_candidates: list[Path] = []
    log("[select] phase A: single-body median candidates")
    for cat in CATEGORIES:
        cat_dir = mechcad / cat
        if not cat_dir.is_dir():
            sys.exit(f"category folder missing: {cat_dir}")
        picked = pick_single_body_candidates(cat_dir, loader, want, args.scan_cap)
        if len(picked) < args.per_cat:
            sys.exit(f"category {cat}: only {len(picked)} single-body candidates "
                     f"found (need {args.per_cat}); raise --scan-cap")
        candidates[cat] = picked
        all_candidates.extend(picked)

    # ---- phase B: validate encoding through the real flow --------------------
    log("[select] phase B: encode validation")
    ok_paths = validate_encoding(all_candidates, work_dir, args.validate_workers)

    # ---- phase C: keep the first per-cat clean parts (still median-ordered) --
    log("[select] phase C: final selection")
    selected: dict[str, list[Path]] = {}
    for cat in CATEGORIES:
        clean = [p for p in candidates[cat] if str(p) in ok_paths]
        if len(clean) < args.per_cat:
            sys.exit(f"category {cat}: only {len(clean)} parts encoded cleanly "
                     f"(need {args.per_cat}); raise --candidate-slack")
        selected[cat] = clean[:args.per_cat]
        log(f"  {cat}: {len(selected[cat])} parts selected")

    # ---- phase D: copy x5 + build filelists ---------------------------------
    log("[select] phase D: copy x5 + filelists")
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)

    # copies grouped so round-robin can interleave categories evenly
    per_copy_rows: list[list[str]] = [[] for _ in range(args.copies)]
    manifest = {"mechcad": str(mechcad), "per_cat": args.per_cat,
                "copies": args.copies, "categories": {}}
    total = 0
    for cat in CATEGORIES:
        cat_out = parts_dir / cat
        cat_out.mkdir()
        cat_files: list[str] = []
        for i, src in enumerate(selected[cat]):
            stem = src.stem
            for c in range(args.copies):
                dst = cat_out / f"{cat}_{stem}_c{c+1}.stp"
                shutil.copyfile(src, dst)
                per_copy_rows[c].append(str(dst))
                cat_files.append(str(dst))
                total += 1
        manifest["categories"][cat] = {
            "sources": [str(p) for p in selected[cat]],
            "n_copies": len(cat_files),
        }

    # round-robin: all c1 copies (one per cat, interleaved), then c2, ...
    interleaved: list[str] = []
    for copy_rows in per_copy_rows:
        interleaved.extend(copy_rows)

    fl_dir = BENCH_ROOT / "filelists"
    fl_dir.mkdir(exist_ok=True)
    (fl_dir / "parts500.txt").write_text("\n".join(interleaved) + "\n",
                                         encoding="utf-8")
    # smoke: first selected part of each category (1 copy)
    smoke = [str(parts_dir / cat / f"{cat}_{selected[cat][0].stem}_c1.stp")
             for cat in CATEGORIES]
    (fl_dir / "parts_smoke.txt").write_text("\n".join(smoke) + "\n",
                                            encoding="utf-8")

    (parts_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    log(f"[select] DONE: {total} files in {parts_dir}")
    log(f"[select] filelists/parts500.txt ({len(interleaved)} paths)")
    log(f"[select] filelists/parts_smoke.txt ({len(smoke)} paths)")


if __name__ == "__main__":
    main()
