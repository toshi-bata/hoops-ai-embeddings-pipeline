"""Production pipeline runner: encode -> train -> index, ONCE, on a target CAD folder.

This is deliberately NOT a benchmark tool -- for parameter sweeps see
run_benchmark.ps1 (Windows/local) or run_heavy_batch.sh / run_heavy_scaling.sh
(AWS/Ubuntu). This script runs bench_step1_dataprep.py, bench_step2_training.py
and bench_step3_indexing.py exactly once each, in the right order, wiring the
dataset pointer from step 1 into step 2 and the freshly trained checkpoint from
step 2 into step 3 -- so the index it builds reflects the model you just
trained, not the shipped warm-start checkpoint.

The intended use case is training a custom model and building its index on a
GPU box (e.g. AWS EC2) instead of on a desktop machine, then handing the two
resulting files -- a checkpoint and a matching FAISS index, both named after
--run-name -- to a desktop client such as qt_sandbox. Both are copied to
flat, predictable paths under out/<run-name>/ for exactly that handoff; see
the final "Deliverables" printout after a run.

Example (AWS GPU box, training from scratch on a customer corpus):
    python run_pipeline.py --run-name acme_gearbox --source-dir ~/data/acme_parts \\
        --accelerator gpu --workers 16 --epochs 10 --batch-size 32 --no-warm-start

Example (CPU-only, warm-started fine-tune, small folder):
    python run_pipeline.py --run-name demo_run --source-dir ./my_parts \\
        --accelerator cpu --workers 8 --epochs 3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bench_common import (BENCH_ROOT, CAD_EXTENSIONS, check_hoops_ai_importable,
                          discover_cad_files, freeze_filelist, parse_extensions)


def discover_filelist(source_dir: Path, run_name: str, extensions: set[str] | None) -> Path:
    """Scan source_dir recursively and freeze the result to a file list, so
    step 1 and step 3 process the exact same files in the exact same order
    even if the folder's contents change partway through a long run."""
    files = discover_cad_files(source_dir, extensions)
    if not files:
        sys.exit(f"no CAD files found under {source_dir} (recursively)")
    listing = freeze_filelist(files, BENCH_ROOT / "filelists" / f"{run_name}.txt")
    print(f"[pipeline] discovered {len(files)} CAD files -> {listing}", flush=True)
    return listing


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[pipeline] {label} failed (exit {result.returncode}); aborting.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True,
                     help="tag for this run; used for the dataset/checkpoint "
                          "pointer files, output folders and the FAISS index name")
    ap.add_argument("--source-dir", required=True, type=Path,
                     help="folder of CAD files to encode, train on and index "
                          "(scanned recursively; see --extensions)")
    ap.add_argument("--filelist", type=Path, default=None,
                     help="use this pre-built file list instead of scanning --source-dir")
    ap.add_argument("--extensions", default=None,
                     help="comma-separated list of file extensions to pick up "
                          "when scanning --source-dir, e.g. '.stp,.catpart'. "
                          f"Default: {','.join(sorted(CAD_EXTENSIONS))}.")
    ap.add_argument("--accelerator", choices=["cpu", "gpu"], default="gpu",
                     help="device for step 2 training (default gpu)")
    ap.add_argument("--workers", type=int, default=8,
                     help="worker count for step 1 (max_workers) and step 3 (num_workers)")
    ap.add_argument("--time-limit", type=float, default=None,
                     help="per-file time limit in seconds, applied to step 1 and step 3")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--train-num-workers", type=int, default=0,
                     help="DataLoader worker count for step 2 (default 0)")
    ap.add_argument("--epochs", type=int, default=10,
                     help="new epochs to train past the warm-start checkpoint "
                         "(or from scratch if --no-warm-start)")
    ap.add_argument("--no-warm-start", action="store_true",
                     help="train from scratch instead of the shipped SIGNAL checkpoint")
    ap.add_argument("--base-ckpt", default=None,
                     help="warm-start checkpoint for step 2 (default: auto-resolve "
                         "the shipped ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt)")
    ap.add_argument("--gen-images", action=argparse.BooleanOptionalAction, default=True,
                     help="render a PNG (and, per hoops_ai, an accompanying .scs scene "
                         "cache) per part during step 3 indexing, in a "
                         "<run-name>/ folder next to the saved index -- this matches "
                         "what qt_sandbox's own \"Add Folder\" produces, which is the "
                         "point of this default being on. Pass --no-gen-images to skip "
                         "it for a faster run if you don't need a qt_sandbox-compatible "
                         "index.")
    args = ap.parse_args()

    # Fail fast, before running any of the three steps: they run as
    # subprocesses of THIS interpreter (sys.executable below), so if hoops_ai
    # isn't importable here it won't be importable in any of them either.
    check_hoops_ai_importable()

    if not args.source_dir.is_dir():
        sys.exit(f"--source-dir not found: {args.source_dir}")

    filelist = args.filelist or discover_filelist(
        args.source_dir, args.run_name, parse_extensions(args.extensions))

    # ---- step 1: encode -------------------------------------------------
    step1 = [sys.executable, str(BENCH_ROOT / "bench_step1_dataprep.py"),
             "--env-tag", args.run_name, "--max-workers", str(args.workers),
             "--filelist", str(filelist), "--source-dir", str(args.source_dir),
             "--pointer-tag", args.run_name, "--keep-output",
             "--phase", "prod", "--note", "run_pipeline"]
    if args.time_limit is not None:
        step1 += ["--time-limit-s", str(args.time_limit)]
    run_step("step 1/3 - encoding", step1)

    # ---- step 2: train ---------------------------------------------------
    step2 = [sys.executable, str(BENCH_ROOT / "bench_step2_training.py"),
             "--env-tag", args.run_name, "--accelerator", args.accelerator,
             "--batch-size", str(args.batch_size),
             "--num-workers", str(args.train_num_workers),
             "--new-epochs", str(args.epochs),
             "--dataset-pointer", str(BENCH_ROOT / "results" / f"dataset_{args.run_name}.json"),
             "--phase", "prod", "--note", "run_pipeline"]
    if args.no_warm_start:
        step2.append("--no-warm-start")
    if args.base_ckpt:
        step2 += ["--ckpt", args.base_ckpt]
    run_step("step 2/3 - training", step2)

    # bench_step2_training.py already copies the checkpoint out of Lightning's
    # deeply nested, timestamped experiment folder into a flat path named
    # after --env-tag (== --run-name here); just read that pointer.
    ckpt_pointer_path = BENCH_ROOT / "results" / f"checkpoint_{args.run_name}.json"
    if not ckpt_pointer_path.is_file():
        sys.exit(f"[pipeline] step 2 did not produce {ckpt_pointer_path}; cannot run step 3.")
    trained_ckpt = Path(json.loads(ckpt_pointer_path.read_text(encoding="utf-8"))["checkpoint"])
    deliverable_dir = BENCH_ROOT / "out" / args.run_name

    # ---- step 3: index, using the model just trained in step 2 ----------
    step3 = [sys.executable, str(BENCH_ROOT / "bench_step3_indexing.py"),
             "--env-tag", args.run_name, "--num-workers", str(args.workers),
             "--filelist", str(filelist), "--ckpt", str(trained_ckpt),
             "--index-name", args.run_name, "--save-index",
             "--phase", "prod", "--note", "run_pipeline"]
    if args.time_limit is not None:
        step3 += ["--time-limit", str(args.time_limit)]
    if args.gen_images:
        step3.append("--gen-images")
    run_step("step 3/3 - indexing", step3)

    index_path = deliverable_dir / "indexing" / f"{args.run_name}.faiss"
    print(f"\n[pipeline] done. Deliverables for a desktop client (e.g. qt_sandbox):", flush=True)
    print(f"  checkpoint : {trained_ckpt}", flush=True)
    print(f"  index      : {index_path}  (+ matching .meta alongside it)", flush=True)
    print("[pipeline] copy both to the desktop machine: load the checkpoint first, then", flush=True)
    print("           open the index -- they must be loaded as a matched pair, since", flush=True)
    print("           embeddings from a different model are not comparable.", flush=True)
    print("[pipeline] one row per step was appended to results/results.csv", flush=True)


if __name__ == "__main__":
    main()
