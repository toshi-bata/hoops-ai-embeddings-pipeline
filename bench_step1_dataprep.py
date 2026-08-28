"""Step 1 benchmark: CAD encoding (demo_HOOPS_EMBEDDINGS_DataPrep.ipynb).

Measures hoops_ai.create_flow(...).process() for one value of max_workers and
one file list. The flow itself records a per-task breakdown in its .flow file
(GatherCADFiles / EncodingTask / AutoDatasetExportTask), which is what we
actually care about -- EncodingTask is the parallel part.

Reference point from the existing GPU1.1 run (500 files, max_workers=12):
    total 1541.85 s = gather 54.7 + encode 1465.2 + export 21.9

Example:
    python bench_step1_dataprep.py --env-tag gpu1.1 --max-workers 12 \
        --filelist filelists/sub100.txt --keep-output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from bench_common import (BENCH_ROOT, REPO_ROOT, CAD_EXTENSIONS, PeakSampler, Timer,
                          check_hoops_ai_importable, discover_cad_files, freeze_filelist,
                          make_run_id, parse_extensions, peak_gpu_mb, probe_hardware,
                          require_license, write_row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-tag", required=True, help="cpu1.1 | gpu1.1")
    ap.add_argument("--max-workers", type=int, required=True)
    ap.add_argument("--filelist", default=None,
                    help="pre-built, frozen file list (one path per line). If "
                         "omitted, --source-dir is scanned recursively for CAD "
                         "files (see --extensions) and the result is frozen to "
                         "filelists/<env-tag>_discovered.txt so repeated runs "
                         "process the same files in the same order. Sweeps "
                         "that need reproducibility across many configurations "
                         "should still pass an explicit, reused --filelist "
                         "(see run_benchmark.ps1).")
    ap.add_argument("--extensions", default=None,
                    help="comma-separated list of file extensions to pick up "
                         "when scanning --source-dir, e.g. '.stp,.catpart'. "
                         f"Default: {','.join(sorted(CAD_EXTENSIONS))}. Only "
                         "used when --filelist is omitted.")
    ap.add_argument("--phase", default="1")
    ap.add_argument("--out-dir", default=None,
                    help="flow output root (default benchmark/out/<env-tag>)")
    ap.add_argument("--flow-name", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--source-dir", default=str(REPO_ROOT / "screw"))
    ap.add_argument("--keep-output", action="store_true",
                    help="keep the encoded dataset (needed for step 2/3). "
                         "Without it the flow still writes to disk but the name "
                         "is a scratch one that later runs overwrite.")
    ap.add_argument("--time-limit-s", type=float, default=None,
                    help="per-file encode time limit (create_flow "
                         "parallel_task_kwargs). Default flow limit is 120s, "
                         "which the heavy assembly needs raised.")
    ap.add_argument("--pointer-tag", default=None,
                    help="suffix for the dataset pointer file "
                         "(results/dataset_<tag>.json). Defaults to env-tag; "
                         "use e.g. gpu1.1_heavy to avoid clobbering the parts "
                         "dataset.")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    check_hoops_ai_importable()
    require_license()

    if args.filelist:
        filelist = Path(args.filelist)
        if not filelist.is_absolute():
            filelist = BENCH_ROOT / filelist
        if not filelist.is_file():
            sys.exit(f"filelist not found: {filelist}")
        # utf-8-sig tolerates the BOM that PowerShell 5.1 writes with -Encoding UTF8
        n_files = len([l for l in filelist.read_text(encoding="utf-8-sig").splitlines()
                       if l.strip()])
    else:
        source_dir = Path(args.source_dir)
        if not source_dir.is_dir():
            sys.exit(f"--source-dir not found: {source_dir}")
        files = discover_cad_files(source_dir, parse_extensions(args.extensions))
        if not files:
            sys.exit(f"no CAD files found under {source_dir} (recursively)")
        filelist = freeze_filelist(
            files, BENCH_ROOT / "filelists" / f"{args.env_tag}_discovered.txt")
        n_files = len(files)
        print(f"[step1] discovered {n_files} CAD files under {source_dir} -> {filelist}",
              flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else BENCH_ROOT / "out" / args.env_tag
    flow_name = args.flow_name or (
        f"KEEP_dataprep_{args.env_tag}_n{n_files}" if args.keep_output
        else f"SWEEP_dataprep_{args.env_tag}_mw{args.max_workers}_n{n_files}")

    # bench_tasks reads these at import time, so they must be set first.
    os.environ["BENCH_FILELIST"] = str(filelist)
    os.environ["BENCH_OUT_DIR"] = str(out_dir)
    os.environ["BENCH_FLOW_NAME"] = flow_name

    import hoops_ai  # noqa: E402  (after env is set)
    from bench_tasks import (encode_data_for_ml_training,  # noqa: E402
                             flows_outputdir, gather_cad_files)

    print(f"[step1] env={args.env_tag} max_workers={args.max_workers} n={n_files} "
          f"flow={flow_name}", flush=True)
    print(f"[step1] output -> {flows_outputdir / 'flows' / flow_name}", flush=True)

    row = {
        "run_id": make_run_id("dataprep", args.env_tag, "max_workers",
                              args.max_workers, n_files),
        "timestamp": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "phase": args.phase,
        "step": "dataprep",
        "env_tag": args.env_tag,
        "accelerator": "-",
        "n_files": n_files,
        "param_name": "max_workers",
        "param_value": args.max_workers,
        "extra": {"flow_name": flow_name, "filelist": filelist.name,
                  "keep_output": args.keep_output},
        "note": args.note,
    }

    try:
        create_kwargs = dict(
            name=flow_name,
            tasks=[gather_cad_files, encode_data_for_ml_training],
            max_workers=args.max_workers,
            flows_outputdir=str(flows_outputdir),
            ml_task="Private HOOPS Embedings Model",
            export_visualization=False,
        )
        if args.time_limit_s is not None:
            tl = float(args.time_limit_s)
            create_kwargs["parallel_task_kwargs"] = {
                "time_limit_overall": tl,
                "time_limit_small": tl,
                "time_limit_medium": tl,
                "time_limit_large": tl,
            }
            print(f"[step1] per-file time_limit={tl}s (overall/small/medium/large)", flush=True)
        data_prep = hoops_ai.create_flow(**create_kwargs)
        # Sample memory DURING the run: the encoder workers are separate
        # processes that exit before process() returns, so measuring afterwards
        # reports an empty process tree.
        with PeakSampler() as sampler, Timer() as t:
            flow_output, output_dict, flow_file = data_prep.process(
                inputs={"cad_datasources": [args.source_dir]},
                clean_ouput_dir=True,
            )
        wall = t.elapsed

        # The .flow JSON holds the authoritative per-task breakdown.
        sub, n_failed = {}, 0
        try:
            meta = json.loads(Path(flow_file).read_text(encoding="utf-8"))
            sub = meta.get("Duration [seconds]", {})
            n_failed = sum(meta.get("error_distribution", {}).values())
        except Exception as exc:
            sub = {"_flow_file_error": repr(exc)}

        encode_s = sub.get("EncodingTask") or wall
        row.update({
            "wall_s": round(wall, 2),
            "sub_timings": {**sub, **sampler.as_dict()},
            "throughput": round(n_files / encode_s, 3) if encode_s else "",
            "peak_rss_mb": sampler.peak_rss_mb,
            "peak_gpu_mb": peak_gpu_mb(),
            "n_ok": n_files - n_failed,
            "n_failed": n_failed,
            "status": "OK",
        })
        print(f"[step1] done in {wall:.1f}s (encode {encode_s:.1f}s, "
              f"{n_files / encode_s:.2f} files/s), failures={n_failed}", flush=True)
    except Exception as exc:
        traceback.print_exc()
        row.update({"wall_s": "", "status": "FAILED",
                    "note": (args.note + " | " + repr(exc))[:500]})

    write_row(row, args.csv)

    # Record the artifact paths so step 2/3 can find them without guessing.
    if args.keep_output and row["status"] == "OK":
        root = flows_outputdir / "flows" / flow_name
        pointer_tag = args.pointer_tag or args.env_tag
        pointer = BENCH_ROOT / "results" / f"dataset_{pointer_tag}.json"
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(json.dumps({
            "env_tag": args.env_tag,
            "flow_name": flow_name,
            "flow_root": str(root),
            "dataset": str(root / f"{flow_name}.dataset"),
            "infoset": str(root / f"{flow_name}.infoset"),
            "n_files": n_files,
            "max_workers": args.max_workers,
            "hardware": probe_hardware(),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[step1] dataset pointer -> {pointer}", flush=True)


if __name__ == "__main__":
    main()
