"""Step 3 benchmark: embedding + FAISS indexing (demo_HOOPS_Embeddings_indexing.ipynb).

Splits the pipeline into three separately-timed pieces, because they scale
completely differently:

  embed_shape_batch()   CPU-parallel B-rep encoding + model forward -> scales with num_workers
  CADSearch.index_shape()  FAISS add                                -> ~O(n), single-threaded
  save_shape_index()       disk write                               -> ~O(n)

`generate_images` is OFF by default: rendering PNGs per part is a large,
constant, non-ML cost that would mask the num_workers effect. Phase 4 runs it
once with --gen-images so the overhead is quantified separately.
"""
from __future__ import annotations

import argparse
import datetime
import inspect
import os
import subprocess
import sys
import traceback
from pathlib import Path

from bench_common import (BENCH_ROOT, CAD_EXTENSIONS, PeakSampler, Timer, apply_license,
                          check_hoops_ai_importable, discover_cad_files, freeze_filelist,
                          make_run_id, parse_extensions, peak_gpu_mb, require_license,
                          resolve_checkpoint, write_row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-tag", required=True, help="cpu1.1 | gpu1.1")
    ap.add_argument("--num-workers", type=int, required=True)
    ap.add_argument("--filelist", default=None,
                    help="pre-built, frozen file list (one path per line). If "
                         "omitted, --source-dir is scanned recursively for CAD "
                         "files (see --extensions) and the result is frozen to "
                         "filelists/<env-tag>_discovered.txt.")
    ap.add_argument("--source-dir", default=None,
                    help="folder to scan recursively when --filelist is omitted")
    ap.add_argument("--extensions", default=None,
                    help="comma-separated list of file extensions to pick up "
                         "when scanning --source-dir, e.g. '.stp,.catpart'. "
                         f"Default: {','.join(sorted(CAD_EXTENSIONS))}. Only "
                         "used when --filelist is omitted.")
    ap.add_argument("--model-name", default="HOOPS Embeddings SIGNAL preview")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--gen-images", action="store_true",
                    help="also render a PNG per part (measures the extra cost)")
    ap.add_argument("--save-index", action="store_true",
                    help="write the .faiss/.meta to disk (adds an I/O phase)")
    ap.add_argument("--index-name", default=None,
                    help="filename (without extension) for the saved index "
                         "when --save-index is set, e.g. 'acme_gearbox' -> "
                         "acme_gearbox.faiss + acme_gearbox.meta. This is what "
                         "a qt_sandbox-style desktop client opens directly, so "
                         "name it something you'll recognize later. Default: "
                         "'<env-tag>_n<n_files>'.")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="per-file embed time limit (seconds). Sets the "
                         "embed_shape_batch time_limit_* specs; the heavy "
                         "assembly needs this raised well above the default.")
    ap.add_argument("--phase", default="4")
    ap.add_argument("--csv", default=None)
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
        raw = [l.strip().lstrip("﻿")
               for l in filelist.read_text(encoding="utf-8-sig").splitlines()
               if l.strip()]
        # Portable file lists store paths RELATIVE to the corpus root so the same
        # list works on Windows and on the AWS Ubuntu box (mirrors Step 1's
        # gather_cad_files): set BENCH_FILE_ROOT to wherever the mechcad tree lives.
        # Absolute entries are used as-is.
        file_root = os.environ.get("BENCH_FILE_ROOT", "").strip()
        cad_files = []
        for entry in raw:
            p = Path(entry)
            if not p.is_absolute() and file_root:
                p = Path(file_root) / entry
            cad_files.append(str(p))
    else:
        if not args.source_dir:
            sys.exit("pass --filelist or --source-dir")
        source_dir = Path(args.source_dir)
        if not source_dir.is_dir():
            sys.exit(f"--source-dir not found: {source_dir}")
        cad_files = discover_cad_files(source_dir, parse_extensions(args.extensions))
        if not cad_files:
            sys.exit(f"no CAD files found under {source_dir} (recursively)")
        filelist = freeze_filelist(
            cad_files, BENCH_ROOT / "filelists" / f"{args.env_tag}_discovered.txt")
        print(f"[step3] discovered {len(cad_files)} CAD files under {source_dir} -> {filelist}",
              flush=True)

    n_files = len(cad_files)
    missing = [f for f in cad_files if not Path(f).is_file()]
    if missing:
        sys.exit(f"{len(missing)} of {n_files} files listed in {filelist} do not "
                 f"exist, first: {missing[0]!r}")

    # HOOPSEmbeddings picks its own device. If CUDA is present but this torch
    # build has no kernels for the card, that auto-selection would fail on every
    # file. Hide the GPU from torch entirely so the run is a clean, correctly
    # labelled CPU run instead of a pile of failures.
    #
    # This has to happen before torch is imported.
    cuda_masked = False
    if not os.environ.get("BENCH_NO_CUDA_MASK"):
        probe = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, r'%s');"
             "from bench_common import cuda_usable;"
             "ok, d = cuda_usable(); print('USABLE' if ok else 'UNUSABLE');"
             "print(d.get('gpu_capability', '')); print(d.get('cuda_error', ''))"
             % str(BENCH_ROOT)],
            capture_output=True, text=True, timeout=300)
        out = (probe.stdout or "").splitlines()
        if out and out[0].strip() == "UNUSABLE":
            cap = out[1].strip() if len(out) > 1 else "?"
            err = out[2].strip() if len(out) > 2 else ""
            print(f"[step3] CUDA present but unusable ({cap}): {err}", flush=True)
            print("[step3] masking the GPU (CUDA_VISIBLE_DEVICES='') - this run is CPU",
                  flush=True)
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            cuda_masked = True

    # Only now activate the license: apply_license imports hoops_ai, which pulls
    # in torch, and CUDA_VISIBLE_DEVICES has to be set before that happens.
    apply_license()

    import torch
    from hoops_ai.ml import CADSearch
    from hoops_ai.ml.embeddings import HOOPSEmbeddings

    out_dir = BENCH_ROOT / "out" / args.env_tag / "indexing"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Same name used for the saved index below, computed up front so
    # --gen-images can use it too: qt_sandbox's own "Add Folder" writes
    # per-part images (and, unconditionally, a matching .scs scene cache
    # file per part -- HOOPS AI generates the thumbnail image FROM an SCS
    # scene, so the scene file is kept alongside it) into a folder named
    # after the index, sitting directly next to <index_name>.faiss/.meta,
    # mirroring the source folder's relative paths underneath. Matching that
    # layout here means embed_shape_batch's own asset-generation path -- not
    # a separate step -- so this repo relies on --gen-images to reproduce it;
    # see the printed embed_shape_batch(...) signature below if you need to
    # confirm what today's hoops_ai build actually accepts/produces.
    index_name = args.index_name or f"{args.env_tag}_n{n_files}"
    images_out_dir = out_dir / index_name
    images_out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = resolve_checkpoint(args.ckpt)
    print(f"[step3] env={args.env_tag} num_workers={args.num_workers} n={n_files} "
          f"images={args.gen_images}", flush=True)
    print(f"[step3] checkpoint {ckpt}", flush=True)

    row = {
        "run_id": make_run_id("indexing", args.env_tag, "num_workers",
                              args.num_workers, n_files),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "phase": args.phase,
        "step": "indexing",
        "env_tag": args.env_tag,
        # Report the device actually used, not the venv it came from.
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "n_files": n_files,
        "param_name": "num_workers",
        "param_value": args.num_workers,
        "extra": {
            "gen_images": args.gen_images,
            "save_index": args.save_index,
            "filelist": filelist.name,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_masked": cuda_masked,
        },
        "note": (args.note + (" | GPU masked (unusable by this torch build)"
                              if cuda_masked else "")).strip(" |"),
    }

    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with Timer() as t_reg:
            HOOPSEmbeddings.register_model(model_name=args.model_name,
                                           checkpoint_path=str(ckpt))
            embedder = HOOPSEmbeddings(model=args.model_name)
        print(f"[step3] model ready in {t_reg.elapsed:.1f}s "
              f"(dim={embedder.embedding_dim})", flush=True)

        specs = {"generate_images": bool(args.gen_images),
                 "images_out_dir": images_out_dir}
        if args.time_limit is not None:
            specs.update({
                "time_limit_overall": args.time_limit,
                "time_limit_small": args.time_limit,
                "time_limit_medium": args.time_limit,
                "time_limit_large": args.time_limit,
            })
            print(f"[step3] per-file time_limit={args.time_limit}s", flush=True)

        # Log the real signature once -- the API is EXPERIMENTAL-adjacent and
        # keyword names have moved between releases.
        try:
            print(f"[step3] embed_shape_batch{inspect.signature(embedder.embed_shape_batch)}",
                  flush=True)
        except (TypeError, ValueError):
            pass

        with PeakSampler() as sampler, Timer() as t_embed:
            batch = embedder.embed_shape_batch(
                cad_files,
                num_workers=args.num_workers,
                show_progress=True,
                specifications=specs,
            )
        embed_s = t_embed.elapsed
        n_ok = len(batch.ids)
        n_failed = int(batch.metadata.get("failed_count", 0) or 0)

        with Timer() as t_index:
            searcher = CADSearch(shape_model=embedder)
            searcher.index_shape(batch)
        index_s = t_index.elapsed

        save_s = None
        if args.save_index:
            target = out_dir / f"{index_name}.faiss"
            with Timer() as t_save:
                searcher.save_shape_index(target)
            save_s = round(t_save.elapsed, 2)
            print(f"[step3] index -> {target}", flush=True)

        total = embed_s + index_s + (save_s or 0)
        row.update({
            "wall_s": round(total, 2),
            "sub_timings": {
                "model_load_s": round(t_reg.elapsed, 2),
                "embed_s": round(embed_s, 2),
                "faiss_index_s": round(index_s, 2),
                "save_s": save_s,
                "embed_dim": embedder.embedding_dim,
                **sampler.as_dict(),
            },
            "throughput": round(n_ok / embed_s, 3) if embed_s else "",
            "peak_rss_mb": sampler.peak_rss_mb,
            "peak_gpu_mb": peak_gpu_mb(),
            "n_ok": n_ok,
            "n_failed": n_failed,
            "status": "OK",
        })
        print(f"[step3] embed {embed_s:.1f}s ({n_ok / embed_s:.2f} files/s), "
              f"faiss {index_s:.2f}s, failed={n_failed}", flush=True)
    except Exception as exc:
        traceback.print_exc()
        row.update({"wall_s": "", "status": "FAILED",
                    "note": (args.note + " | " + repr(exc))[:500]})

    write_row(row, args.csv)


if __name__ == "__main__":
    main()
