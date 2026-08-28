"""Step 2 benchmark: embedding-model training (demo_HOOPS_EMBEDDINGS_training.ipynb).

Measures FlowTrainer.train() for one (accelerator, batch_size, num_workers,
max_epochs) combination against an already-encoded dataset produced by
bench_step1_dataprep.py --keep-output.

Two things are pinned so that runs are comparable:
  * early_stopping=False  -> every run does exactly max_epochs epochs.
    (FlowTrainer defaults to True, which would make wall-clock depend on when
    the loss happens to plateau.)
  * train_shuffle=False + train_seed  -> same batch order every time.

The headline metric is s/epoch, not total wall, because trainer start-up
(checkpoint load, CUDA context, dataloader spin-up) is a fixed cost that would
otherwise distort short runs. Both are recorded.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import traceback
from pathlib import Path

from bench_common import (BENCH_ROOT, PeakSampler, Timer, apply_license,
                          check_hoops_ai_importable, cuda_usable, make_run_id, peak_gpu_mb,
                          resolve_checkpoint, write_row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-tag", required=True, help="cpu1.1 | gpu1.1")
    ap.add_argument("--accelerator", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader worker processes (FlowTrainer default is 0)")
    ap.add_argument("--new-epochs", type=int, default=3,
                    help="number of ADDITIONAL epochs to train past the resumed "
                         "checkpoint. This is the benchmark-relevant quantity; "
                         "max_epochs is derived from it (default 3)")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="absolute Trainer(max_epochs=...). Overrides --new-epochs. "
                         "Must exceed the checkpoint's epoch when warm starting.")
    ap.add_argument("--matmul-precision", default="high", choices=["high", "highest"])
    ap.add_argument("--dataset-pointer", default=None,
                    help="results/dataset_<env>.json from step 1 (default: match --env-tag)")
    ap.add_argument("--ckpt", default=None,
                    help="warm-start checkpoint; default = auto-resolve "
                         "ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt")
    ap.add_argument("--no-warm-start", action="store_true",
                    help="train from scratch instead of resuming the 2M SIGNAL ckpt")
    ap.add_argument("--run-test", action="store_true",
                    help="also time flow_trainer.test() on the best checkpoint")
    ap.add_argument("--phase", default="3")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    check_hoops_ai_importable()
    apply_license()

    import torch
    torch.set_float32_matmul_precision(args.matmul_precision)
    import warnings
    warnings.filterwarnings("ignore")

    if args.accelerator == "gpu":
        if not torch.cuda.is_available():
            sys.exit(f"--accelerator gpu requested but torch.cuda.is_available() is "
                     f"False (torch {torch.__version__}). Use the GPU1.1 venv.")
        # is_available() is not enough: it lies when the wheel has no kernels for
        # this compute capability. Check before burning a training run.
        ok, detail = cuda_usable()
        if not ok:
            print("[step2] GPU is present but NOT usable by this torch build:",
                  flush=True)
            for k in ("gpu_capability", "torch_arch_list", "cuda_error", "cuda_hint"):
                if detail.get(k):
                    print(f"         {k}: {detail[k]}", flush=True)
            sys.exit(2)

    pointer_path = Path(args.dataset_pointer) if args.dataset_pointer else \
        BENCH_ROOT / "results" / f"dataset_{args.env_tag}.json"
    if not pointer_path.is_file():
        sys.exit(f"dataset pointer not found: {pointer_path}\n"
                 "Run phase 2 (bench_step1_dataprep.py --keep-output) first.")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    dataset, infoset = pointer["dataset"], pointer["infoset"]
    flow_root = Path(pointer["flow_root"])
    n_files = pointer["n_files"]
    for p in (dataset, infoset):
        if not Path(p).exists():
            sys.exit(f"missing artifact: {p}")

    import hoops_ai  # noqa: F401  (license already activated by apply_license above)
    from hoops_ai.dataset import DatasetLoader
    from hoops_ai.ml.EXPERIMENTAL import EmbeddingFlowModel, FlowTrainer

    # ------------------------------------------------------------------
    # Resolve the epoch budget.
    #
    # PyTorch Lightning restores the epoch COUNTER from a resumed checkpoint and
    # refuses to run if max_epochs <= that counter:
    #   "You restored a checkpoint with current_epoch=7, but you have set
    #    Trainer(max_epochs=1)."
    # The shipped SIGNAL-preview checkpoint sits at epoch 7, which is why the
    # demo notebook uses max_epochs=10 - that is 3 NEW epochs, not 10.
    #
    # So the benchmark works in "new epochs" and derives max_epochs from the
    # checkpoint. It also means s/epoch must be divided by the epochs actually
    # executed, not by max_epochs.
    # ------------------------------------------------------------------
    ckpt_path = None if args.no_warm_start else resolve_checkpoint(args.ckpt)
    ckpt_epoch = 0
    if ckpt_path is not None:
        try:
            meta = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            ckpt_epoch = int(meta.get("epoch", 0) or 0)
            del meta
        except Exception as exc:
            print(f"[step2] WARNING could not read the epoch from {ckpt_path}: {exc!r}",
                  flush=True)

    if args.max_epochs is not None:
        max_epochs = args.max_epochs
        epochs_run = max_epochs - ckpt_epoch
    else:
        epochs_run = args.new_epochs
        max_epochs = ckpt_epoch + args.new_epochs

    if epochs_run <= 0:
        sys.exit(f"max_epochs={max_epochs} is not greater than the checkpoint's "
                 f"epoch ({ckpt_epoch}); Lightning would refuse to train. "
                 f"Use --new-epochs, or --max-epochs > {ckpt_epoch}.")
    print(f"[step2] checkpoint epoch={ckpt_epoch} -> max_epochs={max_epochs} "
          f"({epochs_run} new epochs to be timed)", flush=True)

    result_dir = BENCH_ROOT / "out" / args.env_tag / "training" / (
        f"{args.accelerator}_bs{args.batch_size}_nw{args.num_workers}"
        f"_ep{epochs_run}_{args.matmul_precision}")
    result_dir.mkdir(parents=True, exist_ok=True)

    label = (f"{args.env_tag}/{args.accelerator} bs={args.batch_size} "
             f"nw={args.num_workers} new_ep={epochs_run} mm={args.matmul_precision}")
    print(f"[step2] {label}", flush=True)

    row = {
        "run_id": make_run_id("training", args.env_tag,
                              f"{args.accelerator}_bs", args.batch_size, n_files),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "phase": args.phase,
        "step": "training",
        "env_tag": args.env_tag,
        "accelerator": args.accelerator,
        "n_files": n_files,
        "param_name": "batch_size",
        "param_value": args.batch_size,
        "extra": {
            "num_workers": args.num_workers,
            "max_epochs": max_epochs,
            "ckpt_epoch": ckpt_epoch,
            "epochs_run": epochs_run,
            "matmul_precision": args.matmul_precision,
            "warm_start": not args.no_warm_start,
            "torch": torch.__version__,
            "torch_cuda": getattr(torch.version, "cuda", None),
        },
        "note": args.note,
    }

    try:
        if args.accelerator == "gpu":
            torch.cuda.reset_peak_memory_stats()

        with Timer() as t_split:
            loader = DatasetLoader(merged_store_path=dataset, parquet_file_path=infoset)
            loader.split(key="random", group="faces",
                         train=0.33, validation=0.33, test=0.34)
        print(f"[step2] split in {t_split.elapsed:.1f}s", flush=True)

        model = EmbeddingFlowModel(
            result_dir=str(result_dir),
            log_file=str(result_dir / "flow.log"),
            # Warm start loads the pretrained SIGNAL weights (and its epoch
            # counter) into the nn module; from-scratch must NOT, or Lightning
            # resumes at the checkpoint's epoch (7) and runs far fewer epochs.
            load_checkpoint_using_nn_module=not args.no_warm_start,
            temp_init=0.07,
            temp_min=0.1,
            temp_max=0.15,
        )

        resume = str(ckpt_path) if ckpt_path is not None else None
        if resume:
            print(f"[step2] warm start from {resume}", flush=True)

        trainer = FlowTrainer(
            flowmodel=model,
            datasetLoader=loader,
            experiment_name="BENCH_SIGNAL_embeddings",
            result_dir=result_dir,
            accelerator=args.accelerator,
            devices=[0] if args.accelerator == "gpu" else 1,
            max_epochs=max_epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            early_stopping=False,          # pin the epoch count
            resume_checkpoint_path=resume,
        )

        with PeakSampler() as sampler, Timer() as t_train:
            trained_model_path = trainer.train(train_shuffle=False, train_seed=1234)
        train_s = t_train.elapsed

        test_s = None
        if args.run_test:
            with Timer() as t_test:
                trainer.test(trained_model_path)
            test_s = round(t_test.elapsed, 2)

        # Divide by the epochs ACTUALLY executed. Dividing by max_epochs would
        # understate s/epoch by the checkpoint's epoch offset (7 for the shipped
        # SIGNAL checkpoint - a >3x error at these epoch counts).
        s_per_epoch = train_s / max(1, epochs_run)
        row.update({
            "wall_s": round(train_s, 2),
            "sub_timings": {
                "split_s": round(t_split.elapsed, 2),
                "train_s": round(train_s, 2),
                "s_per_epoch": round(s_per_epoch, 3),
                "test_s": test_s,
                **sampler.as_dict(),
            },
            "throughput": round(epochs_run / train_s, 4) if train_s else "",
            "peak_rss_mb": sampler.peak_rss_mb,
            "peak_gpu_mb": peak_gpu_mb(),
            "n_ok": n_files,
            "n_failed": 0,
            "status": "OK",
        })
        print(f"[step2] train {train_s:.1f}s -> {s_per_epoch:.2f} s/epoch "
              f"(ckpt {trained_model_path})", flush=True)

        # Lightning writes the checkpoint deep inside its own timestamped
        # experiment folder (out/<env>/training/.../flowtrainer/<experiment>/
        # <date>/<time>/best.ckpt) -- fine for bookkeeping, not somewhere a
        # human (or a downstream tool like a qt_sandbox-style desktop client)
        # wants to go hunting. Copy it to a flat, predictable path too, named
        # after --env-tag, so this is the same result whether bench_step2 is
        # run standalone or via run_pipeline.py.
        deliverable_ckpt = BENCH_ROOT / "out" / args.env_tag / f"{args.env_tag}.ckpt"
        deliverable_ckpt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trained_model_path, deliverable_ckpt)
        print(f"[step2] checkpoint copied to {deliverable_ckpt}", flush=True)

        # Record where the freshly trained checkpoint landed so a downstream
        # step 3 run can index with THIS model instead of the warm-start
        # checkpoint (mirrors step 1's dataset_<tag>.json pointer pattern).
        # "checkpoint" points at the flat copy above -- that's the one to
        # hand off; "checkpoint_raw" keeps Lightning's original path too.
        ckpt_pointer = BENCH_ROOT / "results" / f"checkpoint_{args.env_tag}.json"
        ckpt_pointer.parent.mkdir(parents=True, exist_ok=True)
        ckpt_pointer.write_text(json.dumps({
            "env_tag": args.env_tag,
            "checkpoint": str(deliverable_ckpt),
            "checkpoint_raw": str(trained_model_path),
            "max_epochs": max_epochs,
            "epochs_run": epochs_run,
            "batch_size": args.batch_size,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[step2] checkpoint pointer -> {ckpt_pointer}", flush=True)
    except Exception as exc:
        traceback.print_exc()
        msg = repr(exc)
        status = "FAILED"
        if "out of memory" in msg.lower():
            status = "FAILED"
            row["note"] = (args.note + " | CUDA OOM").strip(" |")
        row.update({"wall_s": "", "status": status,
                    "note": (row.get("note", "") + " | " + msg)[:500]})

    write_row(row, args.csv)


if __name__ == "__main__":
    main()
