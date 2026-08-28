# HOOPS AI Embeddings Pipeline: CPU vs GPU

*(日本語版: [README.ja.md](README.ja.md))*

A reusable 3-step pipeline for the HOOPS AI embeddings workflow --
**encode** a folder of CAD files, **train** the embedding model, and **index**
the result for similarity search -- plus the tooling used to benchmark it
CPU vs GPU, from a laptop up to an AWS GPU instance on a 10,000+ file corpus.

This started as an ad hoc benchmark harness and turned into three things:

1. **Three CLI scripts** (`bench_step1_dataprep.py` / `bench_step2_training.py`
   / `bench_step3_indexing.py`) that wrap the corresponding HOOPS AI embeddings
   demo notebooks, each with explicit `--accelerator` / target-folder /
   worker-count / time-limit parameters.
2. **`run_pipeline.py`**, a production runner that chains all three steps
   *once*, in order, against a folder of CAD files -- no sweeping, no
   matrices, just encode -> train -> index.
3. **Sweep harnesses** (`run_benchmark.ps1`, `run_local_sweep.ps1` for
   Windows; `run_heavy_batch.sh`, `run_heavy_scaling.sh` for AWS/Ubuntu) that
   run the same three scripts across many `(accelerator, workers, batch_size,
   ...)` combinations and turn the results into an HTML report -- see
   [`reports/`](reports/).

See [`cdk/`](cdk/) for an AWS CDK project that provisions a GPU EC2 instance
(driver, venvs, this repo) to run all of the above without hand-configuring a
box.

> **This is sample/benchmark code, not a product.** It is a reasonable
> starting point for automating the HOOPS AI embeddings workflow, not a
> hardened pipeline -- read the "Known pitfalls" section before pointing it
> at data you care about.

## Prerequisites

This describes the state your CPU-build and GPU-build venvs need to be in --
if you already have a working HOOPS AI SDK install (per HOOPS AI's own setup
docs) with `hoops_ai` importable, you likely already satisfy all of this and
can skip straight to the Quickstart below. These commands matter when you're
building a venv from scratch (e.g. on a fresh machine, or via `cdk/`, which
runs the CPU/GPU pip installs below automatically):

- A licensed HOOPS AI SDK install, with the `hoops_ai` package importable.
- `torch`, installed as a build that matches your accelerator:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU venv
  pip install torch --index-url https://download.pytorch.org/whl/cu130   # GPU venv (CUDA 13.0)
  ```
- `pip install -r requirements.txt` (just `psutil`; everything else above).
- **A `.env` file** -- copy [`.env.example`](.env.example) to `.env` (repo
  root, gitignored) and fill it in. This one file is read by everything here:
  every `bench_step*.py` / `run_pipeline.py` invocation picks up
  `HOOPS_AI_LICENSE` and `HOOPS_AI_CKPT` from it (via `bench_common.py`'s
  `load_dotenv()`), and `run_benchmark.ps1` / `preflight.ps1` /
  `run_local_sweep.ps1` pick up `CPU_PY` / `GPU_PY` / `HOOPS_AI_CKPT` from it
  too (via `resolve_local_env.ps1`) -- there is no separate PowerShell-only
  config file. Search order (first found wins): `./.env`, `../.env` (one
  directory up -- e.g. the SDK install root if this repo sits next to
  `CPU1.1`/`GPU1.1`), `../CPU1.1/.env`, `../GPU1.1/.env`.
  - `HOOPS_AI_LICENSE` -- required for every script. Export it directly
    instead of using `.env` if you'd rather not have a file.
  - `HOOPS_AI_CKPT` -- only needed if the automatic search for
    `ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt` under
    `*/packages/trained_ml_models/` doesn't find it (e.g. because this repo
    isn't nested inside your SDK install directory, which it usually won't
    be). Also overridable per-invocation with `--ckpt`.
  - `CPU_PY` / `GPU_PY` -- only needed by the three Windows PowerShell
    scripts above. SDK install folder names are not standardized (`V1.1` on
    one machine, `CPU1.1` on another after a rename), so this repo does not
    guess them; also overridable per-invocation with `-CpuPy`/`-GpuPy`.
    `run_heavy_batch.sh` / `run_heavy_scaling.sh` on Linux use the same
    `CPU_PY` / `GPU_PY` names, but as plain shell environment variables
    (export them, or source `.env` yourself) rather than through `.env`.
- On a headless Linux box (no monitor attached, e.g. an EC2 instance): an X
  display. HOOPS AI needs one even for offscreen work. `run_heavy_batch.sh`
  and `run_heavy_scaling.sh` auto-start `Xvfb :99` if `$DISPLAY` is unset; the
  CDK stack in `cdk/` also installs it as a systemd service.

## Quickstart: run the pipeline once, on your own data

> **Use your CPU or GPU venv's own `python.exe`, not a bare `python`.**
> `hoops_ai` is installed inside that venv (e.g.
> `C:\SDK\HOOPS_AI\CPU1.1\.venv\Scripts\python.exe` on Windows, or whatever
> `$CPU_PY`/`$GPU_PY` point at on Linux) -- there's no global environment or
> "activate" step in this repo's workflow. Every command below assumes you've
> substituted the right interpreter; if you see
> `ModuleNotFoundError: No module named 'hoops_ai'`, that's the fix.
> `run_pipeline.py` in particular runs all three steps with *whichever*
> interpreter launched it, so `--accelerator gpu` needs to be your GPU venv's
> python.exe specifically.

```bash
python run_pipeline.py \
  --run-name my_corpus --source-dir /path/to/cad_files \
  --accelerator gpu --workers 16 --epochs 10 --batch-size 32
```

This scans `--source-dir` recursively for CAD files by extension -- STEP,
IGES, Parasolid, ACIS, CATIA V4/V5/V6, SolidWorks, Inventor, NX/Creo/Pro-E,
Solid Edge, JT, Rhino, PRC: the formats [HOOPS Exchange's docs](https://docs.techsoft3d.com/hoops/exchange/start/supported-formats.html)
list as carrying full B-rep, not mesh-only formats like STL (HOOPS AI needs
an actual B-rep to encode). See `CAD_EXTENSIONS` in `bench_common.py` for the
exact list, or override it with `--extensions .stp,.catpart,...`. It then
freezes the discovered list and runs step 1 (encode, keeping the
dataset), step 2 (train, warm-started from the shipped checkpoint by
default -- pass `--no-warm-start` to train from scratch), and step 3 (embed +
build a FAISS index), pointing step 3 at the checkpoint step 2 *just trained*
rather than the original warm-start weights. It also renders a PNG + `.scs`
scene cache per part by default (`--no-gen-images` to skip), matching what
qt_sandbox's own "Add Folder" produces -- these are not optional there.
Each step appends one row to `results/results.csv`; the deliverables land
under `out/<run-name>/`:

- `<run-name>.ckpt` -- a flat copy of the trained checkpoint (Lightning
  itself writes it several directories deeper, under
  `training/.../flowtrainer/...`)
- `indexing/<run-name>.faiss` + `.meta` -- the saved index
- `indexing/<run-name>/` -- the per-part `.png`/`.scs` assets, mirroring
  `--source-dir`'s folder structure underneath

Copy all of these to a desktop client such as qt_sandbox to use them there --
load the checkpoint first, then open the index, since embeddings from a
different model aren't comparable.

Run `python run_pipeline.py --help` for the full parameter list (worker
counts, per-file time limit, DataLoader workers, PNG rendering during
indexing, etc).

## The three steps, standalone

| | Step 1 -- encode | Step 2 -- train | Step 3 -- index |
|---|---|---|---|
| script | `bench_step1_dataprep.py` | `bench_step2_training.py` | `bench_step3_indexing.py` |
| CPU/GPU | implicit (which venv you run it with) | `--accelerator {cpu,gpu}` | auto-detected; falls back to CPU and labels itself correctly if the installed torch build can't actually run kernels for the card |
| target folder | `--source-dir`, scanned recursively (or `--filelist`, see below) | `--dataset-pointer` (points at step 1's output) | `--source-dir`, scanned recursively (or `--filelist`) |
| worker count | `--max-workers` | `--num-workers` (DataLoader) | `--num-workers` |
| per-file time limit | `--time-limit-s` | -- (training is epoch-bounded by design, not wall-clock) | `--time-limit` |

Step 1 and step 3 both accept a plain folder via `--source-dir`: every file
under it (recursing into subfolders) whose extension is in `CAD_EXTENSIONS`
(`bench_common.py` -- the formats HOOPS Exchange's docs list as carrying full
B-rep; override per-run with `--extensions .stp,.catpart,...` if it doesn't
match your data or your license) is included, sorted for a deterministic
order, and frozen to `filelists/<env-tag>_discovered.txt` so repeated runs
process the same files the same way. Pass `--filelist <path>` instead of
`--source-dir` when you need an exact, reused, hand-curated list -- that's
what the sweep harnesses below do, so every configuration in a sweep is
compared on identical input.

Each script is independently runnable, e.g.:

```bash
python bench_step1_dataprep.py --env-tag gpu --max-workers 16 \
    --source-dir /path/to/cad --keep-output

python bench_step2_training.py --env-tag gpu --accelerator gpu \
    --batch-size 32 --new-epochs 10

python bench_step3_indexing.py --env-tag gpu --num-workers 16 \
    --source-dir /path/to/cad --save-index
```

`run_pipeline.py` is a thin wrapper around exactly these three invocations,
wired together (dataset pointer, then checkpoint pointer) so you don't have
to track the intermediate JSON files yourself. Note that `bench_step2_training.py`
always copies its trained checkpoint to `out/<env-tag>/<env-tag>.ckpt` (in
addition to Lightning's own deeper path) and records both in
`results/checkpoint_<env-tag>.json` -- this happens whether you call it
standalone or through `run_pipeline.py`, so the flat, findable path is always
there. Use `--index-name` on `bench_step3_indexing.py` to name the saved
index the same way instead of the default `<env-tag>_n<n_files>` -- with
`--gen-images` it also controls where the per-part `.png`/`.scs` assets land
(`<index-name>/`, directly next to `<index-name>.faiss`), matching
qt_sandbox's own layout. Unlike `run_pipeline.py`, `--gen-images` here
defaults to *off*: this script doubles as the benchmark harness's building
block, and the original point of the flag was to isolate rendering's
(large, constant) cost from the `num_workers` signal -- see "Fair-comparison
design decisions" below.

## Benchmarking: CPU vs GPU across the pipeline

The scripts above are also what the sweep harnesses call, one process per
measurement, so CUDA contexts/worker pools/allocator state never leak between
runs. Two entry points:

- **`run_benchmark.ps1`** (Windows, local machine) -- a phased, time-budgeted
  sweep: probe -> step 1 `max_workers` sweep -> full encode (both envs) ->
  step 2 `accelerator x batch_size x num_workers` matrix -> step 3
  `num_workers` sweep + full index build -> report. Supports `-DryRun`,
  `-Unattended` (overnight runs: no prompts, suppresses sleep, hard per-run
  timeout so one hung measurement can't eat the whole budget), `-Phases`.
- **`run_heavy_batch.sh`** / **`run_heavy_scaling.sh`** (AWS/Ubuntu, heavy
  corpus) -- the same idea at 10,000+ file scale: `run_heavy_batch.sh` runs
  step 1 once (CPU, keeping the dataset) then step 2 CPU vs GPU on that one
  dataset so both sides train the *same* model; `run_heavy_scaling.sh` sweeps
  step 1/3 worker counts on a representative subset.

Results all land in `results/results.csv` (gitignored -- can contain your CAD
file paths); `make_report.py --lang en --scope {local,aws,both}` turns it
into `results/REPORT_*.md/.html`. The two reports already committed under
[`reports/`](reports/) are from the original CPU1.1/GPU1.1 laptop and
g6.8xlarge/L4 runs behind this repo -- regenerate your own with the same
tool once you have your own `results.csv`.

## Fair-comparison design decisions

A few choices exist specifically so CPU-vs-GPU and worker-count numbers are
comparable, not just fast:

- **Frozen file lists, not re-scanned directory walks, across a sweep.**
  `--source-dir` is convenient for a one-off run (see above), but every
  configuration in a *sweep* is pointed at the same explicit `--filelist` so
  they all process identical files in identical order; `random.shuffle` in
  the original demo code would otherwise skew per-run timings.
- **Size-stratified subsets.** Smaller sweep subsets are picked by sorting on
  file size and taking an even stride across the whole range, not `[::k]` --
  a naive slice drops the largest files and makes the subset lighter (and
  the sweep look faster) than the full corpus.
- **One configuration per process.** Every measurement launches a fresh
  Python process so CUDA context / worker pool / allocator state from the
  previous run can't leak into the next one's numbers.
- **Step 2 pins `early_stopping=False`, a fixed batch order
  (`train_shuffle=False`, `train_seed=1234`) and epoch count**, so wall-clock
  differences are a pure device/parameter comparison, not an artifact of the
  loss curve plateauing at different points.

## Known pitfalls

| symptom | cause / fix |
|---|---|
| `HOOPS_AI_LICENSE environment variable is required` | create the `.env` described above |
| `Could not find ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt` | pass `--ckpt` / `HOOPS_AI_CKPT` explicitly |
| `--accelerator gpu requested but torch.cuda.is_available() is False` | you're running the CPU-only venv; use the GPU one |
| `torch.cuda.is_available()` returns `True` but every GPU run fails with `no kernel image is available for execution on the device` | your torch build doesn't include kernels for your GPU's compute capability (seen on an older Pascal-generation card with a torch wheel that only ships sm_75+). Install a torch build whose CUDA/kernel support matches your GPU (check `torch.cuda.get_device_capability()` against the wheel's supported architectures), or benchmark CPU-only -- "this GPU isn't supported by HOOPS AI's current torch requirement" is itself a useful result. `bench_common.cuda_usable()` verifies with a real matmul launch rather than trusting `is_available()`, and step 3 auto-masks an unusable GPU (`CUDA_VISIBLE_DEVICES=""`) so it doesn't just fail on every file |
| step 2 raises `CUDA out of memory` | that `batch_size` exceeds VRAM; the sweep harnesses record it as `FAILED` and move on |
| step 1/3 throughput is noisy between runs | on Windows, real-time antivirus scanning the STEP files during the run; exclude the corpus and output folders |
| a sweep runs much slower than the `-DryRun` estimate | the estimate assumes ~34 core-seconds/file until real Phase 1 numbers replace it |
| headless Linux box hangs or errors on the first CAD file | no X display -- HOOPS AI needs one even offscreen; start `Xvfb :99` and `export DISPLAY=:99` (both heavy-corpus shell scripts do this automatically) |
| step 1 gets stuck repeating `RAM limit reached ... Restarting workers` / `worker still alive after kill attempt` and never progresses | too many `--max-workers` for the RAM actually available; a worker that fails to die stays resident while a replacement is started anyway, so available RAM keeps dropping each restart instead of recovering. Lower `--max-workers` (a handful of files doesn't need 12), check Task Manager / `ps` for leftover worker processes from a previous aborted run before retrying, and close other RAM-heavy applications |

## Source

This repo is provided as a starting point, not production-ready code --
review it (especially `run_pipeline.py` and the CDK stack in `cdk/`) before
pointing it at data or infrastructure you care about. Build/setup details for
the AWS side live in [`cdk/README.md`](cdk/README.md) rather than duplicated
here.
