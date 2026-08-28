#!/usr/bin/env bash
# Heavy (~10k file) benchmark for HOOPS AI 1.1 on AWS (g6.8xlarge Ubuntu).
#
# Goal
# ----
#   Step 1 (encoding): run ONCE on CPU with 16 workers, measure throughput and
#                      keep the encoded dataset.
#   Step 2 (training): compare CPU vs GPU at IDENTICAL settings so both produce
#                      the SAME trained model - same dataset, same batch_size,
#                      same epochs, same seed. Only the device changes, so the
#                      wall-clock difference is a pure speed comparison.
#
# Why one dataset for both Step 2 runs
#   Batch_size is a contrastive-learning hyperparameter: it sets the in-batch
#   negatives, so a different batch trains a different model. Likewise, training
#   on two different encodings would compare two different models. We therefore
#   encode once (Step 1, CPU) and point BOTH Step 2 runs at that single dataset.
#
# Portability
#   The file list (filelists/mech_heavy.txt) stores paths RELATIVE to the corpus
#   root. Set FILE_ROOT to wherever the mechcad tree lives on this box; the
#   harness prepends it via BENCH_FILE_ROOT.
#
# Usage
#   FILE_ROOT=/home/ubuntu/mechcad \
#   CPU_PY=/home/ubuntu/venvs/cpu/bin/python \
#   GPU_PY=/home/ubuntu/venvs/gpu/bin/python \
#   ./run_heavy_batch.sh
#
set -uo pipefail

# ---- config (override via environment) -----------------------------------
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPU_PY="${CPU_PY:-python}"
GPU_PY="${GPU_PY:-python}"
FILE_ROOT="${FILE_ROOT:-/home/ubuntu/mechcad}"
FILELIST="${FILELIST:-filelists/mech_heavy.txt}"
WORKERS="${WORKERS:-16}"          # Step 1 CPU max_workers (16 phys cores on g6.8xlarge)
BATCH="${BATCH:-64}"              # Step 2 batch_size (tutorial default; same model both devices)
EPOCHS="${EPOCHS:-10}"           # Step 2 epochs (from scratch)
NUMWORKERS="${NUMWORKERS:-0}"     # Step 2 DataLoader workers
MATMUL="${MATMUL:-high}"          # Step 2 matmul precision
TIME_LIMIT="${TIME_LIMIT:-1200}"  # Step 1 per-file encode limit (s)
CPU_TAG="cpu1.1"
GPU_TAG="gpu1.1"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$BENCH_DIR/logs"
mkdir -p "$LOG_DIR"

export BENCH_FILE_ROOT="$FILE_ROOT"

cd "$BENCH_DIR" || exit 1

say() { printf '%s\n' "$*"; }
rule() { printf '=%.0s' {1..74}; printf '\n'; }

# HOOPS AI initialisation needs an X display even for offscreen work. On a
# headless box start a virtual one (Xvfb) if none is set; the exported DISPLAY
# is inherited by the python harness and its fork worker processes.
XVFB_PID=""
if [[ -z "${DISPLAY:-}" ]]; then
  if command -v Xvfb >/dev/null 2>&1; then
    Xvfb :99 -screen 0 1280x1024x24 >/tmp/xvfb_bench.log 2>&1 &
    XVFB_PID=$!
    export DISPLAY=:99
    sleep 2
    say "[display] started Xvfb on :99 (pid $XVFB_PID)"
  else
    say "[display] WARNING: no DISPLAY and Xvfb not installed."
    say "[display]   sudo apt install -y xvfb libgl1 libglu1-mesa libxrender1 libxext6 libsm6"
  fi
fi
cleanup() { [[ -n "$XVFB_PID" ]] && kill "$XVFB_PID" 2>/dev/null; }
trap cleanup EXIT

rule; say "HOOPS AI 1.1 heavy batch  ($(wc -l < "$FILELIST" 2>/dev/null || echo '?') files)"; rule
say "bench dir : $BENCH_DIR"
say "corpus    : $FILE_ROOT   (BENCH_FILE_ROOT)"
say "filelist  : $FILELIST"
say "CPU py    : $CPU_PY"
say "GPU py    : $GPU_PY"
say "step1     : CPU max_workers=$WORKERS  time_limit=${TIME_LIMIT}s (keep dataset)"
say "step2     : CPU vs GPU  batch_size=$BATCH  epochs=$EPOCHS  num_workers=$NUMWORKERS  matmul=$MATMUL"
say ""

# sanity: corpus + first file present
first_rel="$(head -n1 "$FILELIST" 2>/dev/null)"
if [[ ! -f "$FILE_ROOT/$first_rel" ]]; then
  say "ERROR: corpus check failed - '$FILE_ROOT/$first_rel' not found."
  say "       Set FILE_ROOT to the directory that holds bearing/ bolt/ ... "
  exit 1
fi

run() {  # label python script args...
  local label="$1" py="$2"; shift 2
  local log="$LOG_DIR/${STAMP}-${label}.log"
  say ""; say ">> $label"; say "   $py $*"
  "$py" "$@" 2>&1 | tee "$log"
  local code=${PIPESTATUS[0]}
  if [[ $code -ne 0 ]]; then say "   [warn] exit=$code (see $log)"; fi
}

# ==========================================================================
# STEP 1 - encoding (CPU, 16 workers), keep the dataset for Step 2
# ==========================================================================
rule; say "STEP 1  encoding - CPU max_workers=$WORKERS"; rule
run "s1-cpu-mw${WORKERS}" "$CPU_PY" bench_step1_dataprep.py \
    --env-tag "$CPU_TAG" --max-workers "$WORKERS" \
    --filelist "$FILELIST" --source-dir "$FILE_ROOT" \
    --phase HB1 --time-limit-s "$TIME_LIMIT" \
    --keep-output --note "heavy-batch"

PTR="$BENCH_DIR/results/dataset_${CPU_TAG}.json"
if [[ ! -f "$PTR" ]]; then
  say ""; say "ERROR: Step 1 did not produce $PTR - cannot run Step 2."; exit 1
fi

# ==========================================================================
# STEP 2 - training: CPU vs GPU, IDENTICAL settings, SAME dataset ($PTR)
# ==========================================================================
rule; say "STEP 2  training - CPU vs GPU @ batch_size=$BATCH (same model)"; rule

run "s2-cpu-bs${BATCH}" "$CPU_PY" bench_step2_training.py \
    --env-tag "$CPU_TAG" --accelerator cpu \
    --dataset-pointer "$PTR" \
    --batch-size "$BATCH" --num-workers "$NUMWORKERS" \
    --no-warm-start --max-epochs "$EPOCHS" \
    --matmul-precision "$MATMUL" --phase HB2 --note "heavy-batch same-dataset"

run "s2-gpu-bs${BATCH}" "$GPU_PY" bench_step2_training.py \
    --env-tag "$GPU_TAG" --accelerator gpu \
    --dataset-pointer "$PTR" \
    --batch-size "$BATCH" --num-workers "$NUMWORKERS" \
    --no-warm-start --max-epochs "$EPOCHS" \
    --matmul-precision "$MATMUL" --phase HB2 --note "heavy-batch same-dataset"

rule; say "DONE"; rule
say "Results CSV : $BENCH_DIR/results/results.csv"
say "Rows        : phase HB1 (Step1 CPU), phase HB2 (Step2 CPU & GPU, same dataset)"
say "Both Step 2 runs used $PTR -> identical training data, batch_size=$BATCH,"
say "epochs=$EPOCHS, seed=1234: same model, so wall-clock is a pure CPU-vs-GPU speed test."
