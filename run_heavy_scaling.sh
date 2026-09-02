#!/usr/bin/env bash
# Heavy (~10k file) worker-scaling benchmark for HOOPS AI 1.1 on AWS
# (g6.8xlarge Ubuntu, AMD EPYC 16 physical cores + NVIDIA L4).
#
# Goal
# ----
#   Characterise how Step 1 (encoding) and Step 3 (embedding + FAISS indexing)
#   scale with worker count on the SAME 10k heavy dataset used for Step 2. Both
#   stages are CPU-bound, so we sweep worker count on the CPU install and expect
#   throughput to rise until ~the physical-core count (16) and then flatten.
#
#   Because the curve is nearly flat around the peak, we sweep in coarse steps
#   (8, 12, 16, 20, 24) rather than every 2, to cover below / at / above the
#   16-core mark without wasting hours on near-identical points.
#
# What it runs (per worker count W in WORKERS)
#   Step 1: bench_step1_dataprep.py  --max-workers W   (phase HB1, re-encodes all)
#   Step 3: bench_step3_indexing.py  --num-workers W   (phase HB3, re-embeds all)
#
#   Each point re-processes the whole corpus, so this is a multi-hour run.
#
# Portability
#   The file list stores paths RELATIVE to the corpus root. Set FILE_ROOT to the
#   mechcad tree on this box; both harnesses prepend it via BENCH_FILE_ROOT.
#
# Step 3 needs the SIGNAL checkpoint
#   ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt. Point CKPT at it (or export
#   HOOPS_AI_CKPT). It ships inside the HOOPS AI install's
#   packages/trained_ml_models/ directory.
#
# Usage
#   FILE_ROOT=/home/ubuntu/dataset/mechcad/mechcad \
#   CPU_PY=/var/HOOPS_AI/CPU1.1/.venv/bin/python \
#   CKPT=/path/to/ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt \
#   FILELIST=filelists/mech_heavy_aws.txt \
#   ./run_heavy_scaling.sh
#
set -uo pipefail

# ---- config (override via environment) -----------------------------------
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPU_PY="${CPU_PY:-python}"
FILE_ROOT="${FILE_ROOT:-/home/ubuntu/dataset/mechcad/mechcad}"
FILELIST="${FILELIST:-filelists/mech_heavy.txt}"
WORKERS="${WORKERS:-8 12 16 20 24}"   # coarse sweep around the 16-core peak
TIME_LIMIT="${TIME_LIMIT:-1200}"      # per-file limit (s) for both stages
CKPT="${CKPT:-}"                      # Step 3 SIGNAL checkpoint (or HOOPS_AI_CKPT)
DO_STEP1="${DO_STEP1:-1}"
DO_STEP3="${DO_STEP3:-1}"
CPU_TAG="cpu1.1"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$BENCH_DIR/logs"
mkdir -p "$LOG_DIR"

export BENCH_FILE_ROOT="$FILE_ROOT"
[[ -n "$CKPT" ]] && export HOOPS_AI_CKPT="$CKPT"

cd "$BENCH_DIR" || exit 1

say() { printf '%s\n' "$*"; }
rule() { printf '=%.0s' {1..74}; printf '\n'; }

# Step 3 opens many files at once (SIGNAL checkpoint + Zarr chunks + pool pipes,
# per worker); a many-worker sweep can blow past the default soft nofile limit
# (often 1024) and die with "Too many open files". Raise it toward the hard
# limit, mirroring the Xvfb auto-setup below.
_hard_nofile="$(ulimit -Hn 2>/dev/null || echo 1024)"
_want_nofile=1048576
if [[ "$_hard_nofile" == "unlimited" ]]; then
  ulimit -n "$_want_nofile" 2>/dev/null || true
elif [[ "$_hard_nofile" =~ ^[0-9]+$ ]] && (( _hard_nofile > 1024 )); then
  (( _want_nofile > _hard_nofile )) && _want_nofile="$_hard_nofile"
  ulimit -n "$_want_nofile" 2>/dev/null || true
fi
say "[ulimit] open-file soft limit = $(ulimit -Sn)"

# HOOPS AI initialisation needs an X display even for offscreen work.
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

rule; say "HOOPS AI 1.1 heavy worker-scaling  ($(wc -l < "$FILELIST" 2>/dev/null || echo '?') files)"; rule
say "bench dir : $BENCH_DIR"
say "corpus    : $FILE_ROOT   (BENCH_FILE_ROOT)"
say "filelist  : $FILELIST"
say "CPU py    : $CPU_PY"
say "workers   : $WORKERS"
say "step1     : $([[ $DO_STEP1 == 1 ]] && echo ON || echo off)  (phase HB1, max_workers sweep)"
say "step3     : $([[ $DO_STEP3 == 1 ]] && echo ON || echo off)  (phase HB3, num_workers sweep)"
say "ckpt      : ${CKPT:-<from HOOPS_AI_CKPT / resolve_checkpoint>}"
say ""

# sanity: corpus + first file present
first_rel="$(head -n1 "$FILELIST" 2>/dev/null)"
if [[ ! -f "$FILE_ROOT/$first_rel" && ! -f "$first_rel" ]]; then
  say "ERROR: corpus check failed - neither '$FILE_ROOT/$first_rel' nor '$first_rel' found."
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
# STEP 1 - encoding throughput vs max_workers (CPU)
# ==========================================================================
if [[ "$DO_STEP1" == "1" ]]; then
  rule; say "STEP 1  encoding - CPU max_workers sweep: $WORKERS"; rule
  for W in $WORKERS; do
    run "s1-cpu-mw${W}" "$CPU_PY" bench_step1_dataprep.py \
        --env-tag "$CPU_TAG" --max-workers "$W" \
        --filelist "$FILELIST" --source-dir "$FILE_ROOT" \
        --phase HB1 --time-limit-s "$TIME_LIMIT" \
        --note "heavy-scaling"
  done
fi

# ==========================================================================
# STEP 3 - embedding + FAISS indexing throughput vs num_workers (CPU)
# ==========================================================================
if [[ "$DO_STEP3" == "1" ]]; then
  rule; say "STEP 3  indexing - CPU num_workers sweep: $WORKERS"; rule
  for W in $WORKERS; do
    run "s3-cpu-nw${W}" "$CPU_PY" bench_step3_indexing.py \
        --env-tag "$CPU_TAG" --num-workers "$W" \
        --filelist "$FILELIST" \
        --time-limit "$TIME_LIMIT" \
        --phase HB3 --note "heavy-scaling"
  done
fi

rule; say "DONE"; rule
say "Results CSV : $BENCH_DIR/results/results.csv"
say "Rows        : phase HB1 (Step1 max_workers sweep), phase HB3 (Step3 num_workers sweep)"
say "Both stages are CPU-bound; throughput should rise to ~16 workers then flatten."
