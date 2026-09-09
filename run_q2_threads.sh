#!/usr/bin/env bash
# Q2 re-test: is the Step 3 (embedding + FAISS indexing) throughput DECLINE at
# high worker counts caused by intra-op CPU-thread oversubscription?
#
# Background
# ----------
#   On the g6.8xlarge (16 physical / 32 logical cores) heavy run, Step 3
#   throughput peaked around num_workers=12 and then *fell* as workers were
#   added. Our hypothesis: each embed worker is a separate process that, by
#   default, spins up ~one BLAS/OpenMP (intra-op) thread per core, so past a few
#   workers the total thread count (workers x cores) massively oversubscribes
#   the machine and slows every worker down. The engineering team could not
#   confirm this (they guessed RAM), so this run tests it directly.
#
# Method
# ------
#   `embed_shape_batch` only exposes num_workers (the process count), NOT the
#   per-worker thread count. bench_step3_indexing.py --intraop-threads N caps it
#   by exporting OMP/MKL/OPENBLAS/NUMEXPR/VECLIB_NUM_THREADS before torch is
#   imported (inherited by the spawned workers). We sweep num_workers at three
#   thread caps and compare the curves:
#
#     * threads = default (no cap): reproduce the decline (control)
#     * threads = 1              : total threads = workers x 1  -> no oversubscription until workers > cores
#     * threads = 2              : middle ground
#
#   If capping threads flattens/removes the decline at high worker counts, the
#   oversubscription hypothesis is confirmed. If the decline persists, RAM (or
#   something else) is the driver -- peak_rss_mb is recorded either way to
#   separate the two.
#
#   Step 3 is CPU-bound (the GPU sits idle here), so this runs on the CPU venv.
#
# Dataset (reasonable subset, NOT the full 10k)
# ---------------------------------------------
#   A balanced 500-file subset of mechcad (50 per category, size-stratified and
#   round-robin interleaved so heavy files are spread evenly across the list --
#   critical for a fair worker sweep). Build it on this box first:
#
#     python make_filelist_mech.py --source /home/ubuntu/mechcad --per-category 50
#     # -> filelists/mech500.txt   (absolute paths, used as-is)
#
# Usage
# -----
#   CPU_PY=/home/ubuntu/CPU1.1/.venv/bin/python \
#   FILELIST=filelists/mech500.txt \
#   ./run_q2_threads.sh
#
#   Optional overrides:
#     WORKERS="8 12 16 24 32 40"   worker counts to sweep
#     THREADS="def 1 2"            intra-op thread caps (def = no cap / control)
#     TIME_LIMIT=1200              per-file embed time limit (s)
#     CKPT=/path/to/ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt
#     WARMUP=1                     do one discarded warm-up run first
#
set -uo pipefail

# ---- config (override via environment) -----------------------------------
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPU_PY="${CPU_PY:-python}"
FILELIST="${FILELIST:-filelists/mech500.txt}"
WORKERS="${WORKERS:-8 12 16 24 32 40}"   # below / at / above 16 physical & 32 logical
THREADS="${THREADS:-def 1 2}"            # def = default (no cap, the control)
TIME_LIMIT="${TIME_LIMIT:-1200}"         # per-file embed time limit (s)
CKPT="${CKPT:-}"                         # Step 3 SIGNAL checkpoint (or HOOPS_AI_CKPT)
WARMUP="${WARMUP:-1}"
CPU_TAG="cpu1.1"
PHASE="Q2T"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$BENCH_DIR/logs"
mkdir -p "$LOG_DIR"

[[ -n "$CKPT" ]] && export HOOPS_AI_CKPT="$CKPT"

cd "$BENCH_DIR" || exit 1

say() { printf '%s\n' "$*"; }
rule() { printf '=%.0s' {1..74}; printf '\n'; }

# Step 3 opens many files at once per worker; a many-worker sweep can blow past
# the default soft nofile limit and die with "Too many open files". Raise it.
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
    Xvfb :99 -screen 0 1280x1024x24 >/tmp/xvfb_q2.log 2>&1 &
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

if [[ ! -f "$FILELIST" ]]; then
  say "ERROR: filelist not found: $FILELIST"
  say "       Build it first, e.g.:"
  say "       $CPU_PY make_filelist_mech.py --source /home/ubuntu/mechcad --per-category 50"
  exit 1
fi

rule; say "Q2 intra-op thread test  ($(wc -l < "$FILELIST" 2>/dev/null || echo '?') files)"; rule
say "bench dir : $BENCH_DIR"
say "filelist  : $FILELIST"
say "CPU py    : $CPU_PY"
say "workers   : $WORKERS"
say "threads   : $THREADS   (def = default / no cap = control)"
say "phase     : $PHASE"
say "ckpt      : ${CKPT:-<from HOOPS_AI_CKPT / resolve_checkpoint>}"
say ""

# sanity: first file present (list stores absolute paths)
first="$(head -n1 "$FILELIST" 2>/dev/null)"
if [[ ! -f "$first" ]]; then
  say "ERROR: corpus check failed - first listed file not found: '$first'"
  say "       Regenerate the list on this box so the paths point at the local corpus."
  exit 1
fi

run() {  # label args...
  local label="$1"; shift
  local log="$LOG_DIR/${STAMP}-${label}.log"
  say ""; say ">> $label"; say "   $CPU_PY bench_step3_indexing.py $*"
  "$CPU_PY" bench_step3_indexing.py "$@" 2>&1 | tee "$log"
  local code=${PIPESTATUS[0]}
  if [[ $code -ne 0 ]]; then say "   [warn] exit=$code (see $log)"; fi
}

# One discarded warm-up: the first run pays venv/lib load and a first filesystem
# scan of the STEP files, which would otherwise penalise whichever point runs
# first. Not written to a distinct phase; note marks it discarded.
if [[ "$WARMUP" == "1" ]]; then
  rule; say "WARM-UP (discarded)"; rule
  run "warmup-nw8" --env-tag "$CPU_TAG" --num-workers 8 \
      --filelist "$FILELIST" --time-limit "$TIME_LIMIT" \
      --phase "${PHASE}_WARM" --note "q2-threads warmup discarded"
fi

# ==========================================================================
# Sweep: for each thread cap, sweep num_workers.
# ==========================================================================
for T in $THREADS; do
  if [[ "$T" == "def" ]]; then
    rule; say "THREAD CAP = default (no cap)  -- control, expect the decline"; rule
    for W in $WORKERS; do
      run "s3-tDEF-nw${W}" --env-tag "$CPU_TAG" --num-workers "$W" \
          --filelist "$FILELIST" --time-limit "$TIME_LIMIT" \
          --phase "$PHASE" --note "q2-threads it=default"
    done
  else
    rule; say "THREAD CAP = $T intra-op thread(s) per worker"; rule
    for W in $WORKERS; do
      run "s3-t${T}-nw${W}" --env-tag "$CPU_TAG" --num-workers "$W" \
          --intraop-threads "$T" \
          --filelist "$FILELIST" --time-limit "$TIME_LIMIT" \
          --phase "$PHASE" --note "q2-threads it=${T}"
    done
  fi
done

rule; say "DONE"; rule
say "Results CSV : $BENCH_DIR/results/results.csv  (phase $PHASE)"
say "Compare files/s vs num_workers across thread caps (extra.intraop_threads):"
say "  - if the default curve declines but the capped curves stay flat/high"
say "    at high num_workers, thread oversubscription is confirmed."
say "  - peak_rss_mb is recorded per run to separate a RAM-driven drop."
