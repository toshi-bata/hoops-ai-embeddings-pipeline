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
#   Step 1: bench_step1_dataprep.py  --max-workers W   (phase HB1/HC1, re-encodes all)
#   Step 3: bench_step3_indexing.py  --num-workers W   (phase HB3/HC3, re-embeds all)
#
#   Each point re-processes the whole corpus, so this is a multi-hour run.
#
#   Phase names depend on the intra-op cap: INTRAOP=0 (uncapped) keeps the
#   original HB1/HB3; any cap (the default INTRAOP=1) uses HC1/HC3 (C = capped)
#   so the capped curves never mix with the earlier uncapped HB1/HB3 rows.
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
#   Re-measurement calls (default INTRAOP=1 -> phases HC1/HC3):
#     Step 3 CPU sweep:  DO_STEP1=0 WORKERS="8 12 16 24 32" ./run_heavy_scaling.sh
#     Step 1 CPU sweep:  DO_STEP3=0 WORKERS="16 24 32" ./run_heavy_scaling.sh
#     GPU parity:        SIDE=gpu DO_STEP1=0 WORKERS="<Step 3 CPU peak>" GPU_PY=/path/gpu/python ./run_heavy_scaling.sh
#                        SIDE=gpu DO_STEP3=0 WORKERS="<Step 1 CPU peak>" GPU_PY=/path/gpu/python ./run_heavy_scaling.sh
#
#   Optional overrides:
#     INTRAOP=1            intra-op threads per worker (0 = uncapped control -> HB1/HB3)
#     SIDE=cpu|gpu         which venv/device (gpu uses GPU_PY + gpu1.1 tag)
#     WARMUP=1             do one discarded warm-up run first (phase WU)
#     WARMUP_N=100         warm up on just the first N files (fixed first-run cost only)
#     PRELOAD=1            read the whole corpus once into the page cache first
#
set -uo pipefail

# ---- config (override via environment) -----------------------------------
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPU_PY="${CPU_PY:-python}"
GPU_PY="${GPU_PY:-}"                   # GPU venv python; required only when SIDE=gpu
FILE_ROOT="${FILE_ROOT:-/home/ubuntu/dataset/mechcad/mechcad}"
FILELIST="${FILELIST:-filelists/mech_heavy.txt}"
WORKERS="${WORKERS:-8 12 16 20 24}"   # coarse sweep around the 16-core peak
TIME_LIMIT="${TIME_LIMIT:-1200}"      # per-file limit (s) for both stages
CKPT="${CKPT:-}"                      # Step 3 SIGNAL checkpoint (or HOOPS_AI_CKPT)
INTRAOP="${INTRAOP:-1}"               # intra-op threads per worker (0 = uncapped control)
SIDE="${SIDE:-cpu}"                   # cpu | gpu  (gpu uses GPU_PY + gpu1.1 tag)
WARMUP="${WARMUP:-1}"                 # one discarded warm-up run first (phase WU)
WARMUP_N="${WARMUP_N:-100}"           # warm up on just the first N files (fixed first-run cost)
PRELOAD="${PRELOAD:-1}"               # read the whole corpus once into the page cache first
DO_STEP1="${DO_STEP1:-1}"
DO_STEP3="${DO_STEP3:-1}"
CPU_TAG="cpu1.1"
GPU_TAG="gpu1.1"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$BENCH_DIR/logs"
mkdir -p "$LOG_DIR"

export BENCH_FILE_ROOT="$FILE_ROOT"
[[ -n "$CKPT" ]] && export HOOPS_AI_CKPT="$CKPT"

# Pick the interpreter / env tag for the requested side. GPU_PY has no default
# (unlike run_heavy_batch.sh, where both sides always run): a scaling sweep is
# CPU by default, so an unset GPU venv must fail loudly rather than silently
# fall back to `python`.
if [[ "$SIDE" == "gpu" ]]; then
  if [[ -z "$GPU_PY" ]]; then
    printf 'ERROR: SIDE=gpu but GPU_PY is not set. Point it at the GPU venv python.\n' >&2
    exit 1
  fi
  PY="$GPU_PY"; TAG="$GPU_TAG"
else
  PY="$CPU_PY"; TAG="$CPU_TAG"
fi

# Phase names: keep the original HB1/HB3 for the uncapped control so capped runs
# (HC1/HC3) never mix with earlier HB1/HB3 rows in results.csv.
if [[ "$INTRAOP" == "0" ]]; then
  PH1="HB1"; PH3="HB3"
else
  PH1="HC1"; PH3="HC3"
fi

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
say "side      : $SIDE   (py: $PY, tag: $TAG)"
say "workers   : $WORKERS"
say "intra-op  : $([[ $INTRAOP == 0 ]] && echo 'uncapped (control)' || echo "$INTRAOP thread(s)/worker")"
say "preload   : $([[ $PRELOAD == 1 ]] && echo 'ON (read corpus into page cache)' || echo off)"
say "warmup    : $([[ $WARMUP == 1 ]] && echo "ON (phase WU, first $WARMUP_N files, discarded)" || echo off)"
say "step1     : $([[ $DO_STEP1 == 1 ]] && echo ON || echo off)  (phase $PH1, max_workers sweep)"
say "step3     : $([[ $DO_STEP3 == 1 ]] && echo ON || echo off)  (phase $PH3, num_workers sweep)"
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

# First worker count in the sweep, used for the warm-up.
WARMUP_W="$(set -- $WORKERS; echo "${1:-8}")"

# Page-cache preload: just after an instance (re)start the corpus is cold on
# EBS, so only the sweep's FIRST run pays the slow cold read and looks slower
# than it is. Read every file once so all worker counts start from the same warm
# page cache. Paths are resolved the way the harnesses do (bench_tasks.py): a
# relative entry is taken under FILE_ROOT, an absolute entry as-is. The NUL
# delimiter keeps paths with spaces intact.
if [[ "$PRELOAD" == "1" ]]; then
  rule; say "PRELOAD corpus into page cache"; rule
  _pl_start=$(date +%s)
  _pl_ok=0; _pl_miss=0; _pl_bytes=0
  while IFS= read -r _rel || [[ -n "$_rel" ]]; do
    [[ -z "$_rel" ]] && continue
    case "$_rel" in
      /*) _abs="$_rel" ;;
      *)  _abs="$FILE_ROOT/$_rel" ;;
    esac
    if [[ -f "$_abs" ]]; then
      _sz=$(stat -c %s "$_abs" 2>/dev/null || echo 0)
      _pl_bytes=$(( _pl_bytes + _sz )); _pl_ok=$(( _pl_ok + 1 ))
    else
      _pl_miss=$(( _pl_miss + 1 ))
    fi
  done < "$FILELIST"
  # Actually pull the bytes through the page cache (missing files just error on
  # cat and are skipped; they are already counted above).
  while IFS= read -r _rel || [[ -n "$_rel" ]]; do
    [[ -z "$_rel" ]] && continue
    case "$_rel" in
      /*) printf '%s\0' "$_rel" ;;
      *)  printf '%s\0' "$FILE_ROOT/$_rel" ;;
    esac
  done < "$FILELIST" | xargs -0 cat > /dev/null 2>/dev/null || true
  _pl_sec=$(( $(date +%s) - _pl_start ))
  _pl_mb=$(( _pl_bytes / 1024 / 1024 ))
  say "[preload] read $_pl_ok files (~${_pl_mb} MB) in ${_pl_sec}s; $_pl_miss missing (skipped)"
fi

# One discarded warm-up on just the first WARMUP_N files: the first real run
# otherwise pays venv/library load and other fixed first-run costs, penalising
# whichever point runs first. A small subset is enough to pay those once (the
# whole corpus does NOT need re-processing), so write the head of FILELIST to a
# temp list and run that. Phase WU keeps it out of the HB*/HC* measurement rows.
if [[ "$WARMUP" == "1" ]]; then
  rule; say "WARM-UP (discarded, phase WU) - first $WARMUP_N files"; rule
  WARMUP_FL="$LOG_DIR/warmup_filelist_${STAMP}.txt"
  head -n "$WARMUP_N" "$FILELIST" > "$WARMUP_FL"
  if [[ "$DO_STEP3" == "1" ]]; then
    run "warmup-s3-nw${WARMUP_W}" "$PY" bench_step3_indexing.py \
        --env-tag "$TAG" --num-workers "$WARMUP_W" \
        --filelist "$WARMUP_FL" --time-limit "$TIME_LIMIT" \
        --intraop-threads "$INTRAOP" \
        --phase WU --note "heavy-scaling warmup discarded"
  elif [[ "$DO_STEP1" == "1" ]]; then
    run "warmup-s1-mw${WARMUP_W}" "$PY" bench_step1_dataprep.py \
        --env-tag "$TAG" --max-workers "$WARMUP_W" \
        --filelist "$WARMUP_FL" --source-dir "$FILE_ROOT" \
        --intraop-threads "$INTRAOP" --time-limit-s "$TIME_LIMIT" \
        --phase WU --note "heavy-scaling warmup discarded"
  fi
fi

# ==========================================================================
# STEP 1 - encoding throughput vs max_workers
# ==========================================================================
if [[ "$DO_STEP1" == "1" ]]; then
  rule; say "STEP 1  encoding - $SIDE max_workers sweep: $WORKERS (intra-op $INTRAOP)"; rule
  for W in $WORKERS; do
    run "s1-${SIDE}-mw${W}" "$PY" bench_step1_dataprep.py \
        --env-tag "$TAG" --max-workers "$W" \
        --filelist "$FILELIST" --source-dir "$FILE_ROOT" \
        --intraop-threads "$INTRAOP" \
        --phase "$PH1" --time-limit-s "$TIME_LIMIT" \
        --note "heavy-scaling"
  done
fi

# ==========================================================================
# STEP 3 - embedding + FAISS indexing throughput vs num_workers
# ==========================================================================
if [[ "$DO_STEP3" == "1" ]]; then
  rule; say "STEP 3  indexing - $SIDE num_workers sweep: $WORKERS (intra-op $INTRAOP)"; rule
  for W in $WORKERS; do
    run "s3-${SIDE}-nw${W}" "$PY" bench_step3_indexing.py \
        --env-tag "$TAG" --num-workers "$W" \
        --filelist "$FILELIST" \
        --intraop-threads "$INTRAOP" \
        --time-limit "$TIME_LIMIT" \
        --phase "$PH3" --note "heavy-scaling"
  done
fi

rule; say "DONE"; rule
say "Results CSV : $BENCH_DIR/results/results.csv"
say "Rows        : phase $PH1 (Step1 max_workers sweep), phase $PH3 (Step3 num_workers sweep)"
say "intra-op $INTRAOP thread(s)/worker; with the cap on, throughput should hold"
say "near the physical-core count instead of dropping at high worker counts."
