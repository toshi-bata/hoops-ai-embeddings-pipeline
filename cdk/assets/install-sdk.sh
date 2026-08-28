#!/bin/bash
#
# Optional: download and install the licensed HOOPS AI SDK, then write the
# license key + .env expected by bench_common.py / bench_tasks.py.
#
# Appended to user-data.sh by the CDK stack ONLY when HOOPS_AI_SDK_URL is
# supplied at deploy time (context `sdkUrl` / env HOOPS_AI_SDK_URL). Runs as
# root, right after user-data.sh has created the CPU1.1 / GPU1.1 venvs.
#
# NOTE for whoever deploys this: the exact archive layout / wheel name below
# is a best guess based on how the CPU1.1 / GPU1.1 venvs are laid out in the
# rest of this repo (a `packages/` folder with the trained checkpoints next
# to the installed SDK). Verify the glob patterns below against your actual
# HOOPS AI SDK download once, and adjust if the archive is shaped differently
# - this script deliberately fails loudly (`set -e`) rather than silently
# skipping the SDK install.
#
set -eu  # no -x: keep the presigned URL and license key out of the log

BENCH_USER=ubuntu
BENCH_HOME=/home/$BENCH_USER
SDK_ROOT=$BENCH_HOME
TMP_ARCHIVE=/tmp/hoops_ai_sdk.download

echo "[install-sdk] downloading HOOPS AI SDK..."
curl -fsSL "$HOOPS_AI_SDK_URL" -o "$TMP_ARCHIVE"

# The presigned URL query string hides the real extension; sniff the actual
# archive type instead of trusting the URL.
ARCHIVE_TYPE=$(file -b "$TMP_ARCHIVE")
EXTRACT_DIR=/opt/hoops_ai_sdk
mkdir -p "$EXTRACT_DIR"
case "$ARCHIVE_TYPE" in
  *Zip*)      unzip -q "$TMP_ARCHIVE" -d "$EXTRACT_DIR" ;;
  *gzip*)     tar xzf "$TMP_ARCHIVE" -C "$EXTRACT_DIR" ;;
  *tar*)      tar xf  "$TMP_ARCHIVE" -C "$EXTRACT_DIR" ;;
  *)          echo "[install-sdk] unrecognised archive type: $ARCHIVE_TYPE" >&2; exit 1 ;;
esac
rm -f "$TMP_ARCHIVE"

# Find the hoops_ai wheel(s) in the extracted tree and install into BOTH
# venvs (same package, different torch backend already installed by
# user-data.sh). Adjust this glob if your SDK ships a different layout
# (e.g. a plain `python -m pip install <dir>` against a setup.py/pyproject).
mapfile -t WHEELS < <(find "$EXTRACT_DIR" -iname 'hoops_ai*.whl')
if [[ ${#WHEELS[@]} -eq 0 ]]; then
  echo "[install-sdk] no hoops_ai*.whl found under $EXTRACT_DIR - archive layout" >&2
  echo "[install-sdk] does not match this script's assumption; install manually." >&2
  exit 1
fi
for venv in "$SDK_ROOT/CPU1.1/.venv" "$SDK_ROOT/GPU1.1/.venv"; do
  sudo -u "$BENCH_USER" "$venv/bin/pip" install "${WHEELS[@]}"
done

# Trained checkpoints (e.g. ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt), if
# bundled, are expected under */packages/trained_ml_models/ by
# bench_common.resolve_checkpoint(). Mirror them into both venv roots so
# either CPU1.1 or GPU1.1 resolves the warm-start checkpoint without
# HOOPS_AI_CKPT.
if [[ -d "$EXTRACT_DIR/packages/trained_ml_models" ]]; then
  for root in "$SDK_ROOT/CPU1.1" "$SDK_ROOT/GPU1.1"; do
    mkdir -p "$root/packages"
    cp -r "$EXTRACT_DIR/packages/trained_ml_models" "$root/packages/"
  done
fi

# License key + .env expected by bench_common.require_license()/apply_license().
if [[ -n "${HOOPS_AI_LICENSE:-}" ]]; then
  cat > "$BENCH_HOME/bench/.env" <<ENV_EOF
HOOPS_AI_LICENSE='$HOOPS_AI_LICENSE'
ENV_EOF
fi

chown -R "$BENCH_USER":"$BENCH_USER" "$SDK_ROOT/CPU1.1" "$SDK_ROOT/GPU1.1" "$BENCH_HOME/bench" 2>/dev/null || true

echo "[install-sdk] done."
