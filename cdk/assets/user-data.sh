#!/bin/bash
#
# HOOPS AI embeddings pipeline - EC2 bootstrap (Ubuntu 24.04 LTS, g6.8xlarge)
#
# Prepares a GPU box to run the 3-step embeddings pipeline (encode / train /
# index) from this repo, both as CPU and as GPU, so the two can be compared
# on identical hardware. Installs:
#   - the NVIDIA driver (matches the process already verified manually on a
#     g6.8xlarge / NVIDIA L4 instance; see README "known pitfalls")
#   - Xvfb as a systemd service (HOOPS AI needs an X display even for
#     offscreen/headless work)
#   - two Python venvs: CPU1.1 (CPU-only torch) and GPU1.1 (CUDA torch)
#   - this repo, cloned to ~/bench
#
# The HOOPS AI SDK itself (the `hoops_ai` package) is proprietary and is NOT
# installed here unless HOOPS_AI_SDK_URL is exported before this script runs
# (see install-sdk.sh, appended conditionally by the CDK stack). Without it,
# the venvs are created but `import hoops_ai` will fail until you install the
# SDK manually - see README.
#
set -eux
export DEBIAN_FRONTEND=noninteractive

BENCH_USER=ubuntu
BENCH_HOME=/home/$BENCH_USER
BENCH_DIR=$BENCH_HOME/bench
SDK_ROOT=$BENCH_HOME  # CPU1.1 / GPU1.1 venvs live directly under here

# 1. System packages
apt-get update
apt-get upgrade -y
apt-get install -y \
    git curl unzip build-essential tmux \
    python3 python3-venv python3-pip \
    xvfb libglu1-mesa mesa-utils xserver-xorg xinit \
    libgl1 libxrender1 libxext6 libsm6 \
    ubuntu-drivers-common

# 2. NVIDIA driver. This installs the kernel module but it is not loaded
#    until the instance reboots once - see README "verify the GPU driver".
if ! command -v nvidia-smi >/dev/null 2>&1; then
  apt-get install -y nvidia-driver-580-server || \
    apt-get install -y linux-headers-generic nvidia-driver-580-server
fi

# 3. Xvfb as a systemd service on :99. bench_step*.py / run_heavy_batch.sh
#    also auto-start Xvfb if $DISPLAY is unset, but a system-wide service
#    means every login shell (and cron/systemd job) sees a working DISPLAY
#    without having to remember to start one.
cat > /etc/systemd/system/xvfb.service <<'XVFB_EOF'
[Unit]
Description=Virtual X server for headless HOOPS AI rendering
After=network.target

[Service]
ExecStart=/usr/bin/Xvfb :99 -screen 0 1280x1024x24
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
XVFB_EOF
systemctl daemon-reload
systemctl enable --now xvfb.service

echo 'export DISPLAY=:99' > /etc/profile.d/hoops_ai_display.sh

# 4. Clone this repo. BENCH_REPO_URL is exported by the CDK stack before this
#    script runs (context `repoUrl`, defaults to the public GitHub repo).
mkdir -p "$BENCH_DIR"
if [[ -n "${BENCH_REPO_URL:-}" ]]; then
  sudo -u "$BENCH_USER" git clone "$BENCH_REPO_URL" "$BENCH_DIR"
fi

# 5. Two venvs: CPU-only torch and CUDA torch. hoops_ai itself is added by
#    install-sdk.sh if HOOPS_AI_SDK_URL was supplied; otherwise these venvs
#    are ready for you to `pip install` the SDK into manually.
sudo -u "$BENCH_USER" python3 -m venv "$SDK_ROOT/CPU1.1/.venv"
sudo -u "$BENCH_USER" "$SDK_ROOT/CPU1.1/.venv/bin/pip" install --upgrade pip
sudo -u "$BENCH_USER" "$SDK_ROOT/CPU1.1/.venv/bin/pip" install \
    torch --index-url https://download.pytorch.org/whl/cpu
if [[ -f "$BENCH_DIR/requirements.txt" ]]; then
  sudo -u "$BENCH_USER" "$SDK_ROOT/CPU1.1/.venv/bin/pip" install -r "$BENCH_DIR/requirements.txt"
fi

sudo -u "$BENCH_USER" python3 -m venv "$SDK_ROOT/GPU1.1/.venv"
sudo -u "$BENCH_USER" "$SDK_ROOT/GPU1.1/.venv/bin/pip" install --upgrade pip
sudo -u "$BENCH_USER" "$SDK_ROOT/GPU1.1/.venv/bin/pip" install \
    torch --index-url https://download.pytorch.org/whl/cu130
if [[ -f "$BENCH_DIR/requirements.txt" ]]; then
  sudo -u "$BENCH_USER" "$SDK_ROOT/GPU1.1/.venv/bin/pip" install -r "$BENCH_DIR/requirements.txt"
fi

mkdir -p "$BENCH_HOME/dataset"
chown -R "$BENCH_USER":"$BENCH_USER" "$BENCH_HOME"

echo "Bootstrap complete. Reboot once so the NVIDIA driver loads, then verify"
echo "with 'nvidia-smi'. Copy your CAD corpus into ~/dataset, then see the"
echo "repo README for how to install the HOOPS AI SDK (if not done via"
echo "HOOPS_AI_SDK_URL) and run the pipeline."
