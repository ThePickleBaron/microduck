#!/usr/bin/env bash
# One-shot setup for Linux or WSL2 Ubuntu (24.04 recommended).
#
#   cd ~/microduck-pretraining && bash setup/setup.sh
#
# Installs system libraries (needs sudo), uv, the Python environment (the
# vendored Microduck training stack + this project), downloads the vendor
# baseline policies, and runs md-check. Safe to re-run.
#
# Options:
#   --no-apt      skip the apt step (no sudo, or packages already installed)
#   --no-policy   skip downloading the vendor baseline policies
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"
DO_APT=1
DO_POLICY=1
for arg in "$@"; do
  case "$arg" in
    --no-apt) DO_APT=0 ;;
    --no-policy) DO_POLICY=0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

if [[ "$REPO" == /mnt/* ]]; then
  echo "WARNING: the repo is on the Windows drive ($REPO)."
  echo "         Builds and training are much faster from the Linux filesystem:"
  echo "         git clone it into ~/ inside WSL instead. Continuing anyway."
fi

if [[ $DO_APT -eq 1 ]] && command -v apt-get >/dev/null; then
  step "System packages (sudo)"
  sudo apt-get update -qq
  # git/curl/build tools for uv + the BAM git dependency; GL/GLFW/EGL for the
  # MuJoCo viewer (WSLg provides the display on Windows 11).
  sudo apt-get install -y -qq \
    git curl ca-certificates build-essential \
    libgl1 libglx-mesa0 libegl1 libglfw3 libosmesa6 mesa-utils
fi

if ! command -v uv >/dev/null; then
  step "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

step "Python environment (first run downloads ~5 GB of PyTorch/CUDA wheels)"
uv sync --extra dev

if [[ $DO_POLICY -eq 1 ]]; then
  step "Vendor baseline policies (C1)"
  uv run python scripts/get_vendor_policy.py || echo "  (download failed - re-run later: uv run python scripts/get_vendor_policy.py)"
fi

step "Checking the install"
uv run md-check || true

cat <<'EOF'

Setup finished. Next:
  uv run md-conditions                 # what the four conditions are
  uv run pytest -q -m "not slow"       # fast tests
  uv run md-train c3 --dry-run         # see the training command
  uv run md-train c3                   # train (desktop with the RTX 3090)
Full workflow: README.md
EOF
