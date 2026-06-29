#!/usr/bin/env bash
# Run on a Vast.ai Ubuntu/Debian CUDA host before benchmarking.
# Installs CUDA toolkit pieces, Rust, uv, build tools, and verifies the GPU.

set -euo pipefail

echo "=== infer - Vast.ai CUDA setup ==="

if ! command -v apt-get >/dev/null 2>&1; then
    echo "error: this setup script requires an Ubuntu/Debian-based instance."
    exit 1
fi

SUDO=()
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    SUDO=(sudo)
fi

ensure_bashrc_line() {
    local marker="$1"
    local line="$2"

    touch "$HOME/.bashrc"
    if ! grep -q "$marker" "$HOME/.bashrc" 2>/dev/null; then
        printf '\n%s\n' "$line" >> "$HOME/.bashrc"
    fi
}

echo "==> installing build tools and utilities..."
"${SUDO[@]}" apt-get update -qq
"${SUDO[@]}" apt-get install -y -qq \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    less \
    libssl-dev \
    pkg-config \
    tmux \
    vim \
    wget

CUDA_VERSION="${CUDA_VERSION:-12.8}"
CUDA_MAJOR="$(echo "$CUDA_VERSION" | cut -d. -f1-2)"
CUDA_PACKAGE="cuda-toolkit-${CUDA_MAJOR/./-}"

if command -v nvcc >/dev/null 2>&1; then
    echo "==> nvcc already installed: $(nvcc --version | grep release)"
else
    echo "==> installing CUDA toolkit $CUDA_MAJOR..."
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
    "${SUDO[@]}" dpkg -i /tmp/cuda-keyring.deb
    rm /tmp/cuda-keyring.deb
    "${SUDO[@]}" apt-get update -qq
    "${SUDO[@]}" apt-get install -y -qq "$CUDA_PACKAGE"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

ensure_bashrc_line "CUDA_HOME" "# CUDA
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"

if command -v cargo >/dev/null 2>&1; then
    echo "==> cargo already installed: $(cargo --version)"
else
    echo "==> installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal
    export PATH="$HOME/.cargo/bin:$PATH"
    ensure_bashrc_line ".cargo/bin" "export PATH=\$HOME/.cargo/bin:\$PATH"
fi

if command -v uv >/dev/null 2>&1; then
    echo "==> uv already installed: $(uv --version)"
else
    echo "==> installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    ensure_bashrc_line ".local/bin" "export PATH=\$HOME/.local/bin:\$PATH"
fi

echo
echo "=== Verification ==="
echo
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "  (nvidia-smi not found - driver may not be loaded)"
echo
echo "nvcc:"
nvcc --version 2>/dev/null || echo "  (nvcc not found)"
echo
echo "cargo:"
cargo --version 2>/dev/null || echo "  (cargo not found)"
echo
echo "uv:"
uv --version 2>/dev/null || echo "  (uv not found)"
echo
echo "=== Vast setup done ==="
