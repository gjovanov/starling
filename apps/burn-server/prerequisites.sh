#!/bin/bash
#
# Burn Server Prerequisites Installer
# Installs system-level dependencies needed before running init.sh.
#
# Usage:
#   ./prerequisites.sh          # Install everything
#   ./prerequisites.sh --check  # Check only, don't install
#
# What this installs:
#   - Rust toolchain (via rustup)
#   - Vulkan SDK + loader (GPU compute via wgpu)
#   - wasm-pack (for WASM browser builds)
#   - pkg-config, libsndfile (audio processing)
#   - FFmpeg (media file decoding)
#   - Build tools (gcc, g++, cmake)
#

set -e

CHECK_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=true ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
fail() { echo -e "  ${RED}[MISSING]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }

echo "========================================"
echo "  Burn Server Prerequisites Check"
echo "========================================"
echo ""

ALL_OK=true

# Rust
if command -v rustc &>/dev/null; then
    RUSTVER=$(rustc --version | awk '{print $2}')
    ok "Rust $RUSTVER"
else
    fail "Rust (rustc not found)"
    ALL_OK=false
fi

# Cargo
if command -v cargo &>/dev/null; then
    ok "Cargo"
else
    fail "Cargo"
    ALL_OK=false
fi

# wasm-pack (optional, for WASM builds)
if command -v wasm-pack &>/dev/null; then
    ok "wasm-pack ($(wasm-pack --version 2>/dev/null | awk '{print $2}'))"
else
    warn "wasm-pack not installed (needed for WASM browser builds)"
fi

# Vulkan (required by wgpu for GPU compute)
VULKAN_OK=false
if ldconfig -p 2>/dev/null | grep -q libvulkan; then
    VULKAN_OK=true
    # Try to get version from vulkaninfo
    if command -v vulkaninfo &>/dev/null; then
        VK_VER=$(vulkaninfo --summary 2>/dev/null | grep "apiVersion" | head -1 | awk '{print $NF}' || echo "")
        if [ -n "$VK_VER" ]; then
            ok "Vulkan $VK_VER"
        else
            ok "Vulkan (libvulkan found)"
        fi
    else
        ok "Vulkan (libvulkan found, vulkaninfo not installed)"
    fi
elif [ -f /usr/lib/x86_64-linux-gnu/libvulkan.so.1 ] || [ -f /usr/lib/aarch64-linux-gnu/libvulkan.so.1 ]; then
    VULKAN_OK=true
    ok "Vulkan (libvulkan.so.1 found)"
else
    fail "Vulkan (libvulkan1 + mesa-vulkan-drivers)"
    echo -e "         ${YELLOW}Vulkan is required by wgpu for GPU compute.${NC}"
    echo -e "         Install with: sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools"
    ALL_OK=false
fi

# FFmpeg
if command -v ffmpeg &>/dev/null; then
    FFVER=$(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')
    ok "FFmpeg $FFVER"
else
    fail "FFmpeg"
    ALL_OK=false
fi

# Build tools
if command -v gcc &>/dev/null && command -v g++ &>/dev/null; then
    ok "GCC/G++"
else
    fail "GCC/G++ (build-essential)"
    ALL_OK=false
fi

if command -v cmake &>/dev/null; then
    ok "CMake"
else
    fail "CMake"
    ALL_OK=false
fi

if command -v pkg-config &>/dev/null; then
    ok "pkg-config"
else
    fail "pkg-config"
    ALL_OK=false
fi

# libsndfile
if ldconfig -p 2>/dev/null | grep -q libsndfile; then
    ok "libsndfile"
elif [ -f /usr/lib/x86_64-linux-gnu/libsndfile.so ] || [ -f /usr/lib/aarch64-linux-gnu/libsndfile.so ]; then
    ok "libsndfile"
else
    fail "libsndfile (libsndfile1-dev)"
    ALL_OK=false
fi

# GPU (optional for native, not needed for WASM)
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(timeout 5 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
    GPU_MEM=$(timeout 5 nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "? MiB")
    ok "NVIDIA GPU: $GPU_NAME ($GPU_MEM)"
else
    warn "No NVIDIA GPU detected (Q4 needs ~700MB VRAM, BF16 needs ~9GB)"
fi

echo ""

if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}All prerequisites satisfied.${NC} You can run ./init.sh"
    exit 0
fi

if [ "$CHECK_ONLY" = true ]; then
    echo -e "${RED}Some prerequisites are missing.${NC} Run without --check to install them."
    exit 1
fi

# ─── Install missing ─────────────────────────────────────────────

echo "Installing missing prerequisites..."
echo ""

# Rust toolchain
if ! command -v rustc &>/dev/null; then
    echo "Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo "  Installed Rust $(rustc --version | awk '{print $2}')"
fi

# wasm-pack
if ! command -v wasm-pack &>/dev/null; then
    echo "Installing wasm-pack..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# System packages
SUDO=""
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
fi

$SUDO apt-get update -qq
$SUDO apt-get install -y -qq \
    ffmpeg \
    build-essential \
    cmake \
    pkg-config \
    libsndfile1-dev \
    curl \
    git \
    libvulkan1 \
    libvulkan-dev \
    mesa-vulkan-drivers \
    vulkan-tools

echo ""
echo "========================================"
echo "  Prerequisites Installation Complete"
echo "========================================"
echo ""

exec "$0" --check
