#!/bin/bash
#
# Voxtral/Starling Prerequisites Installer
# Installs system-level dependencies on Ubuntu needed before running init.sh.
#
# Usage:
#   sudo ./prerequisites.sh          # Install everything
#   sudo ./prerequisites.sh --check  # Check only, don't install
#
# What this installs:
#   - Python 3.10+ with dev headers and venv
#   - FFmpeg (audio decoding)
#   - NVIDIA CUDA toolkit (GPU compute)
#   - Build tools (gcc, g++, cmake) needed by vLLM/Triton
#
# What this does NOT install:
#   - NVIDIA GPU drivers (install separately: sudo apt install nvidia-driver-550)
#   - vLLM, PyTorch, or Python packages (handled by init.sh)
#

set -e

# ─── Parse args ──────────────────────────────────────────────────────

CHECK_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=true ;;
    esac
done

# ─── Colors ──────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
fail() { echo -e "  ${RED}[MISSING]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
info() { echo -e "  $1"; }

# ─── Check current state ────────────────────────────────────────────

echo "========================================"
echo "  Prerequisites Check"
echo "========================================"
echo ""

ALL_OK=true

# Python 3.10+
if command -v python3 &>/dev/null; then
    PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYMAJOR=$(echo "$PYVER" | cut -d. -f1)
    PYMINOR=$(echo "$PYVER" | cut -d. -f2)
    if [ "$PYMAJOR" -ge 3 ] && [ "$PYMINOR" -ge 10 ]; then
        ok "Python $PYVER"
    else
        fail "Python $PYVER (need 3.10+)"
        ALL_OK=false
    fi
else
    fail "python3 not found"
    ALL_OK=false
fi

# Python dev headers
if python3 -c "import sysconfig; import os; assert os.path.exists(os.path.join(sysconfig.get_path('include'), 'Python.h'))" 2>/dev/null; then
    ok "Python dev headers"
else
    fail "Python dev headers (python3-dev)"
    ALL_OK=false
fi

# Python venv
if python3 -c "import venv" 2>/dev/null; then
    ok "Python venv module"
else
    fail "Python venv module (python3-venv)"
    ALL_OK=false
fi

# pip
if python3 -m pip --version &>/dev/null; then
    ok "pip"
else
    fail "pip (python3-pip)"
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
    GCCVER=$(gcc --version | head -1 | awk '{print $NF}')
    ok "GCC/G++ $GCCVER"
else
    fail "GCC/G++ (build-essential)"
    ALL_OK=false
fi

if command -v cmake &>/dev/null; then
    CMAKEVER=$(cmake --version | head -1 | awk '{print $3}')
    ok "CMake $CMAKEVER"
else
    fail "CMake"
    ALL_OK=false
fi

# NVIDIA driver
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(timeout 5 nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
    GPU_MEM=$(timeout 5 nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo "? MiB")
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "?")
    ok "NVIDIA driver $DRIVER_VER ($GPU_NAME, $GPU_MEM)"
else
    fail "NVIDIA driver (nvidia-smi not found)"
    info "       Install with: sudo apt install nvidia-driver-550"
    ALL_OK=false
fi

# CUDA toolkit
if command -v nvcc &>/dev/null; then
    CUDAVER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | cut -d, -f1)
    ok "CUDA toolkit $CUDAVER"
else
    # Check if CUDA is available via driver even without toolkit
    if nvidia-smi &>/dev/null; then
        CUDA_DRIVER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}')
        if [ -n "$CUDA_DRIVER" ]; then
            warn "CUDA toolkit not installed (nvcc missing), but driver CUDA $CUDA_DRIVER available"
            info "       vLLM may work without nvcc, but Triton kernels need it for best performance"
        else
            fail "CUDA toolkit (nvcc)"
        fi
    else
        fail "CUDA toolkit (nvcc)"
    fi
    ALL_OK=false
fi

# libsndfile (needed by soundfile Python package)
if ldconfig -p 2>/dev/null | grep -q libsndfile; then
    ok "libsndfile"
elif [ -f /usr/lib/x86_64-linux-gnu/libsndfile.so ] || [ -f /usr/lib/aarch64-linux-gnu/libsndfile.so ]; then
    ok "libsndfile"
else
    fail "libsndfile (libsndfile1-dev)"
    ALL_OK=false
fi

echo ""

# ─── Summary ─────────────────────────────────────────────────────────

if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}All prerequisites satisfied.${NC} You can run ./init.sh"
    exit 0
fi

if [ "$CHECK_ONLY" = true ]; then
    echo -e "${RED}Some prerequisites are missing.${NC} Run without --check to install them."
    exit 1
fi

# ─── Install missing packages ───────────────────────────────────────

echo "Installing missing prerequisites..."
echo ""

# Check we have root
if [ "$EUID" -ne 0 ] && ! sudo -n true 2>/dev/null; then
    echo -e "${RED}Error:${NC} Root privileges required to install packages."
    echo "  Run with: sudo ./prerequisites.sh"
    exit 1
fi

# Use sudo if not root
SUDO=""
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
fi

$SUDO apt-get update -qq

# Core packages
PACKAGES=(
    python3
    python3-dev
    python3-venv
    python3-pip
    ffmpeg
    build-essential
    cmake
    pkg-config
    libsndfile1-dev
    curl
    git
)

echo "Installing: ${PACKAGES[*]}"
$SUDO apt-get install -y -qq "${PACKAGES[@]}"

# CUDA toolkit — install from NVIDIA repo for latest version
if ! command -v nvcc &>/dev/null; then
    echo ""
    echo "Installing CUDA toolkit..."

    # Check if NVIDIA CUDA repo is already configured
    if ! apt-cache policy 2>/dev/null | grep -q "developer.download.nvidia.com\|cuda-repo"; then
        echo "  Adding NVIDIA CUDA repository..."
        # Detect architecture
        ARCH=$(dpkg --print-architecture)
        DISTRO=$(. /etc/os-release && echo "${ID}${VERSION_ID}" | tr -d '.')

        # Try to add the CUDA keyring package
        KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb"
        KEYRING_DEB="/tmp/cuda-keyring.deb"

        if curl -fsSL "$KEYRING_URL" -o "$KEYRING_DEB" 2>/dev/null; then
            $SUDO dpkg -i "$KEYRING_DEB"
            $SUDO apt-get update -qq
            rm -f "$KEYRING_DEB"
        else
            echo "  Could not add NVIDIA CUDA repo automatically."
            echo "  Visit: https://developer.nvidia.com/cuda-toolkit"
            echo "  Or install via: sudo apt install nvidia-cuda-toolkit"
        fi
    fi

    # Install CUDA toolkit (prefer cuda-toolkit meta-package)
    if apt-cache show cuda-toolkit &>/dev/null 2>&1; then
        echo "  Installing cuda-toolkit..."
        $SUDO apt-get install -y -qq cuda-toolkit
    elif apt-cache show nvidia-cuda-toolkit &>/dev/null 2>&1; then
        echo "  Installing nvidia-cuda-toolkit..."
        $SUDO apt-get install -y -qq nvidia-cuda-toolkit
    else
        echo "  CUDA toolkit package not found in repositories."
        echo "  Install manually from: https://developer.nvidia.com/cuda-toolkit"
    fi
fi

echo ""
echo "========================================"
echo "  Prerequisites Installation Complete"
echo "========================================"
echo ""

# Re-run check
echo "Verifying..."
echo ""
exec "$0" --check
