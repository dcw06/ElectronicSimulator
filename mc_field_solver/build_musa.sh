#!/usr/bin/env bash
set -euo pipefail

MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
MUSA_ARCH="${MUSA_ARCH:-mp_22}"

export PATH="$MUSA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MUSA_HOME/lib:${LD_LIBRARY_PATH:-}"

if ! command -v mcc >/dev/null 2>&1; then
    echo "error: mcc was not found; set MUSA_HOME to the MUSA Toolkit directory" >&2
    exit 1
fi

mkdir -p build-musa
mcc -O3 -std=c++17 --cuda-gpu-arch="$MUSA_ARCH" \
    -Iinclude -DMC_HAVE_MUSA=1 \
    src/main.cpp src/domain.cpp src/io.cpp cpu/reference.cpp \
    kernels/monte_carlo.mu \
    -L"$MUSA_HOME/lib" -lmusart -lmurand \
    -o build-musa/mc_field_solver

echo "Built build-musa/mc_field_solver for $MUSA_ARCH"
