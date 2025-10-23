#!/usr/bin/env bash
set -euo pipefail

# Always run from this script’s directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LLVM_PREFIX="/opt/homebrew/opt/llvm"
CXX="${LLVM_PREFIX}/bin/clang++"

echo "Building with ${CXX} ..."
"${CXX}" -O3 -march=native -std=c++17 -fopenmp \
  -I"${LLVM_PREFIX}/include" kernel.cpp -o cpu_bench \
  -L"${LLVM_PREFIX}/lib" -Wl,-rpath,"${LLVM_PREFIX}/lib" -lomp

echo "Running microbenchmark ..."
./cpu_bench
