

# 🧠 Adaptive Hybrid Runtime – Part 1 (CPU Path)

This project implements the **CPU-side runtime** of an *Adaptive Hybrid Runtime* for Sparse Tensor Contractions.  
The runtime is designed to **adaptively distribute tensor computations** between CPU and GPU based on block density — optimizing cache locality, loop-level parallelism, and vectorization.

---

## 🚀 Project Goal

The objective of **Part 1 (CPU Path)** is to:
- Optimize computations on **sparse multidimensional data** using **OpenMP + SIMD**.
- Implement core **locality-aware optimizations**:
  - Scalar expansion  
  - Loop interchange / distribution  
  - Loop unrolling  
  - 1D tiling (cache blocking)
- Benchmark performance and select the **best-performing variant** automatically.

---

## 🧩 Project Overview

The runtime pipeline consists of four main stages:

| Stage | Description |
|--------|--------------|
| **1. Data Preparation** | Download the OrganMNIST3D dataset (medical 3D images) and save it as `.npz`. |
| **2. Block Tiling & Density Analysis** | Split each 3D volume into smaller 14³ tiles and compute block density (ratio of non-zero entries). |
| **3. CPU Data Extraction** | Select sparse blocks (density < 0.3) and flatten them into 1D vectors for CPU computation. |
| **4. CPU Optimization Benchmark** | Apply OpenMP-based kernel variants (loop unrolling, tiling, etc.) and record runtime, GFLOPs, and cache locality performance. |

---

## 🧠 System Requirements

### 🖥️ Operating System
- macOS (Apple Silicon or Intel)
- Linux (Ubuntu 20.04 or later)
- *Windows WSL2 is also compatible 

### 🧰 Software Requirements

| Component | Version | Purpose |
|------------|----------|----------|
| **Python** | 3.10+ | Data preprocessing, tiling, and vector preparation |
| **Conda** | 4.12+ | Virtual environment management |
| **LLVM (clang++)** | ≥ 15.0 | C++ compilation with OpenMP (macOS recommended) |
| **GCC (g++-14)** | optional | Alternative compiler (OpenMP built-in) |
| **CMake** | optional | Build configuration (if used later) |

---

## 📦 Python Dependencies

| Library | Purpose |
|----------|----------|
| `numpy` | Array and tensor manipulation |
| `pandas` | Manifest management and CSV I/O |
| `tqdm` | Progress visualization |
| `torch` & `torchvision` | Required for `medmnist` backend |
| `medmnist` | OrganMNIST3D dataset download |

---

## Environment Setup

### Prerequisites

- **Operating System**: macOS (Intel or Apple Silicon)
- **Python**: 3.10 or higher
- **Compiler**: GCC with OpenMP support or LLVM Clang 

### Installation Steps



#### 1.Create Conda Environment and install dependencies

```bash
conda env create -f environment.yml python=3.10
conda activate hrt
```


---

## Project Structure

```
hybridrt/
├── 01_download.py          # Dataset acquisition module
├── 02_make_tiles.py        # Tensor decomposition and density analysis
├── prep_vectors.py         # CPU vector preparation utility
├── cpu/
│   ├── kernel.cpp          # OpenMP-accelerated kernel implementations
│   ├── run_cpu.sh          # Automated build and benchmark script
│   └── cpu_bench           # Compiled benchmark executable (generated)
├── blocks/                 # 3D tensor tiles (*.npy files)
├── reports/                # Performance benchmark results (JSON)
│   └── cpu_autotune.json
├── vectors.bin             # Flattened input vectors (binary)
├── W.bin                   # Weight matrix (binary)
├── sizes.json              # Tensor dimension metadata
└── environment.yml         # Conda environment specification
```

---

## Implementation Guide

### Step 1: Dataset Acquisition

Download the OrganMNIST3D dataset, a collection of 3D medical imaging volumes.

**Command:**

```bash
python 01_download.py 
```


**Output:**

```
Saved: ./data/organmnist3d_train.npz
Dataset shape: imgs=(972, 1, 28, 28, 28)
```

**Generated Files:**
- `organmnist3d_train.npz`: Compressed dataset (972 volumes, 28×28×28 resolution)

---

### Step 2: Tensor Decomposition

Split 3D volumes into smaller blocks and compute density metrics.

**Command:**

```bash
python make_tiles.py
```

**Processing Logic:**

```python
# For each volume:
#   1. Extract non-overlapping 3D blocks
#   2. Calculate density = count_nonzero(block) / block.size
#   3. Assign target device based on density threshold
#   4. Save block as .npy file
```

**Output:**

```
Loaded dataset: (972, 1, 28, 28, 28)
Tiling volumes: 100%|████████████████| 972/972 [00:05<00:00, 190.4it/s]
Generated 15,552 blocks
Wrote blocks_manifest.csv
```

**Generated Files:**
- `blocks/blk_0000000.npy` through `blocks/blk_0015551.npy`
- `blocks_manifest.csv` containing:
  - Block ID
  - Density value
  - Target device (CPU/GPU)
  - Dimensions

---

### CPU Vector Preparation

Select sparse blocks, flatten to 1D vectors, and generate weight matrix.

**Command:**

```bash
python prep_vectors.py
```

**Parameters** (configurable in script):
- `num_samples`: Number of CPU blocks to process (default: 100)
- `vector_size`: Flattened block dimension

**Output:**

```
Selected 100 CPU-tagged blocks
Preparing vectors: 100%|██████████| 100/100 [00:02<00:00]
```

**Generated Files:**
- `vectors.bin`: Concatenated input vectors (binary format)
- `W.bin`: Shared weight matrix for tensor operations
- `sizes.json`: Metadata (dimensions, data types, byte offsets)

---

### Compilation and Benchmarking

Run the following commands from the `cpu/` directory:


```bash

 cd cpu  
 
```

```bash
# Using Homebrew GCC
GXX=$(ls /opt/homebrew/bin/g++-* 2>/dev/null | head -n1)
echo "Using $GXX"
"$GXX" -O3 -march=native -std=c++17 -fopenmp kernel.cpp -o cpu_bench

# or, using LLVM Clang
LLVM_PREFIX="$(brew --prefix llvm)"
"$LLVM_PREFIX/bin/clang++" -O3 -march=native -std=c++17 -fopenmp \
  -I"$LLVM_PREFIX/include" kernel.cpp -o cpu_bench \
  -L"$LLVM_PREFIX/lib" -Wl,-rpath,"$LLVM_PREFIX/lib" -lomp

```


---


## Benchmarking Results

### Sample Output

```
===== CPU Kernel Benchmark =====

   baseline  time=2.003 ms  GFLOP/s=14.0282
   scalar_unroll_4  time=0.795 ms  GFLOP/s=35.3441
   scalar_unroll_8  time=0.612 ms  GFLOP/s=45.9127
   loop_interchange  time=0.356 ms  GFLOP/s=78.9285
              ile_i  time=0.331 ms  GFLOP/s=84.8899

BEST VARIANT: tile_i


===================================
OPTIMAL KERNEL: tile_i
===================================
```

### Performance Metrics

- **Execution Time**: Wall-clock time (milliseconds)
- **Throughput**: Giga floating-point operations per second
- **Speedup**: Relative performance vs. baseline

### Result Persistence

Benchmark data is automatically saved to:

```
reports/cpu_autotune.json
```

**JSON Schema:**

```json
{
  "timestamp": "2025-10-22T14:30:00Z",
  "system_info": {
    "cpu": "Apple M1 Pro",
    "cores": 10,
    "compiler": "clang-15.0.0"
  },
  "benchmarks": [
    {
      "variant": "tile_i",
      "time_ms": 52.3,
      "gflops": 21.09,
      "speedup": 2.34
    }
  ],
  "optimal_variant": "tile_i"
}
```

---