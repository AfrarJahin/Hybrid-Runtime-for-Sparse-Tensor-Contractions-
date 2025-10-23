

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

## ⚙️ Environment Setup (macOS Example)


Hpc-project-part1/
│
├── 01_download.py           # Download OrganMNIST3D → .npz
├── make_tiles.py            # Split 3D volumes into tiles & compute density
├── prep_vectors.py          # Prepare CPU vectors and weight matrices
│
├── blocks/                  # Folder containing saved 3D blocks (.npy)
├── blocks_manifest.csv      # CSV file with block density & routing metadata
│
├── cpu/
│   ├── kernel.cpp           # OpenMP/SIMD kernels for CPU optimization
│   └── run_cpu.sh           # Build-and-run script (auto-detects LLVM/GCC)
│
├── reports/
│   └── cpu_autotune.json    # Benchmark performance results (runtime, GFLOPs)
│
└── README.md      

## Project Structure
Hpc-project-part1/
│
├── 01_download.py           # Download OrganMNIST3D → .npz
├── make_tiles.py            # Split 3D volumes into tiles & compute density
├── prep_vectors.py          # Prepare CPU vectors and weight matrices
│
├── blocks/                  # Folder containing saved 3D blocks (.npy)
├── blocks_manifest.csv      # CSV file with block density & routing metadata
│
├── cpu/
│   ├── kernel.cpp           # OpenMP/SIMD kernels for CPU optimization
│   └── run_cpu.sh           # Build-and-run script (auto-detects LLVM/GCC)
│
├── reports/
│   └── cpu_autotune.json    # Benchmark performance results (runtime, GFLOPs)
│
└── README.md                # Project documentation (this file)


## Environment Setup

### Prerequisites

- **Operating System**: macOS (Intel or Apple Silicon)
- **Python**: 3.10 or higher
- **Compiler**: GCC with OpenMP support or LLVM Clang 

### Installation Steps



#### 1.Create Conda Environment

```bash
conda env create -f environment.yml python=3.10
conda activate hrt
```

#### 4. Install Python Dependencies

```bash
pip install numpy pandas tqdm medmnist torch torchvision
```

**Note**: CUDA/GPU dependencies are not required for Part 1.

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

Compile the CPU kernels and execute performance benchmarks.

#### Compilation Options


---


## Benchmarking Results

### Sample Output

```
===== CPU Kernel Benchmark =====

Variant: baseline
  Execution time: 122.4 ms
  Throughput:     8.93 GFLOP/s

Variant: scalar_unroll_4
  Execution time: 78.2 ms
  Throughput:     14.02 GFLOP/s
  Speedup:        1.57×

Variant: scalar_unroll_8
  Execution time: 65.7 ms
  Throughput:     16.74 GFLOP/s
  Speedup:        1.86×

Variant: loop_interchange
  Execution time: 59.1 ms
  Throughput:     18.55 GFLOP/s
  Speedup:        2.07×

Variant: tile_i
  Execution time: 52.3 ms
  Throughput:     21.09 GFLOP/s
  Speedup:        2.34×

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