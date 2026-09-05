# Idempotent-Core: Native C++20 and CUDA Zero-Copy In-Place Compaction Engine

[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.4%20sm__120%20Blackwell-green.svg)]()
[![Patent](https://img.shields.io/badge/Patent-US_64%2F148%2C668-red.svg)]()

**Idempotent-Core** is the unified foundational native runtime engine for the **Idempotent Permutations** ecosystem. It provides direct, zero-dependency C++20 templates, native CUDA kernels, and a pure C ABI (`libidempotent.so` / `idempotent.dll`) for ultra-high-throughput in-place tensor and array compaction across CPU and GPU architectures.

Protected under **U.S. Patent Application No. 64/148,668** (*"Patent Pending"*).  
**Author:** Dr. A. Emre ÇETİN (`aemre.cetin@gmail.com`)  
**Affiliation:** Computational Systems and Cognitive Architectures, Izmir, Turkey  

---

## Key Features

1. **Header-Only C++20 Core (`include/idempotent/idempotent.hpp`):**
   - Zero external dependencies.
   - Strictly $O(1)$ scalar auxiliary register memory (no bitmasks).
   - In-register 2-cycle transposition fast-path.
2. **Native CUDA Engine (`include/idempotent/cuda/idempotent_cuda.cuh`):**
   - Direct CUDA execution for standalone C++ inference pipelines (Triton Inference Server, robotics, embedded edge devices).
   - Tested and verified on **NVIDIA RTX PRO 500 Blackwell Generation GPU (`sm_120`)**.
3. **Pure C ABI (`include/idempotent/c_api.h`):**
   - ABI-stable functions (`idempotent_compact_f32`, `idempotent_compact_f16`, `idempotent_compact_1d_i64`) enabling zero-overhead FFI bindings for Rust, Go, C#, and Java.
4. **Python PyTorch Bridge:**
   - Seamless interoperability with PyTorch tensors on CPU and GPU.

---

## C++ Quickstart

```cpp
#include <idempotent/idempotent.hpp>

// 2D Tensor Compaction along N dimension: [N, D]
idempotent::compact_inplace<float>(data, target_map, N, D);

// 1D Array Compaction (HFT Order Books, Graph Edges): [N]
idempotent::compact_1d_inplace<int64_t>(ids, target_map, N);
```

## Python Quickstart

```python
import torch
import idempotent_core

data = torch.randn((1024, 128), dtype=torch.float32)
scores = torch.rand(1024, dtype=torch.float32)

# Generate idempotent projection map
target_map = idempotent_core.generate_idempotent_map(scores, capacity=256)

# In-place compaction (zero auxiliary memory)
idempotent_core.compact_inplace(data, target_map)

# Retained active slice
active_data = data[:256]
```