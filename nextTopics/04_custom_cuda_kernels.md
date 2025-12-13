# 04. Custom CUDA Kernels

## 🎯 Overview
Writing custom CUDA kernels differentiates you from developers who only use high-level frameworks. Essential for specialized optimizations.

---

## 🔧 When to Write Custom Kernels?

| Situation | Example |
|-----------|---------|
| Operation not in PyTorch | Custom attention variants |
| Fusing multiple ops | MatMul + Bias + Activation |
| Memory-bound ops | Custom data layouts |
| Specialized hardware | Tensor Core utilization |

---

## 📝 CUDA C++ Basics

### Kernel Structure
```cpp
// Kernel definition
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Launch
int blockSize = 256;
int numBlocks = (n + blockSize - 1) / blockSize;
vector_add<<<numBlocks, blockSize>>>(a, b, c, n);
cudaDeviceSynchronize();
```

### Memory Qualifiers
```cpp
__global__ void kernel() {
    // Shared memory (per block)
    __shared__ float cache[256];
    
    // Registers (per thread) - default
    float local_var;
    
    // Global memory (device)
    // Passed as kernel arguments
}
```

---

## 🐍 PyTorch CUDA Extensions

### Setup (setup.py)
```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_ops',
    ext_modules=[
        CUDAExtension('custom_ops', [
            'custom_ops.cpp',
            'custom_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### C++ Wrapper (custom_ops.cpp)
```cpp
#include <torch/extension.h>

torch::Tensor custom_forward(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &custom_forward, "Custom forward");
}
```

### CUDA Kernel (custom_kernel.cu)
```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // Fused: ReLU + Multiply by 2
        output[idx] = max(0.0f, x) * 2.0f;
    }
}

torch::Tensor custom_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output;
}
```

---

## 🔺 Triton Language

OpenAI's Triton: Write GPU kernels in Python!

### Why Triton?
- Python syntax (no C++)
- Automatic optimization
- Portable across GPUs
- Easier debugging

### Example: Fused MatMul + ReLU
```python
import triton
import triton.language as tl

@triton.jit
def fused_matmul_relu_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main loop
    for k in range(0, K, BLOCK_K):
        a = tl.load(A + offs_m[:, None] * stride_am + (k + offs_k) * stride_ak)
        b = tl.load(B + (k + offs_k)[:, None] * stride_bk + offs_n * stride_bn)
        acc += tl.dot(a, b)
    
    # Apply ReLU and store
    acc = tl.maximum(acc, 0.0)  # Fused ReLU!
    tl.store(C + offs_m[:, None] * stride_cm + offs_n * stride_cn, acc)
```

---

## ⚡ Optimization Techniques

### 1. Warp-Level Primitives
```cpp
// Warp reduce (no shared memory needed!)
float val = /* thread value */;
for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
}
// Thread 0 has sum of all 32 threads
```

### 2. Tensor Cores
```cpp
#include <mma.h>
using namespace nvcuda::wmma;

// Declare fragments
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

// Load, compute, store
load_matrix_sync(a_frag, A_ptr, 16);
load_matrix_sync(b_frag, B_ptr, 16);
fill_fragment(c_frag, 0.0f);
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(C_ptr, c_frag, 16, mem_row_major);
```

---

## 💼 Interview Topics

- Memory coalescing in custom kernels
- When to use shared vs register memory
- Thread divergence and how to avoid it
- Occupancy and its effect on performance
- Debugging CUDA kernels (cuda-gdb, printf)
- Triton vs CUDA C++ trade-offs
