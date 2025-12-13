"""
Day 3: Dot Product with Reduction
=================================

PROBLEM STATEMENT:
Compute dot product using shared memory reduction.

ALGORITHM:
1. Each thread computes a[i] * b[i]
2. Store in shared memory
3. Tree reduction: stride = blockDim.x // 2, then stride //= 2
4. Thread 0 writes block's partial sum
5. Sum partials on CPU

KEY PATTERN:
stride = blockDim.x // 2
while stride > 0:
    if tid < stride:
        shared[tid] += shared[tid + stride]
    syncthreads()
    stride //= 2

Run with: $env:NUMBA_ENABLE_CUDASIM = "1"; python dot_product.py
"""

import os
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

import numpy as np
from numba import cuda
import numba

BLOCK_SIZE = 256

@cuda.jit
def dot_product(a, b, partial_sums):
    # TODO: Declare shared array: cuda.shared.array(shape=(BLOCK_SIZE,), dtype=numba.float32)
    
    # TODO: Get thread id and global id
    
    # TODO: Compute product and store in shared memory (with bounds check)
    
    # TODO: cuda.syncthreads()
    
    # TODO: Tree reduction loop
    
    # TODO: Thread 0 writes to partial_sums[blockIdx.x]
    pass

def main():
    N = 10000
    
    # TODO: Create vectors a and b
    
    # TODO: Calculate expected result: np.dot(a, b)
    
    # TODO: Allocate partial sums array (num_blocks size)
    
    # TODO: Launch kernel
    
    # TODO: Sum partial results
    
    # TODO: Compare with expected
    pass

if __name__ == "__main__":
    main()
