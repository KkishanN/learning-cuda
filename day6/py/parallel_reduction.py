"""
Day 6: Parallel Reduction (Sum)
================================

PROBLEM STATEMENT:
Compute array sum using tree reduction.

ALGORITHM:
stride = blockDim.x // 2
while stride > 0:
    if tid < stride:
        shared[tid] += shared[tid + stride]
    syncthreads()
    stride //= 2

COMPLEXITY: O(log N) parallel steps

Run with: $env:NUMBA_ENABLE_CUDASIM = "1"; python parallel_reduction.py
"""

import os
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

import numpy as np
from numba import cuda
import numba

BLOCK_SIZE = 256

@cuda.jit
def reduce_sum(data, partial_sums):
    # TODO: Declare shared memory
    # shared = cuda.shared.array(shape=(BLOCK_SIZE,), dtype=numba.float32)
    
    # TODO: Get tid and gid
    
    # TODO: Load into shared (with bounds check)
    
    # TODO: cuda.syncthreads()
    
    # TODO: Tree reduction loop
    
    # TODO: Thread 0 writes to partial_sums[blockIdx.x]
    pass

def main():
    N = 10000
    
    # TODO: Create array of ones (sum should = N)
    
    # TODO: Calculate num_blocks, allocate partial_sums
    
    # TODO: Launch kernel
    
    # TODO: Sum partials on CPU
    
    # TODO: Verify sum == N
    pass

if __name__ == "__main__":
    main()
