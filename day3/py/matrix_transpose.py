"""
Day 3: Matrix Transpose with Shared Memory
==========================================

PROBLEM STATEMENT:
Implement matrix transpose comparing naive vs shared memory approach.

PART 1 - NAIVE:
- Read: coalesced (consecutive columns)
- Write: non-coalesced (strided by width)

PART 2 - SHARED MEMORY:
- Load tile to shared memory (coalesced read)
- syncthreads()
- Write from shared with transposed indices (coalesced write)

KEY CONCEPT:
Use TILE_SIZE+1 padding in shared memory to avoid bank conflicts.

Run with: $env:NUMBA_ENABLE_CUDASIM = "1"; python matrix_transpose.py
"""

import os
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

import numpy as np
from numba import cuda
import numba

TILE_SIZE = 16

@cuda.jit
def transpose_naive(input_matrix, output_matrix):
    # TODO: Get global x, y using blockIdx, blockDim, threadIdx
    
    # TODO: Bounds check using input_matrix.shape
    
    # TODO: output[x, y] = input[y, x]
    pass

@cuda.jit
def transpose_shared(input_matrix, output_matrix):
    # TODO: Declare shared memory with shape (TILE_SIZE, TILE_SIZE + 1)
    # tile = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE + 1), dtype=numba.float32)
    
    # TODO: Calculate global and local coordinates
    
    # TODO: Load into shared memory
    
    # TODO: cuda.syncthreads()
    
    # TODO: Calculate transposed output coordinates (swap blockIdx.x and blockIdx.y)
    
    # TODO: Write from shared memory
    pass

def main():
    M, N = 64, 64
    
    # TODO: Create random input matrix
    
    # TODO: Copy to device
    
    # TODO: Configure kernel launch (grid and block)
    
    # TODO: Launch both transpose versions
    
    # TODO: Verify results match numpy transpose
    pass

if __name__ == "__main__":
    main()
