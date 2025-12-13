"""
Day 5: Tiled Matrix Multiplication
===================================

PROBLEM STATEMENT:
Implement C = A × B using tiled approach with shared memory.

NAIVE: 2N³ global memory reads
TILED: 2N³/TILE_SIZE reads (16× improvement for TILE_SIZE=16)

ALGORITHM:
1. Loop over tile pairs
2. Load tiles to shared memory
3. syncthreads()
4. Compute partial products
5. syncthreads()
6. Accumulate and write

Run with: $env:NUMBA_ENABLE_CUDASIM = "1"; python matrix_multiply.py
"""

import os
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

import numpy as np
from numba import cuda
import numba

TILE_SIZE = 16

@cuda.jit
def matmul_naive(A, B, C):
    # TODO: Get row and col
    
    # TODO: Bounds check
    
    # TODO: Dot product: loop k from 0 to A.shape[1]
    
    # TODO: Write to C[row, col]
    pass

@cuda.jit
def matmul_tiled(A, B, C):
    # TODO: Declare shared arrays
    # sA = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=numba.float32)
    # sB = cuda.shared.array(shape=(TILE_SIZE, TILE_SIZE), dtype=numba.float32)
    
    # TODO: Get tx, ty, row, col
    
    # TODO: Initialize result = 0.0
    
    # TODO: Loop over tiles
    
        # TODO: Load tile of A into sA
        
        # TODO: Load tile of B into sB
        
        # TODO: cuda.syncthreads()
        
        # TODO: Compute partial product
        
        # TODO: cuda.syncthreads()
    
    # TODO: Write result to C
    pass

def main():
    M, K, N = 64, 64, 64
    
    # TODO: Create random A and B matrices
    
    # TODO: Compute expected result with numpy: A @ B
    
    # TODO: Launch kernel
    
    # TODO: Verify result matches
    pass

if __name__ == "__main__":
    main()
