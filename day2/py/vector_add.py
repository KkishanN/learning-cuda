"""
Day 2: Vector Addition
======================

PROBLEM STATEMENT:
Implement parallel vector addition: C = A + B

REQUIREMENTS:
1. Each thread computes c[i] = a[i] + b[i]
2. Handle bounds checking
3. BONUS: Implement grid-stride loop for large arrays

Run with: $env:NUMBA_ENABLE_CUDASIM = "1"; python vector_add.py
"""

import os
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

import numpy as np
from numba import cuda

@cuda.jit
def vector_add(a, b, c):
    # TODO: Calculate global index
    # TODO: Bounds check
    # TODO: Perform addition
    pass

@cuda.jit
def vector_add_grid_stride(a, b, c):
    # TODO: Calculate starting index
    # TODO: Calculate stride = blockDim.x * gridDim.x
    # TODO: While loop to process multiple elements
    pass

def main():
    N = 10000
    
    # TODO: Create input arrays a and b
    # TODO: Copy to device
    # TODO: Launch kernel
    # TODO: Verify result matches numpy
    pass

if __name__ == "__main__":
    main()
