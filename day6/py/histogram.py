"""
Day 6: Histogram with Atomics
==============================

PROBLEM STATEMENT:
Compute histogram using atomic operations.

RACE CONDITION:
Multiple threads incrementing same bin = data corruption

SOLUTION:
cuda.atomic.add(histogram, bin_idx, 1)

Run with: $env:NUMBA_ENABLE_CUDASIM = "1"; python histogram.py
"""

import os
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

import numpy as np
from numba import cuda
import numba

NUM_BINS = 32

@cuda.jit
def histogram_atomic(data, histogram, min_val, max_val):
    # TODO: Get global index
    
    # TODO: Bounds check
    
    # TODO: Normalize value to [0, 1]
    # normalized = (data[idx] - min_val) / (max_val - min_val)
    
    # TODO: Compute bin index
    # bin_idx = int(normalized * NUM_BINS)
    # bin_idx = min(bin_idx, NUM_BINS - 1)
    
    # TODO: Atomic increment
    # cuda.atomic.add(histogram, bin_idx, 1)
    pass

def main():
    N = 10000
    
    # TODO: Create random data
    
    # TODO: Allocate histogram (zeros)
    
    # TODO: Launch kernel
    
    # TODO: Verify sum of bins == N
    pass

if __name__ == "__main__":
    main()
