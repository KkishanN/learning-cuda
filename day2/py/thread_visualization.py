"""
Day 2: Thread Visualization
===========================

PROBLEM STATEMENT:
Write a CUDA kernel that records each thread's indices into output arrays.
This helps visualize how threads are organized in blocks and grids.

REQUIREMENTS:
1. Each thread should write its blockIdx.x to output_block_idx array
2. Each thread should write its threadIdx.x to output_thread_idx array  
3. Each thread should calculate and write its global ID to output_global_idx
4. Handle bounds checking

FORMULA:
global_id = blockIdx.x * blockDim.x + threadIdx.x

Run with: $env:NUMBA_ENABLE_CUDASIM = "1"; python thread_visualization.py
"""

import os
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

import numpy as np
from numba import cuda

@cuda.jit
def record_thread_info(output_block_idx, output_thread_idx, output_global_idx):
    # TODO: Get block and thread indices using cuda.blockIdx.x, cuda.threadIdx.x
    
    # TODO: Calculate global thread ID
    
    # TODO: Bounds check using output_global_idx.shape[0]
    
    # TODO: Write indices to output arrays
    pass

def main():
    THREADS_PER_BLOCK = 4
    NUM_BLOCKS = 3
    TOTAL_THREADS = THREADS_PER_BLOCK * NUM_BLOCKS
    
    # TODO: Allocate device arrays using cuda.device_array()
    
    # TODO: Launch kernel with [NUM_BLOCKS, THREADS_PER_BLOCK]
    
    # TODO: Copy results to host using .copy_to_host()
    
    # TODO: Print results in table format
    pass

if __name__ == "__main__":
    main()
