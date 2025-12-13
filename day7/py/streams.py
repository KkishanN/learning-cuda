"""
Day 7: CUDA Streams
====================

PROBLEM STATEMENT:
Use streams to overlap operations on different data chunks.

NUMBA API:
stream = cuda.stream()
kernel[grid, block, stream](...)
stream.synchronize()

PATTERN:
for each chunk:
    stream = streams[i]
    copy to device (via stream)
    launch kernel in stream
    copy back (via stream)
synchronize all streams

Run with: $env:NUMBA_ENABLE_CUDASIM = "1"; python streams.py

NOTE: True concurrency only on real GPU, simulator runs sequentially.
"""

import os
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

import numpy as np
from numba import cuda
import time

@cuda.jit
def process_chunk(input_arr, output_arr, multiplier):
    # TODO: Simple processing
    # idx = cuda.grid(1)
    # if idx < input_arr.size:
    #     output_arr[idx] = input_arr[idx] * multiplier
    pass

def sequential_process(chunks):
    # TODO: Process each chunk one by one
    # No streams, just: to_device, kernel, copy_to_host, repeat
    pass

def stream_process(chunks):
    # TODO: Create streams: streams = [cuda.stream() for _ in chunks]
    
    # TODO: Launch all to different streams
    
    # TODO: Synchronize all streams
    pass

def main():
    num_chunks = 4
    chunk_size = 10000
    
    # TODO: Create data chunks
    
    # TODO: Time sequential processing
    
    # TODO: Time stream processing
    
    # TODO: Compare times
    pass

if __name__ == "__main__":
    main()
