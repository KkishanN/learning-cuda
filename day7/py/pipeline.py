"""
Day 7: Data Pipeline
=====================

PROBLEM STATEMENT:
Build multi-stage pipeline with overlapped execution.

STAGES:
1. normalize: (x - mean) / std
2. transform: apply activation (tanh, relu, etc.)
3. aggregate: reduce data size

Each chunk processed through all stages, 
different chunks at different stages concurrently.

Run with: $env:NUMBA_ENABLE_CUDASIM = "1"; python pipeline.py
"""

import os
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

import numpy as np
from numba import cuda
import numba

@cuda.jit
def normalize_kernel(data, output, mean, std):
    # TODO: output[i] = (data[i] - mean) / std
    pass

@cuda.jit
def transform_kernel(data, output):
    # TODO: Apply tanh or similar
    # output[i] = tanh(data[i])
    pass

@cuda.jit
def aggregate_kernel(data, output, agg_size):
    # TODO: Average every agg_size elements
    pass

class Pipeline:
    def __init__(self, chunk_size, agg_size=4):
        # TODO: Store config
        pass
    
    def process_chunk(self, data, stream=None):
        # TODO: Run all three stages
        # TODO: Use stream for all operations
        pass
    
    def process_all(self, chunks, use_streams=True):
        # TODO: If use_streams, create streams and process concurrently
        # TODO: Else process sequentially
        pass

def main():
    num_chunks = 4
    chunk_size = 1000
    
    # TODO: Create pipeline
    
    # TODO: Create data chunks
    
    # TODO: Process with and without streams
    
    # TODO: Compare times
    pass

if __name__ == "__main__":
    main()
