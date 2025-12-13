"""
Day 5: Neural Network Layer
============================

PROBLEM STATEMENT:
Implement Y = ReLU(X @ W + b)

FUSED OPERATIONS:
1. Matrix multiply: X @ W
2. Bias addition: + b
3. ReLU: max(0, x)

All in one kernel = no intermediate memory writes!

Run with: $env:NUMBA_ENABLE_CUDASIM = "1"; python neural_layer.py
"""

import os
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

import numpy as np
from numba import cuda
import numba

TILE_SIZE = 16

@cuda.jit
def dense_layer_fused(X, W, bias, Y):
    # TODO: Use tiled matmul approach
    
    # TODO: After computing matrix product value...
    
    # TODO: Add bias: result += bias[col]
    
    # TODO: Apply ReLU: result = max(0.0, result)
    
    # TODO: Write to Y[row, col]
    pass

def main():
    batch = 32
    in_features = 64
    out_features = 32
    
    # TODO: Create X, W, bias
    
    # TODO: Expected: np.maximum(0, X @ W + bias)
    
    # TODO: Launch kernel
    
    # TODO: Verify result
    pass

if __name__ == "__main__":
    main()
