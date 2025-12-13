"""
Day 4: Sobel Edge Detection
===========================

PROBLEM STATEMENT:
Detect edges using Sobel operator.

SOBEL KERNELS:
Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
Gy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

ALGORITHM:
1. Convolve with Gx -> horizontal gradient
2. Convolve with Gy -> vertical gradient  
3. Magnitude = sqrt(Gx² + Gy²)

Run with: $env:NUMBA_ENABLE_CUDASIM = "1"; python sobel_edge.py
"""

import os
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

import numpy as np
from numba import cuda
import numba

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

@cuda.jit
def sobel_edge(input_img, output_img, sobel_x, sobel_y):
    # TODO: Get 2D coordinates
    
    # TODO: Skip boundary pixels
    
    # TODO: Compute Gx by convolving with sobel_x
    
    # TODO: Compute Gy by convolving with sobel_y
    
    # TODO: Magnitude = (gx*gx + gy*gy) ** 0.5
    
    # TODO: Write to output
    pass

def main():
    size = 64
    
    # TODO: Create test image with clear edges
    # (e.g., rectangle in center)
    
    # TODO: Run Sobel detection
    
    # TODO: Verify edges are detected at rectangle boundaries
    pass

if __name__ == "__main__":
    main()
