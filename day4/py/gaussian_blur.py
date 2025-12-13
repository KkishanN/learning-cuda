"""
Day 4: Gaussian Blur
====================

PROBLEM STATEMENT:
Apply 3x3 Gaussian blur to a grayscale image.

GAUSSIAN KERNEL:
    [1/16, 2/16, 1/16]
    [2/16, 4/16, 2/16]  
    [1/16, 2/16, 1/16]

ALGORITHM:
1. For each pixel, read 3x3 neighborhood
2. Multiply by kernel weights
3. Sum and write to output

2D KERNEL LAUNCH:
block = (16, 16)
grid = ((width + 15) // 16, (height + 15) // 16)

Run with: $env:NUMBA_ENABLE_CUDASIM = "1"; python gaussian_blur.py
"""

import os
os.environ['NUMBA_ENABLE_CUDASIM'] = '1'

import numpy as np
from numba import cuda
import numba

GAUSSIAN_KERNEL = np.array([
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]
], dtype=np.float32)

@cuda.jit
def gaussian_blur(input_img, output_img, kernel):
    # TODO: Get x, y using 2D indexing
    # x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    # TODO: Bounds check - skip boundaries
    
    # TODO: Loop ky from -1 to 1, kx from -1 to 1
    
    # TODO: Accumulate: result += input[y+ky, x+kx] * kernel[ky+1, kx+1]
    
    # TODO: Write result to output[y, x]
    pass

def main():
    # Create test image
    size = 64
    
    # TODO: Create sample image (random noise or pattern)
    
    # TODO: Copy to device
    
    # TODO: Configure 2D kernel launch
    
    # TODO: Launch and get result
    
    # TODO: Display or verify result
    pass

if __name__ == "__main__":
    main()
