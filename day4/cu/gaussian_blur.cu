/*
 * Day 4: Gaussian Blur (Image Convolution)
 * =========================================
 * 
 * PROBLEM STATEMENT:
 * Apply 3x3 Gaussian blur to a grayscale image.
 * 
 * GAUSSIAN KERNEL (normalized):
 *     1/16  2/16  1/16
 *     2/16  4/16  2/16
 *     1/16  2/16  1/16
 * 
 * ALGORITHM:
 * For each pixel (x, y):
 *   1. Read 3x3 neighborhood from input
 *   2. Multiply each neighbor by corresponding kernel weight
 *   3. Sum all products
 *   4. Write result to output[y][x]
 * 
 * BOUNDARY HANDLING:
 * - Skip boundary pixels (simplest approach)
 * - Or clamp coordinates to valid range
 * 
 * 2D INDEXING:
 * x = blockIdx.x * blockDim.x + threadIdx.x
 * y = blockIdx.y * blockDim.y + threadIdx.y
 * 
 * BONUS: Use shared memory to cache tile + halo
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Gaussian kernel weights (can be in constant memory for better performance)
__constant__ float gaussian_kernel[9] = {
    1.0f/16, 2.0f/16, 1.0f/16,
    2.0f/16, 4.0f/16, 2.0f/16,
    1.0f/16, 2.0f/16, 1.0f/16
};

__global__ void gaussian_blur(float* input, float* output, int width, int height) {
    // TODO: Calculate pixel coordinates (x, y)
    
    // TODO: Bounds check - skip if at boundary
    
    // TODO: Loop over 3x3 neighborhood (ky = -1, 0, 1; kx = -1, 0, 1)
    
    // TODO: Accumulate weighted sum
    
    // TODO: Write result to output
}

__global__ void gaussian_blur_shared(float* input, float* output, int width, int height) {
    // BONUS: Implement with shared memory
    // 1. Declare shared tile with halo: [TILE_SIZE + 2][TILE_SIZE + 2]
    // 2. Each thread loads its pixel + possibly halo pixels
    // 3. syncthreads()
    // 4. Compute blur from shared memory
    // 5. Write to output
}

int main() {
    const int WIDTH = 256;
    const int HEIGHT = 256;
    
    // TODO: Create sample image (e.g., checkerboard or gradient)
    
    // TODO: Allocate device memory
    
    // TODO: Copy input to device
    
    // TODO: Launch kernel with 2D grid
    // dim3 block(16, 16)
    // dim3 grid((WIDTH+15)/16, (HEIGHT+15)/16)
    
    // TODO: Copy result back
    
    // TODO: (Optional) Save as image file
    
    return 0;
}
