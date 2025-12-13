/*
 * Day 4: Sobel Edge Detection
 * ============================
 * 
 * PROBLEM STATEMENT:
 * Detect edges in a grayscale image using Sobel operator.
 * 
 * SOBEL KERNELS:
 * 
 * Gx (horizontal edges):     Gy (vertical edges):
 *   -1   0   1                -1  -2  -1
 *   -2   0   2                 0   0   0
 *   -1   0   1                 1   2   1
 * 
 * ALGORITHM:
 * 1. Apply Gx kernel to get horizontal gradient
 * 2. Apply Gy kernel to get vertical gradient
 * 3. Magnitude = sqrt(Gx^2 + Gy^2)
 * 
 * EXPECTED OUTPUT:
 * - High values at edges
 * - Low values in uniform regions
 */

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__constant__ int sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

__global__ void sobel_edge_detection(float* input, float* output, int width, int height) {
    // TODO: Calculate pixel coordinates
    
    // TODO: Skip boundary pixels
    
    // TODO: Apply Sobel X kernel - accumulate Gx
    
    // TODO: Apply Sobel Y kernel - accumulate Gy
    
    // TODO: Calculate magnitude: sqrt(gx*gx + gy*gy)
    
    // TODO: Normalize if needed (divide by max possible value ~4*sqrt(2))
    
    // TODO: Write to output
}

int main() {
    const int WIDTH = 256;
    const int HEIGHT = 256;
    
    // TODO: Create test image with clear edges
    // (e.g., white rectangle on black background)
    
    // TODO: Allocate and copy to device
    
    // TODO: Launch kernel
    
    // TODO: Get result and verify edges detected
    
    return 0;
}
