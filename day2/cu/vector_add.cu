/*
 * Day 2: Vector Addition
 * ======================
 * 
 * PROBLEM STATEMENT:
 * Implement parallel vector addition: C = A + B
 * Each thread adds one pair of elements.
 * 
 * REQUIREMENTS:
 * 1. Each thread should compute c[i] = a[i] + b[i]
 * 2. Handle arrays larger than grid size (optional: grid-stride loop)
 * 3. Proper bounds checking
 * 
 * CONSTRAINTS:
 * - Array size: up to 1,000,000 elements
 * - Use 256 threads per block
 * 
 * EXAMPLE:
 * a = [1, 2, 3, 4, 5]
 * b = [10, 20, 30, 40, 50]
 * c = [11, 22, 33, 44, 55]
 */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vector_add(float* a, float* b, float* c, int n) {
    // TODO: Calculate global thread index
    
    // TODO: Bounds check
    
    // TODO: Perform addition
}

// BONUS: Grid-stride loop version for very large arrays
__global__ void vector_add_grid_stride(float* a, float* b, float* c, int n) {
    // TODO: Calculate starting index
    
    // TODO: Calculate stride (total threads in grid)
    
    // TODO: Loop with stride to handle multiple elements per thread
}

int main() {
    const int N = 1000000;
    const int THREADS = 256;
    const int BLOCKS = (N + THREADS - 1) / THREADS;
    
    // TODO: Allocate and initialize host arrays
    
    // TODO: Allocate device arrays
    
    // TODO: Copy data to device
    
    // TODO: Launch kernel
    
    // TODO: Copy result back
    
    // TODO: Verify result
    
    // TODO: Cleanup
    
    return 0;
}
