/*
 * Day 6: Parallel Reduction (Sum)
 * ================================
 * 
 * PROBLEM STATEMENT:
 * Compute sum of an array using parallel tree reduction.
 * 
 * SEQUENTIAL: O(N) steps
 * PARALLEL:   O(log N) steps
 * 
 * ALGORITHM:
 * 1. Load elements into shared memory
 * 2. Tree reduction:
 *    stride = blockDim.x / 2
 *    while stride > 0:
 *        if tid < stride:
 *            shared[tid] += shared[tid + stride]
 *        syncthreads()
 *        stride /= 2
 * 3. Thread 0 writes block result to partial_sums
 * 4. Sum partial results on CPU
 * 
 * EXAMPLE (8 elements):
 * Step 0: [1, 2, 3, 4, 5, 6, 7, 8]
 * Step 1: [6, 8, 10, 12, -, -, -, -]  (stride=4)
 * Step 2: [16, 20, -, -, -, -, -, -]  (stride=2)
 * Step 3: [36, -, -, -, -, -, -, -]   (stride=1)
 * Result: 36
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void reduce_sum(float* input, float* partial_sums, int n) {
    // TODO: Declare shared memory
    // __shared__ float shared[BLOCK_SIZE];
    
    // TODO: Get thread id and global id
    
    // TODO: Load into shared memory (with bounds check, else 0)
    
    // TODO: syncthreads()
    
    // TODO: Tree reduction loop
    // for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    //     if (tid < stride) shared[tid] += shared[tid + stride]
    //     syncthreads()
    
    // TODO: Thread 0 writes to partial_sums[blockIdx.x]
}

// OPTIMIZATION: First add during load
__global__ void reduce_sum_optimized(float* input, float* partial_sums, int n) {
    // TODO: Each thread loads and adds TWO elements
    // This halves the number of threads needed in first level
    
    // TODO: Rest same as above
}

int main() {
    const int N = 1000000;
    
    // TODO: Allocate and initialize array with known sum
    // e.g., all 1s -> sum = N
    
    // TODO: Allocate partial sums array
    
    // TODO: Launch kernel
    
    // TODO: Sum partial results on CPU
    
    // TODO: Verify: sum should equal N
    
    return 0;
}
