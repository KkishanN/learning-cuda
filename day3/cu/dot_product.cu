/*
 * Day 3: Dot Product with Shared Memory Reduction
 * ================================================
 * 
 * PROBLEM STATEMENT:
 * Compute dot product of two vectors using shared memory for block-level reduction.
 * 
 * ALGORITHM:
 * 1. Each thread computes a[i] * b[i] for its element
 * 2. Store products in shared memory
 * 3. Perform tree-based reduction within the block
 * 4. Thread 0 writes block's partial sum to global memory
 * 5. Sum partial results on CPU (or with another kernel)
 * 
 * KEY CONCEPTS:
 * - Shared memory for fast inter-thread communication
 * - Tree reduction: O(log N) parallel steps
 * - syncthreads() between reduction steps
 * 
 * EXPECTED: For a = [1,2,3,...,N] and b = [1,1,1,...,1]
 *           Result = 1 + 2 + 3 + ... + N = N*(N+1)/2
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void dot_product(float* a, float* b, float* partial_sums, int n) {
    // TODO: Declare shared memory array of size BLOCK_SIZE
    
    // TODO: Get thread and global indices
    
    // TODO: Each thread computes its product (with bounds check)
    
    // TODO: Store in shared memory
    
    // TODO: syncthreads()
    
    // TODO: Tree reduction in shared memory
    // for (stride = blockDim.x / 2; stride > 0; stride /= 2)
    //     if (tid < stride) shared[tid] += shared[tid + stride]
    //     syncthreads()
    
    // TODO: Thread 0 writes block result to partial_sums
}

int main() {
    const int N = 100000;
    
    // TODO: Allocate and initialize vectors a and b
    
    // TODO: Calculate expected result: N*(N+1)/2
    
    // TODO: Allocate device memory
    
    // TODO: Launch kernel
    
    // TODO: Sum partial results on CPU
    
    // TODO: Compare with expected result
    
    // TODO: Cleanup
    
    return 0;
}
