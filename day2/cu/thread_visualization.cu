/*
 * Day 2: Thread Visualization
 * ===========================
 * 
 * PROBLEM STATEMENT:
 * Write a CUDA kernel that records each thread's indices into output arrays.
 * This helps visualize how threads are organized in blocks and grids.
 * 
 * REQUIREMENTS:
 * 1. Each thread should write its blockIdx.x to output_block_idx array
 * 2. Each thread should write its threadIdx.x to output_thread_idx array  
 * 3. Each thread should calculate and write its global ID to output_global_idx
 * 4. Handle bounds checking to avoid out-of-bounds access
 * 
 * FORMULA:
 * global_id = blockIdx.x * blockDim.x + threadIdx.x
 * 
 * EXPECTED OUTPUT (for 3 blocks × 4 threads):
 * Global ID:  0  1  2  3  4  5  6  7  8  9  10  11
 * Block ID:   0  0  0  0  1  1  1  1  2  2   2   2
 * Thread ID:  0  1  2  3  0  1  2  3  0  1   2   3
 */

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void record_thread_info(int* output_block_idx, 
                                    int* output_thread_idx, 
                                    int* output_global_idx, 
                                    int n) {
    // TODO: Get block and thread indices
    
    // TODO: Calculate global thread ID
    
    // TODO: Bounds check and write to output arrays
}

int main() {
    const int THREADS_PER_BLOCK = 4;
    const int NUM_BLOCKS = 3;
    const int TOTAL_THREADS = THREADS_PER_BLOCK * NUM_BLOCKS;
    
    // TODO: Allocate host memory
    
    // TODO: Allocate device memory
    
    // TODO: Launch kernel
    
    // TODO: Copy results back
    
    // TODO: Print results in table format
    
    // TODO: Free memory
    
    return 0;
}
