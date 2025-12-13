/*
 * Day 3: Matrix Transpose (Memory Coalescing Demo)
 * =================================================
 * 
 * PROBLEM STATEMENT:
 * Implement matrix transpose and compare coalesced vs non-coalesced access.
 * 
 * PART 1 - NAIVE TRANSPOSE:
 * Write a kernel that transposes a matrix but has non-coalesced writes.
 * Read row-wise (coalesced), write column-wise (non-coalesced).
 * 
 * PART 2 - SHARED MEMORY TRANSPOSE:
 * Use shared memory to achieve coalesced reads AND writes.
 * 1. Load tile into shared memory (coalesced read)
 * 2. Synchronize threads
 * 3. Write from shared memory with transposed indices (coalesced write)
 * 
 * KEY CONCEPTS:
 * - Coalescing: Threads in a warp access consecutive memory addresses
 * - Shared memory: Fast on-chip memory shared by block
 * - syncthreads(): Barrier for thread synchronization
 * 
 * TILE_SIZE: 16x16 (256 threads per block)
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void transpose_naive(float* input, float* output, int width, int height) {
    // TODO: Calculate global x, y coordinates
    
    // TODO: Bounds check
    
    // TODO: Read from input[y * width + x]
    // TODO: Write to output[x * height + y]
    // NOTE: The write is non-coalesced!
}

__global__ void transpose_shared(float* input, float* output, int width, int height) {
    // TODO: Declare shared memory tile with padding: [TILE_SIZE][TILE_SIZE + 1]
    // The +1 padding avoids bank conflicts when reading columns
    
    // TODO: Calculate global and local coordinates
    
    // TODO: Load tile into shared memory (coalesced read)
    
    // TODO: syncthreads() - wait for all threads to finish loading
    
    // TODO: Calculate transposed output coordinates
    
    // TODO: Write from shared memory (coalesced write)
}

int main() {
    const int WIDTH = 1024;
    const int HEIGHT = 1024;
    
    // TODO: Allocate host matrices
    
    // TODO: Initialize input matrix
    
    // TODO: Allocate device matrices
    
    // TODO: Copy input to device
    
    // TODO: Launch naive transpose kernel
    
    // TODO: Launch shared memory transpose kernel
    
    // TODO: Compare results
    
    // TODO: Cleanup
    
    return 0;
}
