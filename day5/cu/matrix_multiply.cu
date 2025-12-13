/*
 * Day 5: Tiled Matrix Multiplication
 * ===================================
 * 
 * PROBLEM STATEMENT:
 * Implement C = A × B using tiled approach with shared memory.
 * 
 * NAIVE APPROACH PROBLEM:
 * - Each thread reads entire row of A and column of B
 * - For NxN matrices: 2N global memory reads per thread
 * - Total: 2N³ reads for N² output elements
 * 
 * TILED APPROACH:
 * 1. Divide matrices into TILE_SIZE × TILE_SIZE tiles
 * 2. For each tile pair:
 *    a. Load A tile into shared memory
 *    b. Load B tile into shared memory
 *    c. syncthreads()
 *    d. Compute partial product from shared memory
 *    e. syncthreads()
 * 3. Accumulate partial products
 * 
 * MEMORY REDUCTION:
 * - Reads reduced by factor of TILE_SIZE
 * - TILE_SIZE=16: 16× fewer global reads!
 * 
 * TILE_SIZE: 16 (256 threads per block, fits in shared memory)
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_naive(float* A, float* B, float* C, int M, int K, int N) {
    // TODO: Calculate row and col for this thread
    
    // TODO: Bounds check
    
    // TODO: Loop k from 0 to K
    // TODO: Accumulate: result += A[row*K + k] * B[k*N + col]
    
    // TODO: Write to C[row*N + col]
}

__global__ void matmul_tiled(float* A, float* B, float* C, int M, int K, int N) {
    // TODO: Declare shared memory tiles
    // __shared__ float sA[TILE_SIZE][TILE_SIZE];
    // __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    // TODO: Get thread position (tx, ty) and global position (row, col)
    
    // TODO: Initialize accumulator
    
    // TODO: Loop over tiles
    // for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++)
    
        // TODO: Load tile of A into sA (with bounds check)
        
        // TODO: Load tile of B into sB (with bounds check)
        
        // TODO: syncthreads()
        
        // TODO: Compute partial product from shared memory
        // for (int k = 0; k < TILE_SIZE; k++)
        //     result += sA[ty][k] * sB[k][tx];
        
        // TODO: syncthreads()
    
    // TODO: Write final result (with bounds check)
}

int main() {
    const int M = 256, K = 256, N = 256;
    
    // TODO: Allocate host matrices A, B, C
    
    // TODO: Initialize A and B with random values
    
    // TODO: Allocate device matrices
    
    // TODO: Copy A and B to device
    
    // TODO: Configure kernel launch
    // dim3 block(TILE_SIZE, TILE_SIZE);
    // dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    // TODO: Launch both kernels, compare results
    
    // TODO: Verify against CPU matrix multiply
    
    return 0;
}
