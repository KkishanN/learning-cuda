/*
 * Day 6: Histogram with Atomics
 * ==============================
 * 
 * PROBLEM STATEMENT:
 * Compute histogram of values using atomic operations.
 * 
 * THE PROBLEM:
 * Multiple threads may try to increment same bin -> race condition!
 * 
 * SOLUTION:
 * atomicAdd(&bins[bin_idx], 1) - thread-safe increment
 * 
 * ALGORITHM:
 * 1. Each thread reads one value
 * 2. Compute bin index: bin = (value - min) / (max - min) * num_bins
 * 3. Atomically increment: atomicAdd(&histogram[bin], 1)
 * 
 * OPTIMIZATION (Privatization):
 * 1. Each block has local histogram in shared memory
 * 2. Atomics to local (fewer conflicts)
 * 3. syncthreads()
 * 4. Merge local to global
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_BINS 256

__global__ void histogram_atomic(unsigned char* data, int* histogram, int n) {
    // TODO: Get global index
    
    // TODO: Bounds check
    
    // TODO: Get value at data[idx]
    
    // TODO: atomicAdd(&histogram[value], 1)
}

__global__ void histogram_privatized(unsigned char* data, int* histogram, int n) {
    // TODO: Declare shared memory histogram
    // __shared__ int local_hist[NUM_BINS];
    
    // TODO: Initialize local histogram to 0 (thread tid = 0..255 sets local_hist[tid] = 0)
    
    // TODO: syncthreads()
    
    // TODO: Each thread processes its elements, atomicAdd to local_hist
    
    // TODO: syncthreads()
    
    // TODO: Merge to global: atomicAdd(&histogram[tid], local_hist[tid])
}

int main() {
    const int N = 1000000;
    
    // TODO: Allocate and initialize random data (0-255)
    
    // TODO: Allocate histogram (256 bins, initialized to 0)
    
    // TODO: Launch kernel
    
    // TODO: Verify: sum of histogram = N
    
    return 0;
}
