/*
 * Day 7: Data Pipeline
 * =====================
 * 
 * PROBLEM STATEMENT:
 * Build a 3-stage pipeline using streams:
 * 1. Load: Copy data to GPU
 * 2. Process: Transform data
 * 3. Store: Copy result back
 * 
 * PIPELINE PATTERN:
 * Chunk 0:   [Load] [Process] [Store]
 * Chunk 1:          [Load]    [Process] [Store]
 * Chunk 2:                    [Load]    [Process] [Store]
 * 
 * Each stage uses different hardware:
 * - Load/Store: DMA engines
 * - Process: CUDA cores
 * 
 * EVENTS API (for timing):
 * cudaEvent_t start, end;
 * cudaEventCreate(&start);
 * cudaEventRecord(start, stream);
 * ... kernel ...
 * cudaEventRecord(end, stream);
 * cudaEventSynchronize(end);
 * cudaEventElapsedTime(&time_ms, start, end);
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_CHUNKS 8
#define CHUNK_SIZE 500000

// Pipeline stage: normalize data
__global__ void normalize(float* data, float* output, float mean, float std, int n) {
    // TODO: output[i] = (data[i] - mean) / std
}

// Pipeline stage: apply non-linearity
__global__ void transform(float* data, float* output, int n) {
    // TODO: output[i] = tanh(data[i]) or similar
}

int main() {
    // TODO: Create streams for pipeline stages
    
    // TODO: Create events for timing
    
    // TODO: Allocate pinned host memory
    
    // TODO: Allocate device memory (double buffer for overlapping)
    
    // TODO: Implement pipeline:
    // for each chunk:
    //   - async copy to device
    //   - launch normalize kernel
    //   - launch transform kernel  
    //   - async copy back to host
    //   (all operations on same stream for this chunk)
    
    // TODO: Synchronize and measure time
    
    // TODO: Cleanup
    
    return 0;
}
