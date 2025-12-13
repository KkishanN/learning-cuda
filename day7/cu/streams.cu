/*
 * Day 7: CUDA Streams
 * ====================
 * 
 * PROBLEM STATEMENT:
 * Use CUDA streams to overlap kernel execution and memory transfers.
 * 
 * CONCEPTS:
 * - Stream: Sequence of operations that execute in order
 * - Different streams can execute concurrently
 * - Default stream (0) is synchronous
 * 
 * API:
 * cudaStream_t stream;
 * cudaStreamCreate(&stream);
 * kernel<<<grid, block, 0, stream>>>(...);
 * cudaMemcpyAsync(..., stream);
 * cudaStreamSynchronize(stream);
 * cudaStreamDestroy(stream);
 * 
 * TASK:
 * Process 4 data chunks using 4 streams to overlap execution.
 * Compare time vs sequential processing.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 4
#define CHUNK_SIZE 1000000

__global__ void process_chunk(float* input, float* output, int n) {
    // TODO: Simple processing - e.g., square each element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        output[idx] = val * val;
    }
}

void sequential_processing(float** h_input, float** h_output, 
                           float** d_input, float** d_output, int chunks) {
    // TODO: Process each chunk sequentially
    // for each chunk:
    //   cudaMemcpy to device
    //   launch kernel
    //   cudaDeviceSynchronize()
    //   cudaMemcpy to host
}

void stream_processing(float** h_input, float** h_output,
                       float** d_input, float** d_output, 
                       cudaStream_t* streams, int chunks) {
    // TODO: Process chunks concurrently using streams
    // for each chunk i:
    //   cudaMemcpyAsync(d_input[i], h_input[i], ..., streams[i])
    //   kernel<<<..., streams[i]>>>(d_input[i], d_output[i], ...)
    //   cudaMemcpyAsync(h_output[i], d_output[i], ..., streams[i])
    
    // TODO: Wait for all streams
    // for each stream: cudaStreamSynchronize(stream)
}

int main() {
    // TODO: Create NUM_STREAMS streams
    // cudaStream_t streams[NUM_STREAMS];
    // for (...) cudaStreamCreate(&streams[i]);
    
    // TODO: Allocate pinned host memory for async transfers
    // cudaMallocHost(...) instead of malloc()
    
    // TODO: Allocate device memory
    
    // TODO: Initialize data
    
    // TODO: Time sequential processing
    
    // TODO: Time stream processing
    
    // TODO: Compare times
    
    // TODO: Cleanup: destroy streams, free memory
    
    return 0;
}
