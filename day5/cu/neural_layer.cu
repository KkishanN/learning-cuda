/*
 * Day 5: Simple Neural Network Layer
 * ===================================
 * 
 * PROBLEM STATEMENT:
 * Implement forward pass: Y = ReLU(X @ W + b)
 * 
 * DIMENSIONS:
 * - X: (batch_size, input_features)
 * - W: (input_features, output_features)
 * - b: (output_features,)
 * - Y: (batch_size, output_features)
 * 
 * OPERATIONS:
 * 1. Matrix multiplication: X @ W
 * 2. Bias addition: + b (broadcast to each row)
 * 3. ReLU activation: max(0, x)
 * 
 * OPTIMIZATION:
 * Fuse all three operations into single kernel to avoid
 * intermediate memory writes.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Simple ReLU kernel (for learning)
__global__ void relu(float* input, float* output, int size) {
    // TODO: Calculate index
    
    // TODO: output[i] = max(0, input[i])
}

// Fused: matmul + bias + relu
__global__ void dense_layer_fused(float* X, float* W, float* bias, float* Y,
                                   int batch, int in_features, int out_features) {
    // TODO: Use tiled matmul from matrix_multiply.cu
    
    // TODO: After computing Y[row][col] = dot product of X row and W column
    
    // TODO: Add bias: result += bias[col]
    
    // TODO: Apply ReLU: result = max(0, result)
    
    // TODO: Write to Y[row * out_features + col]
}

int main() {
    const int BATCH = 32;
    const int IN_FEATURES = 784;   // e.g., 28x28 image
    const int OUT_FEATURES = 128;
    
    // TODO: Allocate X, W, bias, Y
    
    // TODO: Initialize with random values
    // W should be initialized with small random values (He initialization)
    
    // TODO: Launch fused kernel
    
    // TODO: Verify some output values are zero (ReLU effect)
    
    return 0;
}
