# Day 5: Matrix Operations & Neural Networks

## 🎯 Learning Objectives
- Implement efficient GPU matrix multiplication
- Understand tiling for matrix operations
- Build simple neural network layers
- See how deep learning uses GPU parallelism

---

## 🔢 Matrix Multiplication Basics

### The Operation
```
C[i,j] = Σ A[i,k] × B[k,j]  (sum over k)
```

### Naive Approach
```
For 1024×1024 matrices:
- ~1 billion multiply-add operations
- Each output element = 1024 operations
- Perfect for parallelism: each C[i,j] independent!
```

---

## 🧱 Tiled Matrix Multiplication

### The Problem with Naive Approach
Each thread reads an entire row of A and column of B.
For 1024×1024: each thread reads 2048 values from global memory!

### The Solution: Tiling
Load submatrices (tiles) into shared memory:

```
        B (load columns)
        ┌───┬───┬───┐
        │ B │ B │ B │
        │ 0 │ 1 │ 2 │
    ┌───┼───┼───┼───┤
  A │ A │ C │ C │ C │
    │ 0 │ 00│ 01│ 02│
load├───┼───┼───┼───┤
rows│ A │ C │ C │ C │
    │ 1 │ 10│ 11│ 12│
    └───┴───┴───┴───┘
```

### Tiled Algorithm
```python
for each tile pair (A_tile, B_tile):
    1. Load A_tile into shared memory
    2. Load B_tile into shared memory
    3. syncthreads()
    4. Compute partial products
    5. syncthreads()
    6. Accumulate to result
```

### Benefits
| Approach | Global Memory Reads | Speedup |
|----------|--------------------:|--------:|
| Naive | N × 2N = 2N² | 1× |
| Tiled (T=16) | N × 2N/T = N²/8 | ~16× |
| Tiled (T=32) | N × 2N/T = N²/16 | ~32× |

---

## 🧠 Neural Network on GPU

### Why GPUs for Deep Learning?
```
Neural Network = Matrix Operations!

Forward pass:  Y = ReLU(W × X + b)
- W × X: Matrix multiplication
- + b:   Vector addition (broadcasting)
- ReLU:  Element-wise max(0, x)

All massively parallel operations!
```

### Single Layer
```
Input: X (batch × features)
Weight: W (features × outputs)  
Bias: b (outputs)
Output: Y (batch × outputs)

Y = activation(X @ W + b)
```

---

## 🚀 ReLU Activation

```python
@cuda.jit
def relu_kernel(input, output):
    idx = cuda.grid(1)
    if idx < input.size:
        output.flat[idx] = max(0.0, input.flat[idx])
```

ReLU is embarrassingly parallel:
- Each element processed independently
- No memory dependencies
- Perfect GPU utilization

---

## 📊 Batch Processing

Process multiple samples simultaneously:

```
Batch of 256 images, 784 features → 128 outputs

X: (256, 784)   # 256 images, 784 pixels each
W: (784, 128)   # Weight matrix
b: (128,)       # Bias vector
Y: (256, 128)   # 256 outputs, 128 features each

One kernel launch processes all 256 images!
```

---

## ✅ Day 5 Summary

| Concept | Key Point |
|---------|-----------|
| Matrix Multiply | C[i,j] = Σ A[i,k] × B[k,j] |
| Tiled Approach | Load tiles to shared memory |
| Benefit | Reduces global memory reads by tile size |
| Neural Layer | Y = activation(X @ W + b) |
| Batch Processing | Process many samples in one kernel |
