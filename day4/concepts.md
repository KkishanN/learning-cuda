# Day 4: GPU Image Processing

## 🎯 Learning Objectives
- Understand 2D kernel launches for image data
- Implement convolution operations on GPU
- Apply Gaussian blur and edge detection
- Visualize before/after results

---

## 🖼️ Images as 2D Arrays

Images are naturally parallel:
- Each pixel can be processed independently
- 2D grid of threads maps directly to 2D image

```
Image (Height × Width):
┌─────┬─────┬─────┬─────┬─────┐
│ P00 │ P01 │ P02 │ P03 │ P04 │
├─────┼─────┼─────┼─────┼─────┤
│ P10 │ P11 │ P12 │ P13 │ P14 │
├─────┼─────┼─────┼─────┼─────┤
│ P20 │ P21 │ P22 │ P23 │ P24 │
└─────┴─────┴─────┴─────┴─────┘

Thread Assignment:
Thread(x,y) → processes Pixel(y,x)
```

---

## 🔲 Convolution Operation

Convolution applies a **kernel** (small matrix) to each pixel:

```
  Kernel (3×3):          Image Region:
  ┌─────┬─────┬─────┐    ┌─────┬─────┬─────┐
  │ K00 │ K01 │ K02 │    │ P00 │ P01 │ P02 │
  ├─────┼─────┼─────┤  × ├─────┼─────┼─────┤
  │ K10 │ K11 │ K12 │    │ P10 │ P11 │ P12 │
  ├─────┼─────┼─────┤    ├─────┼─────┼─────┤
  │ K20 │ K21 │ K22 │    │ P20 │ P21 │ P22 │
  └─────┴─────┴─────┘    └─────┴─────┴─────┘
  
  Output = Σ(Kij × Pij)
```

---

## 🌫️ Gaussian Blur

Smooths image by averaging with weighted neighbors:

```python
# 3×3 Gaussian kernel (normalized)
kernel = [
    [1/16, 2/16, 1/16],
    [2/16, 4/16, 2/16],
    [1/16, 2/16, 1/16]
]
```

Effect: Each pixel becomes weighted average of neighbors.

---

## 🔳 Sobel Edge Detection

Detects edges using gradient approximation:

```python
# Horizontal edges (Gx)
sobel_x = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]

# Vertical edges (Gy)
sobel_y = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
]

# Gradient magnitude
edge = sqrt(Gx² + Gy²)
```

---

## 🧵 2D Kernel Launch

```python
# Image dimensions
height, width = image.shape

# Block size (16×16 = 256 threads typical)
block = (16, 16)

# Grid size (enough blocks to cover image)
grid_x = (width + block[0] - 1) // block[0]
grid_y = (height + block[1] - 1) // block[1]
grid = (grid_x, grid_y)

# Launch
kernel[grid, block](image, output)
```

---

## ⚠️ Boundary Handling

Pixels at edges don't have all neighbors:

```python
# Options:
# 1. Skip boundary pixels
if x > 0 and x < width-1 and y > 0 and y < height-1:
    # Process

# 2. Clamp to edge
neighbor_x = min(max(x + dx, 0), width - 1)

# 3. Reflect/mirror
# 4. Wrap around
```

---

## ✅ Day 4 Summary

| Concept | Key Point |
|---------|-----------|
| 2D Launch | `kernel[grid, block]` with 2D tuples |
| Thread Mapping | `x = blockIdx.x * blockDim.x + threadIdx.x` |
| Convolution | Weighted sum of neighbors |
| Gaussian Blur | Smoothing kernel (1-2-1 weights) |
| Sobel Edge | Gradient detection (Gx, Gy) |
