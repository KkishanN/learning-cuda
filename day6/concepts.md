# Day 6: Parallel Reduction & Atomics

## 🎯 Learning Objectives
- Understand parallel reduction algorithms
- Implement tree-based reduction
- Use atomic operations safely
- Build histogram computation

---

## 🔄 What is Reduction?

Reduction: Combine many values into one using an associative operator.

```
Examples:
- Sum:     [1, 2, 3, 4, 5] → 15
- Max:     [3, 1, 4, 1, 5] → 5
- Product: [1, 2, 3, 4]    → 24
```

### Why It's Tricky on GPU
- Can't just sum in a loop (that's sequential!)
- Need parallel algorithm with log(N) steps

---

## 🌳 Tree-Based Reduction

### Algorithm
```
Step 0: [1] [2] [3] [4] [5] [6] [7] [8]
            ↓       ↓       ↓       ↓
Step 1: [ 3 ]   [ 7 ]   [11 ]   [15 ]
              ↓               ↓
Step 2: [   10    ]   [   26    ]
                    ↓
Step 3: [         36          ]
```

### Parallel Complexity
| Approach | Time Complexity |
|----------|-----------------|
| Sequential | O(N) |
| Tree Parallel | O(log N) |

For 1M elements: Sequential = 1M steps, Tree = 20 steps!

---

## 🔧 Reduction Kernel Pattern

```python
@cuda.jit
def reduce_sum(data, partial_sums):
    shared = cuda.shared.array(256, dtype=float32)
    tid = cuda.threadIdx.x
    gid = cuda.blockIdx.x * cuda.blockDim.x + tid
    
    # Load into shared memory
    shared[tid] = data[gid] if gid < data.size else 0
    cuda.syncthreads()
    
    # Tree reduction in shared memory
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if tid < stride:
            shared[tid] += shared[tid + stride]
        cuda.syncthreads()
        stride //= 2
    
    # Thread 0 writes block result
    if tid == 0:
        partial_sums[cuda.blockIdx.x] = shared[0]
```

---

## ⚛️ Atomic Operations

### The Race Condition Problem
```python
# BAD: Multiple threads updating same location
count[category] += 1  # Data race!
```

### Solution: Atomics
```python
# GOOD: Atomic operations are thread-safe
cuda.atomic.add(count, category, 1)
```

### Available Atomic Operations
| Operation | Function | Use Case |
|-----------|----------|----------|
| Add | `cuda.atomic.add(arr, idx, val)` | Histogram, counters |
| Max | `cuda.atomic.max(arr, idx, val)` | Finding maximum |
| Min | `cuda.atomic.min(arr, idx, val)` | Finding minimum |
| CAS | `cuda.atomic.compare_and_swap` | Custom atomics |

---

## 📊 Histogram Pattern

```python
@cuda.jit
def histogram(data, bins, num_bins):
    idx = cuda.grid(1)
    if idx < data.size:
        value = data[idx]
        bin_idx = int(value * num_bins)  # Map to bin
        bin_idx = min(bin_idx, num_bins - 1)
        cuda.atomic.add(bins, bin_idx, 1)
```

### Optimization: Privatization
```
Instead of:  All threads → Global histogram (lots of conflicts)
Better:      Threads → Local histogram → Merge to global
```

---

## ⚠️ Atomic Performance

Atomics are **slow** when many threads contend:

| Scenario | Performance |
|----------|-------------|
| No contention | Fast |
| Low contention | Good |
| High contention | Very slow (serializes) |

### Tips
1. Use shared memory privatization
2. Reduce number of atomic operations
3. Use warp-level primitives when possible

---

## ✅ Day 6 Summary

| Concept | Key Point |
|---------|-----------|
| Reduction | Combine N values → 1 value |
| Tree Algorithm | O(log N) parallel steps |
| Atomics | Thread-safe memory updates |
| Histogram | Count occurrences with atomic.add |
| Privatization | Local copies reduce contention |
