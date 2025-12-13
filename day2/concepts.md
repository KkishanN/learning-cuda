# Day 2: GPU Architecture & Threading Fundamentals

## 🎯 Learning Objectives
By the end of this day, you'll understand:
- How GPUs differ from CPUs architecturally
- The Thread → Block → Grid hierarchy
- How to calculate thread indices
- The SIMT (Single Instruction, Multiple Threads) execution model

---

## 🖥️ GPU vs CPU Architecture

### CPU (Central Processing Unit)
```
┌─────────────────────────────────────────┐
│  CPU: Optimized for SEQUENTIAL tasks    │
├─────────────────────────────────────────┤
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │Core1│ │Core2│ │Core3│ │Core4│       │
│  │ 💪  │ │ 💪  │ │ 💪  │ │ 💪  │       │
│  └─────┘ └─────┘ └─────┘ └─────┘       │
│  Few powerful cores (4-64)              │
│  Large caches, complex control logic    │
│  Great for: branching, serial code      │
└─────────────────────────────────────────┘
```

### GPU (Graphics Processing Unit)
```
┌─────────────────────────────────────────────────────────────┐
│  GPU: Optimized for PARALLEL tasks                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐  │
│  │🔸││🔸││🔸││🔸││🔸││🔸││🔸││🔸││🔸││🔸││🔸││🔸││🔸││🔸│  │
│  └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘  │
│  ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐  │
│  │🔸││🔸││🔸││🔸││🔸││🔸││🔸││🔸││🔸││🔸││🔸││🔸││🔸││🔸│  │
│  └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘  │
│  Thousands of small cores (1000-16000+)                     │
│  Small caches, simple control logic                         │
│  Great for: data parallelism, matrix ops                    │
└─────────────────────────────────────────────────────────────┘
```

### Key Insight
> **CPUs**: Few workers, each very smart (complex tasks)
> **GPUs**: Many workers, each simple (same task, different data)

---

## 🧵 Thread Hierarchy: Thread → Block → Grid

This is the **most important concept** in CUDA programming!

### Visual Representation
```
                          GRID (All Blocks)
    ┌───────────────────────────────────────────────────────┐
    │  Block(0,0)     Block(1,0)     Block(2,0)             │
    │  ┌─────────┐    ┌─────────┐    ┌─────────┐            │
    │  │ T T T T │    │ T T T T │    │ T T T T │            │
    │  │ T T T T │    │ T T T T │    │ T T T T │            │
    │  └─────────┘    └─────────┘    └─────────┘            │
    │                                                        │
    │  Block(0,1)     Block(1,1)     Block(2,1)             │
    │  ┌─────────┐    ┌─────────┐    ┌─────────┐            │
    │  │ T T T T │    │ T T T T │    │ T T T T │            │
    │  │ T T T T │    │ T T T T │    │ T T T T │            │
    │  └─────────┘    └─────────┘    └─────────┘            │
    └───────────────────────────────────────────────────────┘
    
    T = Thread
    Each Block has same number of threads
    Grid can have many blocks
```

### Hierarchy Explained

| Level | Description | Size |
|-------|-------------|------|
| **Thread** | Smallest unit, executes the kernel code | 1 |
| **Block** | Group of threads that can cooperate | Up to 1024 threads |
| **Grid** | Collection of all blocks for a kernel | Millions of blocks |

### Why This Hierarchy?

1. **Threads in a block can**:
   - Share memory (shared memory)
   - Synchronize with `syncthreads()`
   - Cooperate on computation

2. **Blocks are independent**:
   - Can execute in any order
   - Cannot communicate directly
   - Enables scaling across GPUs

---

## 🔢 Thread Indexing

### Built-in Variables

```python
# In every CUDA kernel, you have access to:
cuda.threadIdx.x   # Thread index within block (0 to blockDim.x-1)
cuda.threadIdx.y   # For 2D blocks
cuda.threadIdx.z   # For 3D blocks

cuda.blockIdx.x    # Block index within grid (0 to gridDim.x-1)
cuda.blockIdx.y    # For 2D grids
cuda.blockIdx.z    # For 3D grids

cuda.blockDim.x    # Number of threads per block in x dimension
cuda.gridDim.x     # Number of blocks in grid in x dimension
```

### Global Thread ID Formula (1D)

```
global_id = blockIdx.x * blockDim.x + threadIdx.x
```

**Example:**
```
Grid: 3 blocks, each with 4 threads

Block 0: threads 0,1,2,3   → global IDs: 0,1,2,3
Block 1: threads 0,1,2,3   → global IDs: 4,5,6,7
Block 2: threads 0,1,2,3   → global IDs: 8,9,10,11

For Block 1, Thread 2:
global_id = 1 * 4 + 2 = 6 ✓
```

### Global Thread ID Formula (2D)

```python
# For 2D data (like images):
x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

# Linear index in row-major order:
linear_idx = y * width + x
```

---

## 🔄 SIMT Execution Model

**SIMT = Single Instruction, Multiple Threads**

### How It Works
```
Time →
─────────────────────────────────────────────────────────
Cycle 1:  All 32 threads execute: a = x + y
Cycle 2:  All 32 threads execute: b = a * 2
Cycle 3:  All 32 threads execute: result[i] = b
─────────────────────────────────────────────────────────
          ↑ Same instruction, different data per thread
```

### Warps
- GPU schedules threads in groups of **32 threads** called a **warp**
- All threads in a warp execute the **same instruction** simultaneously
- If threads diverge (different if/else branches), performance suffers

### Warp Divergence Problem
```python
# BAD: Causes warp divergence
if threadIdx.x % 2 == 0:
    do_something()      # Half the warp waits
else:
    do_something_else() # Other half waits

# Execution becomes serialized!
```

---

## 🧪 Practice Exercises

1. **Run `thread_visualization.py`** - See how threads are organized
2. **Run `parallel_patterns.py`** - Observe thread ID assignments
3. **Calculate**: If you have 256 threads per block and 1000 elements:
   - How many blocks do you need?
   - Answer: `ceil(1000/256) = 4 blocks`

---

## 📊 Key Formulas Cheat Sheet

```python
# Number of blocks needed
num_blocks = (n + threads_per_block - 1) // threads_per_block

# Global 1D index
idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

# Global 2D index
x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

# Bounds check (always do this!)
if idx < array_size:
    # Safe to access array[idx]
```

---

## ✅ Day 2 Summary

| Concept | Remember |
|---------|----------|
| GPU vs CPU | Many simple cores vs few complex cores |
| Thread | Smallest execution unit |
| Block | Group of cooperating threads (max 1024) |
| Grid | All blocks for a kernel launch |
| SIMT | All threads in warp execute same instruction |
| Warp | 32 threads, scheduled together |
