# Day 3: Memory Management & Optimization

## 🎯 Learning Objectives
By the end of this day, you'll understand:
- GPU memory hierarchy and types
- Memory coalescing for optimal performance
- Shared memory usage and bank conflicts
- When to use each memory type

---

## 🏗️ GPU Memory Hierarchy

```
                    ┌─────────────────────────────────────┐
                    │         HOST (CPU) MEMORY           │
                    │  💾 System RAM (16-128 GB)          │
                    │  Slow access from GPU               │
                    └──────────────┬──────────────────────┘
                                   │ PCIe Bus (slow)
                    ┌──────────────▼──────────────────────┐
                    │        GLOBAL MEMORY                │
                    │  💾 GPU VRAM (8-80 GB)              │
                    │  Accessible by all threads          │
                    │  High latency: ~400-800 cycles      │
                    └──────────────┬──────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼───────┐          ┌───────▼───────┐          ┌───────▼───────┐
│ SHARED MEMORY │          │ L1/L2 CACHE   │          │ CONSTANT MEM  │
│ 🚀 Per Block  │          │ ⚡ Automatic   │          │ 📖 Read-only  │
│ 48-164 KB     │          │ ~1-2 MB       │          │ 64 KB         │
│ ~5 cycles     │          │ ~30-100 cycles│          │ Cached        │
└───────┬───────┘          └───────────────┘          └───────────────┘
        │
┌───────▼───────┐
│   REGISTERS   │
│ ⚡ Per Thread │
│ Fastest       │
│ ~1 cycle      │
└───────────────┘
```

---

## 📊 Memory Types Comparison

| Memory Type | Scope | Speed | Size | Use Case |
|------------|-------|-------|------|----------|
| **Registers** | Thread | Fastest (1 cycle) | ~64KB/SM | Local variables |
| **Shared** | Block | Very Fast (~5 cycles) | 48-164 KB | Thread cooperation |
| **L1/L2 Cache** | Automatic | Fast (~30-100 cycles) | 1-2 MB | Auto-cached globals |
| **Global** | All threads | Slow (~400-800 cycles) | 8-80 GB | Main data storage |
| **Constant** | All threads (read-only) | Fast (cached) | 64 KB | Constants, lookup tables |
| **Local** | Thread (spilled registers) | Slow | In global | Large arrays per thread |

---

## 🚀 Memory Coalescing

### What is Coalescing?
When threads in a warp access **consecutive memory addresses**, the GPU can combine these into a single memory transaction.

### Good (Coalesced) Access Pattern
```python
# All threads in warp access consecutive addresses
# Thread 0 → data[0], Thread 1 → data[1], Thread 2 → data[2], ...

@cuda.jit
def coalesced_access(data, output):
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    output[idx] = data[idx]  # ✅ Coalesced: one transaction for 32 threads
```

```
Memory: [0][1][2][3][4][5][6][7]...
         ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
         T0 T1 T2 T3 T4 T5 T6 T7  ← One memory transaction!
```

### Bad (Strided) Access Pattern
```python
# Threads access with stride > 1
# Thread 0 → data[0], Thread 1 → data[32], Thread 2 → data[64], ...

@cuda.jit  
def strided_access(data, output, stride):
    idx = cuda.threadIdx.x * stride  # ❌ Strided access
    output[idx] = data[idx]  # Multiple transactions needed!
```

```
Memory: [0][1][2]...[32][33]...[64][65]...
         ↑          ↑           ↑
         T0         T1          T2  ← 32 separate transactions!
```

### Performance Impact
| Pattern | Transactions | Relative Speed |
|---------|--------------|----------------|
| Coalesced | 1 | 100% (fastest) |
| Stride 2 | 2 | 50% |
| Stride 32 | 32 | 3% (32x slower!) |

---

## 💾 Shared Memory

### What is Shared Memory?
- Fast on-chip memory (SRAM)
- Shared by all threads in a **block**
- Programmer-managed cache
- Used for thread cooperation and data reuse

### Shared Memory Declaration
```python
@cuda.jit
def use_shared_memory(data, output):
    # Declare shared memory array
    shared = cuda.shared.array(shape=(256,), dtype=numba.float32)
    
    idx = cuda.threadIdx.x
    
    # Load data into shared memory
    shared[idx] = data[cuda.blockIdx.x * cuda.blockDim.x + idx]
    
    # CRITICAL: Synchronize before using shared data
    cuda.syncthreads()
    
    # Now all threads can access any element in shared[]
    # Example: access neighbor's data
    if idx > 0:
        output[idx] = shared[idx] + shared[idx - 1]
```

### When to Use Shared Memory
1. **Data reuse**: Multiple threads read same data
2. **Thread cooperation**: Threads share intermediate results  
3. **Coalescing fix**: Load strided data, rearrange in shared memory
4. **Reduction operations**: Partial sums before writing to global

---

## ⚠️ Bank Conflicts

### What are Banks?
Shared memory is divided into **32 banks** (one per thread in a warp).
Each bank can serve one request per cycle.

```
Shared Memory Banks:
Bank 0  | Bank 1  | Bank 2  | ... | Bank 31
addr 0  | addr 1  | addr 2  | ... | addr 31
addr 32 | addr 33 | addr 34 | ... | addr 63
addr 64 | addr 65 | addr 66 | ... | addr 95
...
```

### No Conflict (Ideal)
```python
# Each thread accesses different bank
shared[threadIdx.x]  # Thread i → Bank i ✅
```

### 2-Way Bank Conflict
```python
# Two threads access same bank
shared[threadIdx.x * 2]  # Threads 0,16 → Bank 0 ❌
                         # Threads 1,17 → Bank 2 ❌
```

### Avoiding Bank Conflicts
```python
# Add padding to shift addresses
# Instead of shared[32][32], use shared[32][33]
shared = cuda.shared.array(shape=(32, 33), dtype=float32)  # +1 padding
```

---

## 🔄 syncthreads() - Thread Synchronization

### Why Synchronize?
Threads execute independently. When sharing data via shared memory, you **must** ensure all threads have completed their writes before any thread reads.

```python
@cuda.jit
def sync_example(data, output):
    shared = cuda.shared.array(256, dtype=float32)
    tid = cuda.threadIdx.x
    
    # Phase 1: All threads write to shared memory
    shared[tid] = data[tid]
    
    # ⚠️ WITHOUT syncthreads(): Race condition!
    # Some threads might read before others write
    
    cuda.syncthreads()  # ✅ Barrier: wait for ALL threads
    
    # Phase 2: Now safe to read any shared memory location
    if tid < 255:
        output[tid] = shared[tid] + shared[tid + 1]
```

### Rules for syncthreads()
1. **All threads must reach it** - No divergent paths around sync
2. **Use after shared memory writes** - Before reading others' data
3. **Use before shared memory overwrites** - Ensure reads complete

---

## 📋 Memory Best Practices

| Goal | Strategy |
|------|----------|
| Maximize bandwidth | Coalesced global memory access |
| Reduce latency | Use shared memory for reused data |
| Avoid bank conflicts | Access consecutive addresses or add padding |
| Thread cooperation | Use shared memory + syncthreads |
| Read-only data | Use constant memory or `__ldg()` |

---

## ✅ Day 3 Summary

| Concept | Key Point |
|---------|-----------|
| Memory Hierarchy | Registers > Shared > L1/L2 > Global |
| Coalescing | Consecutive access = one transaction |
| Shared Memory | Fast, per-block, programmer-managed |
| Bank Conflicts | 32 banks, avoid same-bank access |
| syncthreads() | Barrier for thread synchronization |
