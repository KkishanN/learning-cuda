# 05. Profiling & Debugging

## 🎯 Overview
Performance profiling is essential for GPU optimization. Learn to identify bottlenecks and validate optimizations.

---

## 🔬 NVIDIA Nsight Systems

System-wide performance analysis:

### What It Shows
- GPU kernel timelines
- CPU-GPU synchronization
- Memory transfers
- CUDA API calls

### Basic Usage
```bash
# Profile application
nsys profile --trace=cuda,nvtx python train.py

# Generate report
nsys stats report.qdrep
```

### Key Metrics
```
┌────────────────────────────────────────────────┐
│ Timeline View                                  │
├────────────────────────────────────────────────┤
│ CPU Thread ████░░░░████░░░░████               │
│ CUDA API   ░░██░░░░░░██░░░░░░██               │
│ GPU Kernel ░░░░████░░░░████░░░░████           │
│ Memory     ░░░░░░░░██░░░░░░░░░░░░░░           │
└────────────────────────────────────────────────┘
   │          │
   │          └─ Gaps = GPU idle time
   └─ Find long CPU sections (bottleneck!)
```

---

## 🔍 NVIDIA Nsight Compute

Kernel-level analysis:

### Metrics
- **SM Utilization**: Are all SMs busy?
- **Memory Throughput**: GB/s achieved vs theoretical
- **Occupancy**: Active warps / max warps
- **Compute/Memory Bound**: Which limits performance?

### Usage
```bash
# Profile specific kernel
ncu --set full python train.py

# Target specific kernel
ncu --kernel-name "matmul" python train.py
```

### Roofline Model
```
        │
        │        Compute Bound
Compute │         ╱
 (GFLOPS)│       ╱
        │     ╱  ● Your kernel
        │   ╱
        │ ╱ Memory Bound
        │╱─────────────────────
            Arithmetic Intensity (FLOPS/byte)
```

---

## 🐍 PyTorch Profiler

Integrated Python profiling:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    with record_function("model_inference"):
        output = model(input)

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for TensorBoard
prof.export_chrome_trace("trace.json")
```

### Common Issues Found
| Issue | Symptom | Fix |
|-------|---------|-----|
| CPU bottleneck | GPU utilization gaps | Move preprocessing to GPU |
| Small kernels | Many tiny launches | Batch operations |
| Memory bound | Low compute utilization | Reduce memory access |
| Sync overhead | Frequent synchronization | Async operations |

---

## 🐛 Debugging Techniques

### 1. cuda-memcheck
```bash
# Check for memory errors
compute-sanitizer --tool memcheck python script.py

# Check race conditions
compute-sanitizer --tool racecheck python script.py
```

### 2. Printf Debugging
```cpp
__global__ void kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Debug: value = %f\n", some_value);
    }
}
```

### 3. Python CUDA Debugging
```python
# Synchronous execution for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Check for NaN/Inf
torch.autograd.set_detect_anomaly(True)
```

---

## 📊 Common Pitfalls

### 1. Memory Bandwidth Limited
```
Symptom: Low compute utilization despite high occupancy
Check:   Memory throughput near hardware limit?
Fix:     - Reduce memory access
         - Use shared memory caching
         - Improve coalescing
```

### 2. Low Occupancy
```
Symptom: SM utilization low
Check:   Registers per thread, shared memory per block
Fix:     - Use fewer registers
         - Smaller thread blocks
         - Launch more blocks
```

### 3. Kernel Launch Overhead
```
Symptom: Many tiny kernels
Check:   < 100μs kernel times
Fix:     - Fuse kernels
         - Batch small operations
         - Use CUDA graphs
```

---

## 📈 Optimization Workflow

```
1. Profile baseline
      ↓
2. Identify bottleneck
   - Compute bound? → Optimize algorithm
   - Memory bound?  → Reduce access, cache
   - CPU bound?     → Move to GPU
      ↓
3. Implement optimization
      ↓
4. Profile again
      ↓
5. Verify improvement
      ↓
Repeat until satisfied
```

---

## 💼 Interview Topics

- How to identify compute vs memory bound?
- Explain roofline model
- What is SM occupancy and why does it matter?
- How would you debug a kernel producing wrong results?
- Explain memory coalescing issues and fixes
- CUDA Graphs and when to use them
