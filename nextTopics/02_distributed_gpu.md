# 02. Distributed GPU Programming

## 🎯 Overview
Scale your GPU workloads across multiple GPUs and multiple machines. Essential for training large models and high-throughput inference.

---

## 🖥️ Multi-GPU Architectures

### Single Machine, Multiple GPUs
```
┌─────────────────────────────────────┐
│             Host CPU                │
├────────┬────────┬────────┬──────────┤
│  GPU0  │  GPU1  │  GPU2  │  GPU3    │
│ PCIe   │ PCIe   │ PCIe   │ PCIe     │
└────────┴────────┴────────┴──────────┘
     ↕ NVLink (if available) ↕
```

### Multi-Node Cluster
```
┌─────────────┐    Network    ┌─────────────┐
│   Node 0    │◄────────────►│   Node 1    │
│ GPU0  GPU1  │  (InfiniBand) │ GPU2  GPU3  │
└─────────────┘               └─────────────┘
```

---

## 🔗 NCCL (NVIDIA Collective Communications Library)

NCCL optimizes GPU-to-GPU communication:

### Key Operations

| Operation | Description | Use Case |
|-----------|-------------|----------|
| AllReduce | Sum all values, distribute result | Gradient averaging |
| Broadcast | Send from one to all | Model distribution |
| AllGather | Collect all values together | Activation sharing |
| ReduceScatter | Reduce + scatter | ZeRO optimizer |

### AllReduce Example (Gradient Sync)
```
GPU 0: [1, 2, 3]  ─┐
GPU 1: [4, 5, 6]  ─┼──► AllReduce(sum) ──┬─► GPU 0: [5, 7, 9]
GPU 2: [0, 0, 0]  ─┘                     ├─► GPU 1: [5, 7, 9]
                                         └─► GPU 2: [5, 7, 9]
```

---

## 🔄 Data Parallelism

Most common distributed training strategy:

```python
# PyTorch DistributedDataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')

# Wrap model
model = DDP(model.to(rank), device_ids=[rank])

# Training loop (gradients auto-synchronized)
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # AllReduce happens here
    optimizer.step()
```

### How It Works
1. Each GPU gets copy of model
2. Each GPU processes different data batch
3. Gradients synchronized via AllReduce
4. All GPUs update identically

---

## 🧩 Model Parallelism

For models too large for single GPU:

### Tensor Parallelism
Split individual layers across GPUs:
```
Linear(4096, 4096) split across 4 GPUs:
  GPU0: Linear(4096, 1024)
  GPU1: Linear(4096, 1024)
  GPU2: Linear(4096, 1024)
  GPU3: Linear(4096, 1024)
  Output: Concatenate results
```

### Pipeline Parallelism
Different layers on different GPUs:
```
GPU 0: Layers 0-10   ───micro-batch-0───►
GPU 1: Layers 10-20  ─────────────────── micro-batch-0 ───►
GPU 2: Layers 20-30  ─────────────────────────────────── micro-batch-0 ───►
```

---

## 🚀 DeepSpeed

Microsoft's library for efficient large model training:

### ZeRO (Zero Redundancy Optimizer)
Reduces memory by partitioning optimizer states, gradients, and parameters:

| Stage | Partitioned | Memory Savings |
|-------|-------------|----------------|
| ZeRO-1 | Optimizer states | 4× |
| ZeRO-2 | + Gradients | 8× |
| ZeRO-3 | + Parameters | Linear with GPU count |

### Example Config
```json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_bucket_size": 5e8,
    "reduce_bucket_size": 5e8
  },
  "fp16": {
    "enabled": true
  }
}
```

---

## 📊 Scaling Efficiency

Ideal vs real scaling:

```
GPUs:     1    2    4    8    16
Ideal:    1×   2×   4×   8×   16×
Reality:  1×   1.9× 3.6× 6.8× 12×
```

Overhead sources:
- Communication time
- Synchronization barriers
- Load imbalance
- Memory copies

---

## 💼 Interview Topics

- Explain AllReduce and when to use it
- Data Parallelism vs Model Parallelism trade-offs
- What is gradient bucketing?
- How does ZeRO reduce memory?
- Debugging NCCL hangs and deadlocks
- Ring AllReduce algorithm
