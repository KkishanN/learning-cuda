# 01. TensorRT Optimization

## 🎯 Overview
TensorRT is NVIDIA's high-performance deep learning inference optimizer and runtime. It's essential for deploying models in production.

---

## 🛠️ What is TensorRT?

TensorRT takes a trained neural network and optimizes it for inference:

```
PyTorch/TensorFlow Model
        ↓
[TensorRT Optimizer]
  - Layer fusion
  - Precision calibration
  - Kernel auto-tuning
        ↓
Optimized Engine (2-10x faster!)
```

---

## ⚡ Key Optimizations

### 1. Layer Fusion
Combines multiple layers into single optimized kernels:

```
Before:                  After:
[Conv] → [BN] → [ReLU]   [ConvBNReLU]
3 kernel launches        1 kernel launch
3 memory round-trips     1 memory round-trip
```

### 2. Precision Reduction

| Precision | Bits | Memory | Speed | Accuracy |
|-----------|------|--------|-------|----------|
| FP32 | 32 | 1× | 1× | Baseline |
| FP16 | 16 | 0.5× | 2× | ~Same |
| INT8 | 8 | 0.25× | 4× | Slight loss |

### 3. Kernel Auto-Tuning
TensorRT benchmarks multiple kernel implementations and picks the fastest for your specific GPU.

---

## 📊 Quantization Deep Dive

### FP16 (Half Precision)
- Automatic, no calibration needed
- Minimal accuracy loss
- 2× speedup typical

```python
# PyTorch FP16 inference
with torch.cuda.amp.autocast():
    output = model(input)
```

### INT8 (8-bit Integer)
- Requires calibration dataset
- Map FP32 range to INT8 range
- Up to 4× speedup

```
Calibration Process:
1. Run representative dataset through model
2. Collect activation statistics (min/max)
3. Compute scaling factors
4. Quantize weights and activations
```

---

## 🔧 TensorRT Python Workflow

```python
import tensorrt as trt

# 1. Create builder and network
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(flags)

# 2. Parse ONNX model
parser = trt.OnnxParser(network, logger)
with open("model.onnx", "rb") as f:
    parser.parse(f.read())

# 3. Configure optimization
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

# 4. Build engine
engine = builder.build_serialized_network(network, config)

# 5. Save for deployment
with open("model.engine", "wb") as f:
    f.write(engine)
```

---

## 📈 Performance Comparison

Typical speedups on common models:

| Model | FP32 | FP16 | INT8 |
|-------|------|------|------|
| ResNet-50 | 1× | 2.1× | 3.8× |
| BERT-Base | 1× | 1.9× | 3.2× |
| GPT-2 | 1× | 2.0× | 2.8× |
| Stable Diffusion | 1× | 2.5× | N/A* |

*Some models don't work well with INT8

---

## 🎓 Learning Resources

1. **NVIDIA TensorRT Documentation** - Official guide
2. **TensorRT Developer Guide** - Deep technical details
3. **NVIDIA Deep Learning Examples** - GitHub repo with optimized models
4. **Triton Inference Server** - Deploy TensorRT models at scale

---

## 💼 Interview Topics

- How does layer fusion reduce latency?
- Explain INT8 calibration process
- When would you NOT use TensorRT?
- How to handle dynamic input shapes?
- Debugging accuracy loss after quantization
