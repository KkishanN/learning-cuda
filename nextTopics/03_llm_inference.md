# 03. LLM Inference Optimization

## 🎯 Overview
Large Language Model (LLM) inference is the **hottest skill in GPU programming** right now. Master these techniques to land top-paying AI infrastructure roles.

---

## 🧠 LLM Inference Basics

### The Problem
- GPT-4: ~1.8 trillion parameters
- 70B model = 140GB in FP16
- Generation is **autoregressive** (sequential)

```
Input: "The quick brown"
Step 1: "The quick brown" → "fox"
Step 2: "The quick brown fox" → "jumps"
Step 3: "The quick brown fox jumps" → "over"
...
Each step depends on previous!
```

---

## 🔑 Key Optimization Techniques

### 1. KV Cache
Store computed key-value pairs instead of recomputing:

```
Without KV Cache:
Step N: Compute attention for ALL N tokens

With KV Cache:
Step N: Compute attention for NEW token only
        Reuse K,V from steps 1 to N-1
        
Speedup: O(N²) → O(N)
```

Memory cost: ~2 × layers × hidden_size × sequence_length × batch_size

### 2. PagedAttention (vLLM)
Manage KV cache like virtual memory:

```
Traditional: Contiguous allocation per request
┌─────────────────────────────────────┐
│ Request 1 KV Cache (reserved max)   │ ← Wasted space!
├─────────────────────────────────────┤
│ Request 2 KV Cache (reserved max)   │
└─────────────────────────────────────┘

PagedAttention: Block-based allocation
┌──────┬──────┬──────┬──────┬──────┬──────┐
│ R1.1 │ R1.2 │ R2.1 │ R1.3 │ R2.2 │ Free │
└──────┴──────┴──────┴──────┴──────┴──────┘
         ↑ Blocks allocated on-demand
```

Benefits:
- Near-zero waste
- 24× more concurrent requests
- Dynamic sequence lengths

### 3. Continuous Batching
Don't wait for slowest request:

```
Traditional Batching:
Request 1: [████████████████] (1000 tokens)
Request 2: [████████........] (500 tokens, waiting)
Request 3: [████............] (200 tokens, waiting)
           ↑ All wait for Request 1

Continuous Batching:
Request 3 finishes → immediately insert Request 4
No waiting, maximum GPU utilization!
```

### 4. Speculative Decoding
Use small model to draft, large model to verify:

```
Draft (7B model): "The quick brown fox" → "jumps over the lazy"
Verify (70B model): Accept 4/5 tokens in one forward pass
                    
Result: 2-3× speedup with no quality loss!
```

---

## 🛠️ Key Tools

### vLLM
```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
params = SamplingParams(temperature=0.7, max_tokens=100)

outputs = llm.generate(["Hello, how are"], params)
print(outputs[0].outputs[0].text)
```

Features:
- PagedAttention
- Continuous batching
- Tensor parallelism
- OpenAI-compatible API

### TensorRT-LLM
NVIDIA's optimized LLM inference:
- Custom attention kernels
- INT4/INT8 quantization
- Multi-GPU support
- Flash Attention integration

```bash
# Convert and build
python convert_checkpoint.py --model_dir llama-7b
trtllm-build --checkpoint_dir ./checkpoint --output_dir ./engine
```

### Text Generation Inference (TGI)
HuggingFace's production server:
- Flash Attention
- Quantization (bitsandbytes, GPTQ)
- Safetensors support
- Prometheus metrics

---

## 📊 Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| TTFT (Time To First Token) | Latency to start generating | < 100ms |
| TPOT (Time Per Output Token) | Per-token generation time | < 50ms |
| Throughput | Tokens/second | Maximize |
| Memory Efficiency | Tokens served per GB VRAM | Maximize |

---

## 🔬 Advanced Techniques

### Flash Attention
Fused attention kernel with tiling for O(N) memory:

```
Standard Attention: O(N²) memory (store full attention matrix)
Flash Attention: O(N) memory (tile-based, never materialize full matrix)
```

### Quantization
```
FP16: 2 bytes per param → 14GB for 7B model
INT8: 1 byte per param → 7GB for 7B model
INT4: 0.5 byte per param → 3.5GB for 7B model
```

### Tensor Cores
Use specialized hardware for matrix ops:
- FP16: 312 TFLOPS on A100
- INT8: 624 TOPS on A100

---

## 💼 Interview Topics

- Explain KV cache and why it's necessary
- How does PagedAttention improve memory efficiency?
- Trade-offs of different quantization levels
- Continuous batching vs static batching
- Memory breakdown of serving a 70B model
- How would you reduce TTFT?
- Speculative decoding algorithm and assumptions
