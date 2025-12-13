# Day 7: CUDA Streams & Async Programming

## 🎯 Learning Objectives
- Understand CUDA streams for async execution
- Overlap computation with data transfer
- Use events for timing and synchronization
- Build efficient data pipelines

---

## 🌊 What are CUDA Streams?

A **stream** is a sequence of operations that execute in order.
Multiple streams can execute **concurrently**.

```
Default (synchronous):
┌──────────────────────────────────────────────────┐
│ Copy H→D │ Kernel 1 │ Kernel 2 │ Copy D→H │     │
└──────────────────────────────────────────────────┘
                Time →

With Streams (overlapped):
Stream 0: │ Copy H→D │ Kernel 1 ├─────────│ Copy D→H │
Stream 1: │          │ Copy H→D │ Kernel 2 │ Copy D→H │
          ├──────────┴──────────┴──────────┴──────────┤
                    Time → (shorter!)
```

---

## ⚡ Why Use Streams?

| Without Streams | With Streams |
|-----------------|--------------|
| Operations sequential | Operations overlap |
| GPU often idle | GPU fully utilized |
| Memory copies block | Copies overlap with compute |

### Ideal Use Case
- Processing multiple independent data chunks
- Pipelining: load-compute-store pattern
- Large data transfers

---

## 🔧 Stream API (Numba)

```python
# Create a stream
stream = cuda.stream()

# Launch kernel in stream
kernel[blocks, threads, stream](args)

# Async memory operations
cuda.to_device(data, stream=stream)
cuda.copy_to_host(d_data, stream=stream)

# Wait for stream to complete
stream.synchronize()
```

---

## ⏱️ CUDA Events

Events mark points in stream execution for:
1. **Timing**: Measure kernel execution time
2. **Synchronization**: Wait for specific points

```python
# Create events
start = cuda.event()
end = cuda.event()

# Record events
start.record(stream)
kernel[grid, block, stream](args)
end.record(stream)

# Wait and get elapsed time
end.synchronize()
elapsed_ms = cuda.event_elapsed_time(start, end)
```

---

## 🔄 Pipeline Pattern

Process data in chunks with overlapping stages:

```
Chunk 0:   [Load] [Compute] [Store]
Chunk 1:          [Load]    [Compute] [Store]
Chunk 2:                    [Load]    [Compute] [Store]

Each stage uses different hardware:
- Load:    CPU → GPU memory (DMA engine)
- Compute: CUDA cores
- Store:   GPU → CPU memory (DMA engine)
```

### Benefits
- Hide memory transfer latency
- Keep GPU continuously busy
- Better throughput for streaming data

---

## ⚠️ Synchronization Considerations

### Rules
1. Operations in **same stream** execute in order
2. Operations in **different streams** can overlap
3. Kernel launches are **async** - return immediately
4. `stream.synchronize()` waits for all ops in stream

### Common Pitfall
```python
# BAD: Using result before kernel completes
kernel[grid, block](d_data)
result = d_data.copy_to_host()  # May get wrong data!

# GOOD: Ensure kernel completes
kernel[grid, block](d_data)
cuda.synchronize()  # Wait for all kernels
result = d_data.copy_to_host()  # Now safe
```

---

## ✅ Day 7 Summary

| Concept | Key Point |
|---------|-----------|
| Stream | Sequence of operations on GPU |
| Concurrency | Multiple streams run in parallel |
| Events | Timing and synchronization points |
| Pipeline | Overlap load-compute-store stages |
| Best Use | Multiple independent data chunks |
