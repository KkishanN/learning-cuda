# Learning CUDA - 7-Day GPU Programming Curriculum

Practice GPU programming with skeleton files. Fill in the TODOs!

## 📊 Progress

- [x] **Day 1**: CUDA Basics (vector-add, matrix-add, relu)
- [ ] **Day 2**: Threading & Indexing
- [ ] **Day 3**: Memory Management
- [ ] **Day 4**: Image Processing
- [ ] **Day 5**: Matrix Operations
- [ ] **Day 6**: Parallel Reduction & Atomics
- [ ] **Day 7**: CUDA Streams & Pipelines

## Structure

Each day has two folders:
- **`cu/`** - CUDA C++ files for [LeetGPU](https://leetgpu.com/)
- **`py/`** - Python files for local testing with Numba CUDA simulator

## Quick Setup (Python/Local)

```powershell
pip install numba numpy
$env:NUMBA_ENABLE_CUDASIM = "1"
python day2/py/vector_add.py
```

## Curriculum

| Day | Topic | Problems |
|-----|-------|----------|
| **Day 1** | Basics | `vector-add`, `matrix-add`, `relu` |
| **Day 2** | Threading | `thread_visualization`, `vector_add` |
| **Day 3** | Memory | `matrix_transpose`, `dot_product` |
| **Day 4** | Image Processing | `gaussian_blur`, `sobel_edge` |
| **Day 5** | Matrix Ops | `matrix_multiply`, `neural_layer` |
| **Day 6** | Reduction | `parallel_reduction`, `histogram` |
| **Day 7** | Streams | `streams`, `pipeline` |

## How to Use

1. Read the **problem statement** at the top of each file
2. Fill in the **TODO** sections
3. Test locally with Python (simulator)
4. Submit CUDA version to LeetGPU

## Concepts

Each `dayN/` folder has a `concepts.md` with detailed explanations.

## Advanced Topics

See `nextTopics/` for job-ready skills:
- TensorRT, Distributed GPU, LLM Inference, Custom Kernels, Profiling
