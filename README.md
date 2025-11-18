# CUDA Examples - Learn One Concept at a Time

Step-by-step CUDA programming lessons using Python bindings.

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (version 12.0+)
- Python 3.12+

## Installation

```bash
pip install -e .
```

## Running Lessons

**Option 1** - Using wrapper script (recommended for lesson 3+):
```bash
./run_lesson.sh 01_device_info.py
./run_lesson.sh 02_memory_basics.py
./run_lesson.sh 03_first_kernel.py
./run_lesson.sh 06_cuda_graphs.py
```

**Option 2** - Direct execution (works for lessons 1-2):
```bash
python 01_device_info.py
python 02_memory_basics.py
```

## Lessons

Start with Lesson 1 and work your way through!

### Lesson 1: Device Info 🎯
**File:** `01_device_info.py`

**What you'll learn:**
- What is a GPU device
- How to query GPU properties
- Understanding SMs, warps, and compute capability

**Key concepts:** Device, Streaming Multiprocessors (SMs), Warp size

```bash
python 01_device_info.py
```

---

### Lesson 2: Memory Basics 💾
**File:** `02_memory_basics.py`

**What you'll learn:**
- Allocating GPU memory
- Copying data between CPU and GPU
- Host vs Device memory

**Key concepts:** cuMemAlloc, cuMemcpyHtoD, cuMemcpyDtoH

```bash
python 02_memory_basics.py
```

---

### Lesson 3: First Kernel 🚀
**File:** `03_first_kernel.py`

**What you'll learn:**
- Writing a CUDA kernel
- Understanding threads and blocks
- Launching kernels
- Calculating global thread IDs

**Key concepts:** Thread, Block, Grid, threadIdx, blockIdx

```bash
./run_lesson.sh 03_first_kernel.py
```

---

### Lesson 4: Tensor Cores ⚡
**File:** `04_tensor_cores.py`

**What you'll learn:**
- What are tensor cores vs CUDA cores
- Performance comparison
- When to use tensor cores
- Supported data types (FP16, BF16, TF32, FP8)

**Key concepts:** Tensor cores, Matrix operations, FP16, Mixed precision

```bash
./run_lesson.sh 04_tensor_cores.py
```

---

### Lesson 5: cuDNN Auto-Selection 🤖
**File:** `05_cudnn_tensor_cores.py`

**What you'll learn:**
- How cuDNN/cuBLAS automatically choose between tensor cores and CUDA cores
- Alignment and performance considerations
- TF32 automatic acceleration
- How to verify and control tensor core usage

**Key concepts:** cuDNN, cuBLAS, Automatic dispatch, TF32, Profiling

```bash
./run_lesson.sh 05_cudnn_tensor_cores.py
```

---

### Lesson 6: CUDA Graphs 📊
**Files:** `06_cuda_graphs.py`, `06_cuda_graphs.cu`, `CUDA_GRAPHS_GUIDE.md`

**What you'll learn:**
- What are CUDA Graphs and why they matter
- How to reduce CPU launch overhead by 5-10x
- Stream capture API for easy graph creation
- When to use graphs vs traditional kernel launches
- Real-world performance comparisons

**Key concepts:** CUDA Graphs, Stream capture, Launch overhead, Graph instantiation

**Python version:**
```bash
./run_lesson.sh 06_cuda_graphs.py
```

**C++ version:**
```bash
nvcc -o cuda_graphs 06_cuda_graphs.cu -O3
./cuda_graphs
```

**Read the guide:** Check out `CUDA_GRAPHS_GUIDE.md` for comprehensive documentation

---

## Quick Reference

### CUDA Hierarchy

```
Grid                    (All threads executing a kernel)
└─ Blocks              (Groups of threads, max 1024)
   └─ Threads          (Individual execution units)
```

### Global Thread ID Formula

```cpp
// 1D: Each thread processes one array element
int global_id = blockIdx.x * blockDim.x + threadIdx.x;
```

### Memory Copy Directions

```
Host (CPU) ←→ Device (GPU)

HtoD: Host to Device (CPU → GPU)
DtoH: Device to Host (GPU → CPU)
```

## Resources

### Official Documentation
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuda-python Documentation](https://nvidia.github.io/cuda-python/)
- [CUDA Graphs Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)

### Guides in This Repository
- `TENSOR_CORE_GUIDE.md` - Complete guide to Tensor Cores
- `CUDA_GRAPHS_GUIDE.md` - Complete guide to CUDA Graphs

## Next Steps

After completing these lessons, you can explore:
- Vector addition
- Matrix operations
- Shared memory
- Optimization techniques

