# How cuDNN Auto-Detects and Uses Tensor Cores

## Quick Answer

**cuDNN automatically decides** between tensor cores and CUDA cores based on:
1. **Data type** (FP16/BF16 → tensor cores, FP32/FP64 → CUDA cores*)
2. **GPU compute capability** (>= 7.0 for tensor cores)
3. **Matrix dimensions** (aligned to tensor core tiles = faster)
4. **Operation type** (matrix multiply/convolution → tensor cores)

*Exception: On Ampere+ GPUs (compute 8.0+), FP32 automatically uses **TF32 tensor cores**!

---

## The Decision Tree (What Happens Inside cuDNN)

```
┌──────────────────────────────────────────────────────────────┐
│ You call: cudnnConvolutionForward() or cublasGemm()         │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 1: Check GPU Compute Capability                        │
├──────────────────────────────────────────────────────────────┤
│ if (compute_capability < 7.0)                                │
│     → Use CUDA cores only (no tensor cores available)       │
│ else                                                         │
│     → Continue to Step 2                                     │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 2: Check Data Type                                     │
├──────────────────────────────────────────────────────────────┤
│ FP64 (double)   → CUDA cores (tensor cores don't support)   │
│ FP32 (float)    → CUDA cores OR TF32 tensor cores (Ampere+) │
│ FP16 (half)     → Tensor cores ✓                            │
│ BF16            → Tensor cores ✓ (Ampere+)                  │
│ INT8            → Tensor cores ✓                            │
│ FP8             → Tensor cores ✓ (Ada+)                     │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 3: Check Operation Type                                │
├──────────────────────────────────────────────────────────────┤
│ Matrix Multiplication (GEMM)  → Tensor cores                │
│ Convolution                    → Tensor cores (as GEMM)     │
│ Element-wise operations        → CUDA cores                 │
│ Activation functions           → CUDA cores                 │
│ Reductions                     → CUDA cores                 │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 4: Check Dimension Alignment                           │
├──────────────────────────────────────────────────────────────┤
│ Are dimensions multiples of tensor core tile size?          │
│   FP16: multiples of 8 or 16  → Optimal                    │
│   INT8: multiples of 32       → Optimal                    │
│   Misaligned                  → May use CUDA cores or       │
│                                  padding (slower)            │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 5: Select Best Kernel                                  │
├──────────────────────────────────────────────────────────────┤
│ cuDNN has MULTIPLE implementations for each operation:      │
│   - Pure tensor core kernels                                │
│   - Pure CUDA core kernels                                  │
│   - Hybrid kernels                                          │
│                                                              │
│ Selection based on:                                         │
│   - Internal heuristics database                            │
│   - Runtime benchmarking (for new configs)                  │
│   - Fastest option wins!                                    │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Result: Dispatch to optimal kernel                          │
│   → Tensor core kernel (uses wmma/mma instructions)         │
│   → CUDA core kernel (uses fma instructions)                │
└──────────────────────────────────────────────────────────────┘
```

---

## Code Example: What Happens Internally

### Your Code (PyTorch):
```python
import torch

# You write simple code
A = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
B = torch.randn(1024, 1024, dtype=torch.float16, device='cuda')
C = A @ B  # Matrix multiplication
```

### What Happens Behind the Scenes:

```c++
// Inside PyTorch → cuBLAS
cublasGemmEx(handle, 
    CUBLAS_OP_N, CUBLAS_OP_N,
    1024, 1024, 1024,  // Matrix dimensions
    alpha, 
    A_ptr, CUDA_R_16F,  // FP16 input A
    B_ptr, CUDA_R_16F,  // FP16 input B
    beta,
    C_ptr, CUDA_R_16F,  // FP16 output C
    CUDA_R_32F,         // FP32 accumulation
    CUBLAS_GEMM_DEFAULT_TENSOR_OP  // Allow tensor cores
);

// Inside cuBLAS implementation:
if (gpu_compute_cap >= 70 &&           // Has tensor cores?
    datatype == FP16 &&                // Compatible data type?
    dims_aligned_to_16 &&              // Good alignment?
    mathMode == TENSOR_OP_MATH) {      // Tensor ops enabled?
    
    // Launch tensor core kernel
    launch_wmma_gemm_kernel<<<grid, block>>>(A, B, C);
    //     ↑↑↑↑
    //     Uses tensor core instructions (wmma = warp matrix multiply-accumulate)
    
} else {
    // Launch CUDA core kernel
    launch_standard_gemm_kernel<<<grid, block>>>(A, B, C);
    //     ↑↑↑↑↑↑↑↑
    //     Uses regular FMA instructions
}
```

---

## The Magic of TF32 (Ampere+ GPUs)

On RTX 30/40/50 series (Ampere, Ada, Blackwell):

### Your FP32 Code:
```python
# You write regular FP32 code
A = torch.randn(1024, 1024, dtype=torch.float32, device='cuda')
B = torch.randn(1024, 1024, dtype=torch.float32, device='cuda')
C = A @ B  # Standard FP32 matmul
```

### What cuBLAS Does Automatically:

```
Input: FP32 data (A and B)
   ↓
cuBLAS detects: compute_capability >= 8.0
   ↓
Internally converts: FP32 → TF32 (rounded)
   ↓
Dispatch to: TENSOR CORES (not CUDA cores!)
   ↓
Compute: Using tensor core TF32 operations
   ↓
Output: FP32 result (reconverted)

Result: ~10x speedup, ZERO code changes!
```

---

## Dimension Alignment Example

### Perfectly Aligned (Fast):
```python
A = torch.randn(1024, 1024, dtype=torch.float16)  # 1024 = 64×16
B = torch.randn(1024, 1024, dtype=torch.float16)
C = A @ B
# ✓ All dimensions divisible by 16 → Full tensor core utilization
```

### Misaligned (Slower):
```python
A = torch.randn(1025, 1025, dtype=torch.float16)  # 1025 = 64×16 + 1
B = torch.randn(1025, 1025, dtype=torch.float16)
C = A @ B
# ✗ Extra row/column can't use tensor cores efficiently
# → Padding added (slower) or CUDA cores used for remainder
```

---

## How to Control Tensor Core Usage

### 1. PyTorch
```python
import torch

# Enable TF32 (default on Ampere+, gives ~10x speedup for FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Disable TF32 (for maximum precision)
torch.backends.cuda.matmul.allow_tf32 = False

# Use FP16 tensor cores explicitly (Automatic Mixed Precision)
from torch.cuda.amp import autocast
with autocast():
    output = model(input)  # Uses FP16 tensor cores
```

### 2. Direct cuBLAS Control
```c++
cublasHandle_t handle;
cublasCreate(&handle);

// Enable tensor cores
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

// Disable tensor cores (use only CUDA cores)
cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
```

### 3. cuDNN Control
```c++
cudnnConvolutionDescriptor_t convDesc;
cudnnCreateConvolutionDescriptor(&convDesc);

// Use tensor cores
cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH);

// Use CUDA cores only
cudnnSetConvolutionMathType(convDesc, CUDNN_DEFAULT_MATH);
```

---

## How to Verify Tensor Cores Are Being Used

### Method 1: Performance Check
```python
import torch
import time

size = 4096
A = torch.randn(size, size, dtype=torch.float32, device='cuda')
B = torch.randn(size, size, dtype=torch.float32, device='cuda')

# Test FP32 (should use TF32 tensor cores on Ampere+)
start = time.time()
C = A @ B
torch.cuda.synchronize()
fp32_time = time.time() - start

# Test FP16 (definitely uses tensor cores)
A_fp16 = A.half()
B_fp16 = B.half()
start = time.time()
C_fp16 = A_fp16 @ B_fp16
torch.cuda.synchronize()
fp16_time = time.time() - start

print(f"FP32: {fp32_time:.4f}s")
print(f"FP16: {fp16_time:.4f}s")
print(f"Speedup: {fp32_time/fp16_time:.2f}x")

# If you see 5-20x speedup, tensor cores are working!
```

### Method 2: NVIDIA Profiler
```bash
# Profile your script
nsys profile --stats=true python your_script.py

# Look for in the output:
# - "tensorOp" or "hmma" instructions = Tensor cores ✓
# - "ffma" or "fma" instructions = CUDA cores
```

### Method 3: GPU Metrics
```bash
# Monitor while running
nvidia-smi dmon -s u

# High tensor core utilization shows as:
# - High SM (streaming multiprocessor) activity
# - With compatible data types (FP16/BF16)
```

---

## Summary Table

| Factor | Tensor Cores | CUDA Cores |
|--------|-------------|------------|
| **Data Type** | FP16, BF16, TF32, INT8, FP8 | FP32, FP64, Any type |
| **Operation** | Matrix multiply (A×B+C) | Any operation |
| **Speed** | 5-20x faster for matmul | Standard |
| **Precision** | Lower (FP16 ≈ 3 decimals) | Higher (FP32 ≈ 7 decimals) |
| **Availability** | Compute ≥ 7.0 | All GPUs |
| **Use Case** | Deep learning, linear algebra | General compute |
| **Programming** | Automatic via libraries | Direct in kernels |
| **Alignment** | Benefits from aligned dims | Works with any dims |

---

## Key Takeaways

1. **You don't need to manually select** - cuDNN/cuBLAS automatically choose the best option

2. **Use FP16/BF16 for ML** - Tensor cores give massive speedups with acceptable precision

3. **TF32 is automatic** - On Ampere+ GPUs, your FP32 code automatically gets faster

4. **Alignment matters** - Dimension multiples of 8/16 run faster on tensor cores

5. **Libraries handle everything** - PyTorch, TensorFlow, cuDNN all automatically utilize tensor cores

6. **Both cores work together** - CUDA cores handle non-matrix ops, tensor cores handle matmuls

---

## Next Steps

Run the lessons to see it in action:
```bash
./run_lesson.sh 04_tensor_cores.py        # See tensor core performance
./run_lesson.sh 05_cudnn_tensor_cores.py  # See automatic selection
```



