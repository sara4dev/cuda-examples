"""
Lesson 5: How cuDNN Auto-Detects and Uses Tensor Cores
Goal: Understand how libraries automatically choose between CUDA cores and Tensor cores
"""

import cupy as cp
import numpy as np

from cuda_utils import parse_compute_capability


def main():
    print("=" * 70)
    print("LESSON 5: How cuDNN Decides - Tensor Cores vs CUDA Cores")
    print("=" * 70)
    
    device = cp.cuda.Device()
    compute_cap = device.compute_capability
    major, minor = parse_compute_capability(compute_cap)
    
    print(f"\nYour GPU: Compute Capability {major}.{minor}")
    
    print("\n" + "=" * 70)
    print("HOW cuDNN/cuBLAS DECIDES WHICH CORES TO USE")
    print("=" * 70)
    
    print("""
The decision is made AUTOMATICALLY based on multiple factors:

┌─────────────────────────────────────────────────────────────┐
│ DECISION TREE: Tensor Core vs CUDA Core                    │
└─────────────────────────────────────────────────────────────┘

1. Check GPU Compute Capability
   ├─ < 7.0 → CUDA cores only (no tensor cores available)
   └─ >= 7.0 → Tensor cores available, continue checking...

2. Check Data Type
   ├─ FP32/FP64 → CUDA cores (tensor cores don't support)
   ├─ FP16/BF16 → Tensor cores (preferred)
   └─ TF32 (Ampere+) → Tensor cores automatically for FP32!

3. Check Operation Type
   ├─ Matrix multiplication (GEMM) → Tensor cores
   ├─ Convolution → Tensor cores (implemented as GEMM)
   ├─ Element-wise ops → CUDA cores
   └─ Other operations → CUDA cores

4. Check Matrix Dimensions
   ├─ Dimensions aligned to tensor core tiles → Tensor cores
   │  (multiples of 8 for FP16, 16 for INT8, etc.)
   └─ Small or misaligned → May use CUDA cores

5. Runtime Performance Heuristics
   ├─ cuDNN benchmarks different algorithms
   └─ Picks fastest one (may vary by size/hardware)
    """)
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Automatic Selection in Action")
    print("=" * 70)
    
    print("\nWe'll do the SAME operation (matrix multiply) with different data types")
    print("and see which cores get used:\n")
    
    size = 2048
    A = cp.random.randn(size, size, dtype=cp.float32)
    B = cp.random.randn(size, size, dtype=cp.float32)
    
    scenarios = [
        ("FP64 (double)", cp.float64, "CUDA cores", "No tensor core support"),
        ("FP32 (float)", cp.float32, "CUDA cores*", "Or TF32 tensor cores on Ampere+"),
        ("FP16 (half)", cp.float16, "Tensor cores", "Native tensor core support"),
    ]
    
    print(f"Matrix size: {size}×{size}\n")
    print("Data Type    | Execution Unit | Reason")
    print("-" * 70)
    
    for dtype_name, dtype, execution_unit, reason in scenarios:
        print(f"{dtype_name:12} | {execution_unit:14} | {reason}")
    
    print("\n* On Ampere+ GPUs, FP32 automatically uses TF32 on tensor cores!")
    print("  (same API, no code changes needed)")
    
    print("\n" + "=" * 70)
    print("THE MAGIC: TF32 (TensorFloat-32)")
    print("=" * 70)
    
    if major >= 8:  # Ampere or newer
        print("""
Your GPU supports TF32! This is AUTOMATIC on Ampere+ GPUs.

What happens when you write normal FP32 code:
┌────────────────────────────────────────────────────────┐
│ Your code: C = A @ B    (FP32 matrices)               │
│     ↓                                                  │
│ cuBLAS sees: FP32 operation                           │
│     ↓                                                  │
│ cuBLAS checks: GPU compute capability >= 8.0?         │
│     ↓                                                  │
│ YES! → Internally converts to TF32 format             │
│     ↓                                                  │
│ Dispatches to: TENSOR CORES (not CUDA cores!)        │
│     ↓                                                  │
│ Result: ~10x faster, same code, same API!             │
└────────────────────────────────────────────────────────┘

You get tensor core acceleration WITHOUT changing your code!
        """)
    else:
        print("""
Your GPU uses CUDA cores for FP32.
TF32 is available on Ampere+ GPUs (compute capability >= 8.0).
        """)
    
    print("\n" + "=" * 70)
    print("HOW cuDNN IMPLEMENTS THIS")
    print("=" * 70)
    
    print("""
Inside cuDNN/cuBLAS (simplified pseudocode):

```c++
Status cuDNN_Convolution(input, filter, output, dataType, ...) {
    // 1. Check hardware capabilities
    int compute_cap = getComputeCapability();
    bool has_tensor_cores = (compute_cap >= 70);
    
    // 2. Check data type compatibility
    bool can_use_tensor_cores = has_tensor_cores && 
        (dataType == FP16 || dataType == BF16 || 
         (dataType == FP32 && compute_cap >= 80));  // TF32
    
    // 3. Check dimension alignment
    bool dims_aligned = (dimensions % TENSOR_CORE_TILE == 0);
    
    // 4. Select algorithm
    if (can_use_tensor_cores && dims_aligned) {
        // Use tensor core implementation
        return tensorCoreGEMM(input, filter, output);
    } else {
        // Use CUDA core implementation
        return cudaCoreGEMM(input, filter, output);
    }
}
```

The decision is made AT RUNTIME, per operation!
    """)
    
    print("\n" + "=" * 70)
    print("ALIGNMENT MATTERS: Tensor Core Tile Sizes")
    print("=" * 70)
    
    print("""
Tensor cores work on specific tile sizes (matrix chunks):

Data Type | Tile Size | Matrix Dimension Preference
----------|-----------|---------------------------------
FP16      | 8×8 or    | Multiples of 8 (good)
          | 16×16     | Multiples of 16 (better)
----------|-----------|---------------------------------
TF32      | 16×8      | Multiples of 8/16
----------|-----------|---------------------------------  
INT8      | 32×8      | Multiples of 8
----------|-----------|---------------------------------

Example:
- Matrix 512×512 with FP16: ✓ PERFECT (512 = 16×32)
- Matrix 513×513 with FP16: ✗ Last row/col uses CUDA cores
- Matrix 1000×1000 with FP16: ~ OK (1000 = 8×125)
    """)
    
    print("\n" + "=" * 70)
    print("PRACTICAL DEMO: Aligned vs Unaligned")
    print("=" * 70)
    
    import time
    
    # Test aligned dimensions
    aligned_size = 1024  # Perfect: multiple of 16
    unaligned_size = 1025  # Unaligned: 1024 + 1
    
    print(f"\nComparing {aligned_size}×{aligned_size} vs {unaligned_size}×{unaligned_size}")
    print("(Same data type: FP16, same operation)\n")
    
    # Aligned case
    A_aligned = cp.random.randn(aligned_size, aligned_size, dtype=cp.float16)
    B_aligned = cp.random.randn(aligned_size, aligned_size, dtype=cp.float16)
    
    # Warmup
    _ = cp.matmul(A_aligned, B_aligned)
    cp.cuda.Stream.null.synchronize()
    
    start = time.perf_counter()
    for _ in range(10):
        C_aligned = cp.matmul(A_aligned, B_aligned)
    cp.cuda.Stream.null.synchronize()
    aligned_time = (time.perf_counter() - start) / 10
    
    # Unaligned case  
    A_unaligned = cp.random.randn(unaligned_size, unaligned_size, dtype=cp.float16)
    B_unaligned = cp.random.randn(unaligned_size, unaligned_size, dtype=cp.float16)
    
    # Warmup
    _ = cp.matmul(A_unaligned, B_unaligned)
    cp.cuda.Stream.null.synchronize()
    
    start = time.perf_counter()
    for _ in range(10):
        C_unaligned = cp.matmul(A_unaligned, B_unaligned)
    cp.cuda.Stream.null.synchronize()
    unaligned_time = (time.perf_counter() - start) / 10
    
    print(f"Aligned ({aligned_size}×{aligned_size}):     {aligned_time*1000:.2f} ms")
    print(f"Unaligned ({unaligned_size}×{unaligned_size}):   {unaligned_time*1000:.2f} ms")
    print(f"Slowdown: {(unaligned_time/aligned_time - 1)*100:.1f}%")
    
    print("\nWhy slower? Unaligned dimensions can't fully utilize tensor cores!")
    
    print("\n" + "=" * 70)
    print("CONTROLLING TENSOR CORE USAGE")
    print("=" * 70)
    
    print("""
You CAN control whether tensor cores are used:

1. **In PyTorch:**
   ```python
   # Enable TF32 for faster training (default on Ampere+)
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   
   # Disable TF32 (for maximum precision)
   torch.backends.cuda.matmul.allow_tf32 = False
   torch.backends.cudnn.allow_tf32 = False
   
   # Use automatic mixed precision (FP16 tensor cores)
   from torch.cuda.amp import autocast
   with autocast():
       output = model(input)  # Uses FP16 tensor cores
   ```

2. **In cuBLAS directly:**
   ```c++
   cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);  // Enable
   cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);   // Disable
   ```

3. **In cuDNN:**
   ```c++
   cudnnSetConvolutionMathType(
       convDesc, 
       CUDNN_TENSOR_OP_MATH  // Use tensor cores
   );
   ```

But usually, just let the library decide automatically!
    """)
    
    print("\n" + "=" * 70)
    print("HOW TO VERIFY TENSOR CORES ARE BEING USED")
    print("=" * 70)
    
    print("""
Several ways to check:

1. **NVIDIA Profiler (nsys/nvprof)**
   ```bash
   nsys profile --stats=true python your_script.py
   ```
   Look for: "tensor_op" or "hmma" operations (tensor cores)
   vs "s/d/cmma" operations (CUDA cores)

2. **Performance Comparison**
   - FP16 should be 2-16x faster than FP32
   - If not, tensor cores might not be engaged

3. **GPU Utilization**
   ```bash
   nvidia-smi dmon -s u
   ```
   Tensor core usage shows as high "sm" (streaming multiprocessor) usage

4. **PyTorch Profiler**
   ```python
   from torch.profiler import profile, ProfilerActivity
   with profile(activities=[ProfilerActivity.CUDA]) as prof:
       model(input)
   print(prof.key_averages().table())
   # Look for tensor core operations
   ```
    """)
    
    print("\n" + "=" * 70)
    print("COMMON MISCONCEPTIONS")
    print("=" * 70)
    
    print("""
❌ WRONG: "I need to write special code to use tensor cores"
✓ RIGHT:  Libraries automatically use them with compatible data types

❌ WRONG: "Tensor cores only work for deep learning"
✓ RIGHT:  Any matrix multiplication can use them (GEMM operations)

❌ WRONG: "FP32 never uses tensor cores"
✓ RIGHT:  On Ampere+, FP32 automatically uses TF32 tensor cores!

❌ WRONG: "I need to manually dispatch to tensor cores"
✓ RIGHT:  cuBLAS/cuDNN handle this automatically

❌ WRONG: "Small matrices should always use tensor cores"
✓ RIGHT:  There's overhead; CUDA cores may be faster for small ops
    """)
    
    print("\n" + "=" * 70)
    print("SUMMARY: The Automatic Decision Process")
    print("=" * 70)
    
    print("""
When you call cuBLAS/cuDNN operations:

1. Library checks your GPU's compute capability
2. Examines data types of your matrices/tensors
3. Checks dimensions and alignment
4. Looks up optimal algorithm from internal database
5. May run quick benchmarks (for new configurations)
6. Dispatches to either:
   - Tensor core kernels (wmma/mma instructions)
   - CUDA core kernels (fma instructions)

ALL OF THIS IS TRANSPARENT TO YOU!

You just write:
   C = A @ B  (Python/PyTorch)
   cublasGemm(...)  (C/C++)
   
And get optimal performance automatically.

Best practice: Use FP16/BF16 for training, let libraries handle the rest!
    """)


if __name__ == "__main__":
    main()



