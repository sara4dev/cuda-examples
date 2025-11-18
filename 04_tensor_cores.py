"""
Lesson 4: Understanding Tensor Cores
Goal: Learn what tensor cores are and how they differ from CUDA cores

Tensor Cores = Specialized hardware for matrix operations (A × B + C)
"""

import cupy as cp
import numpy as np
import time


def main():
    print("=" * 70)
    print("LESSON 4: Understanding Tensor Cores")
    print("=" * 70)
    
    # Get device info
    device = cp.cuda.Device()
    attrs = device.attributes
    
    print("\n" + "=" * 70)
    print("WHAT ARE TENSOR CORES?")
    print("=" * 70)
    print("""
Your GPU has TWO types of compute units:

1. CUDA CORES (also called FP32 cores)
   - General purpose processors
   - Handle ANY computation (add, multiply, sine, etc.)
   - Flexible but slower for matrix operations
   - What we used in lessons 1-3

2. TENSOR CORES
   - Specialized matrix multiply-accumulate units
   - ONLY do: D = A × B + C (matrix operations)
   - Extremely fast for this specific operation
   - Critical for AI/deep learning workloads
    """)
    
    # Check tensor core support
    compute_cap = device.compute_capability
    major, minor = divmod(compute_cap, 10)
    
    print("\n" + "=" * 70)
    print("YOUR GPU CAPABILITIES")
    print("=" * 70)
    print(f"Compute Capability: {major}.{minor}")
    print(f"CUDA Cores per SM: {attrs['MaxThreadsPerMultiProcessor']}")
    
    # Tensor core availability by architecture
    if compute_cap >= 120:
        print(f"✓ 5th Gen Tensor Cores (FP8, FP16, BF16, TF32, INT8)")
        print(f"  Architecture: Blackwell (RTX 50 series)")
        has_tensor_cores = True
    elif compute_cap >= 89:
        print(f"✓ 4th Gen Tensor Cores (FP8, FP16, BF16, TF32, INT8)")
        print(f"  Architecture: Ada Lovelace (RTX 40 series)")
        has_tensor_cores = True
    elif compute_cap >= 80:
        print(f"✓ 3rd Gen Tensor Cores (TF32, FP16, BF16, INT8)")
        print(f"  Architecture: Ampere (RTX 30 series, A100)")
        has_tensor_cores = True
    elif compute_cap >= 75:
        print(f"✓ 2nd Gen Tensor Cores (FP16, INT8, INT4)")
        print(f"  Architecture: Turing (RTX 20 series)")
        has_tensor_cores = True
    elif compute_cap >= 70:
        print(f"✓ 1st Gen Tensor Cores (FP16)")
        print(f"  Architecture: Volta (V100)")
        has_tensor_cores = True
    else:
        print(f"✗ No Tensor Cores")
        print(f"  Only CUDA cores available")
        has_tensor_cores = False
    
    if not has_tensor_cores:
        print("\nYour GPU doesn't have tensor cores.")
        print("But you can still follow along to understand the concept!")
        return
    
    # Demo: Matrix multiplication performance
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Tensor Core Performance")
    print("=" * 70)
    print("\nLet's multiply two large matrices: C = A × B")
    
    # Different matrix sizes to test
    sizes = [512, 1024, 2048, 4096]
    
    print("\nMatrix Size | FP32 (CUDA cores) | FP16 (Tensor cores) | Speedup")
    print("-" * 70)
    
    for size in sizes:
        # Create random matrices
        # FP32 - uses CUDA cores
        A_fp32 = cp.random.randn(size, size, dtype=cp.float32)
        B_fp32 = cp.random.randn(size, size, dtype=cp.float32)
        
        # FP16 - can use tensor cores
        A_fp16 = A_fp32.astype(cp.float16)
        B_fp16 = B_fp32.astype(cp.float16)
        
        # Warmup
        _ = cp.matmul(A_fp32, B_fp32)
        _ = cp.matmul(A_fp16, B_fp16)
        cp.cuda.Stream.null.synchronize()
        
        # Benchmark FP32 (CUDA cores)
        start = time.perf_counter()
        for _ in range(10):
            C_fp32 = cp.matmul(A_fp32, B_fp32)
        cp.cuda.Stream.null.synchronize()
        fp32_time = (time.perf_counter() - start) / 10
        
        # Benchmark FP16 (Tensor cores)
        start = time.perf_counter()
        for _ in range(10):
            C_fp16 = cp.matmul(A_fp16, B_fp16)
        cp.cuda.Stream.null.synchronize()
        fp16_time = (time.perf_counter() - start) / 10
        
        speedup = fp32_time / fp16_time
        
        print(f"{size:4d}×{size:<4d} | {fp32_time*1000:8.2f} ms       | "
              f"{fp16_time*1000:8.2f} ms         | {speedup:5.2f}x")
    
    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS")
    print("=" * 70)
    print("""
1. Tensor cores (FP16) are MUCH faster for matrix multiplication
2. Speedup increases with matrix size (better for large matrices)
3. Tensor cores operate on FP16/BF16/TF32 data types
4. Trade-off: Lower precision but much higher throughput
    """)
    
    print("\n" + "=" * 70)
    print("TENSOR CORE OPERATION")
    print("=" * 70)
    print("""
Tensor cores compute: D = A × B + C

For example (4×4 matrices):
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ A matrix│ × │ B matrix│ + │ C matrix│ = │ D result│
│  (4×4)  │   │  (4×4)  │   │  (4×4)  │   │  (4×4)  │
└─────────┘   └─────────┘   └─────────┘   └─────────┘

A single tensor core can compute this entire 4×4 operation
in ONE instruction! (Modern tensor cores handle larger tiles)

Compare to CUDA cores:
- Would need 64 multiply operations (4×4×4)
- Plus 48 addition operations
- Takes many more cycles
    """)
    
    print("\n" + "=" * 70)
    print("CUDA CORES vs TENSOR CORES")
    print("=" * 70)
    print("""
╔═══════════════════╦═══════════════════╦═══════════════════╗
║   Feature         ║   CUDA Cores      ║   Tensor Cores    ║
╠═══════════════════╬═══════════════════╬═══════════════════╣
║ Purpose           ║ General compute   ║ Matrix multiply   ║
║ Operations        ║ Any calculation   ║ Only: A×B+C       ║
║ Precision         ║ FP32, FP64, INT   ║ FP16, BF16, TF32  ║
║ Speed             ║ Normal            ║ 5-20x faster      ║
║ Use case          ║ General GPU code  ║ Deep learning     ║
║ Programming       ║ Write any code    ║ Use libraries     ║
╚═══════════════════╩═══════════════════╩═══════════════════╝
    """)
    
    print("\n" + "=" * 70)
    print("WHEN TO USE TENSOR CORES")
    print("=" * 70)
    print("""
✓ USE Tensor Cores for:
  - Matrix multiplication (GEMM operations)
  - Neural network training/inference
  - Deep learning workloads
  - Convolutions (implemented as matrix ops)
  - Large matrix operations where FP16 precision is okay

✗ DON'T use Tensor Cores for:
  - General computations (if/else, loops, etc.)
  - Operations requiring FP32/FP64 precision
  - Non-matrix operations
  - Small matrices (overhead not worth it)
    """)
    
    print("\n" + "=" * 70)
    print("HOW TO USE TENSOR CORES")
    print("=" * 70)
    print("""
You typically don't program tensor cores directly!
Instead, use libraries that automatically utilize them:

1. **CuBLAS** (CUDA Basic Linear Algebra)
   - Matrix operations
   - Automatically uses tensor cores for FP16

2. **CuDNN** (Deep Neural Network library)
   - Convolutions, pooling, etc.
   - Auto-detects and uses tensor cores

3. **Deep Learning Frameworks**
   - PyTorch: torch.nn uses tensor cores automatically
   - TensorFlow: Uses tensor cores for training
   - JAX: Utilizes tensor cores through XLA

4. **Manual Control**
   - Use FP16 or BF16 data types
   - Use cuBLAS or cuDNN APIs
   - Libraries handle the tensor core dispatch
    """)
    
    print("\n" + "=" * 70)
    print("DATA TYPES FOR TENSOR CORES")
    print("=" * 70)
    print("""
Different tensor core generations support different types:

1. **FP16 (Half precision)**
   - 16-bit floating point
   - Range: ±65,504
   - All tensor cores support this

2. **BF16 (Brain Float 16)**
   - 16-bit with wider range than FP16
   - Same range as FP32 but less precision
   - Better for training (Ampere+)

3. **TF32 (TensorFloat-32)**
   - 19-bit format (internal)
   - FP32 range, FP16 precision
   - Automatic in Ampere+ (no code change!)

4. **FP8 (8-bit floating point)**
   - Very fast, lower precision
   - Ada Lovelace and newer

5. **INT8/INT4**
   - Integer operations
   - Model quantization/inference
    """)
    
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)
    
    num_sms = attrs.get('MultiProcessorCount', 0)
    
    print(f"\nYour GPU (Compute {major}.{minor}):")
    print(f"  SMs: {num_sms}")
    
    if compute_cap >= 120:  # Blackwell
        print(f"  Tensor Cores per SM: ~4 (estimated)")
        print(f"  Total Tensor Cores: ~{num_sms * 4}")
        print(f"  Peak TFLOPS (FP16): ~2000+ TFLOPS")
    elif compute_cap >= 89:  # Ada
        print(f"  Tensor Cores per SM: ~4")
        print(f"  Total Tensor Cores: ~{num_sms * 4}")
        print(f"  Peak TFLOPS (FP16): ~1300 TFLOPS")
    elif compute_cap >= 80:  # Ampere
        print(f"  Tensor Cores per SM: ~4")
        print(f"  Total Tensor Cores: ~{num_sms * 4}")
        print(f"  Peak TFLOPS (FP16): ~300 TFLOPS")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Think of your GPU as having TWO types of workers:

🔧 CUDA Cores = General workers
   - Can do any task you give them
   - Handle your custom kernels from lesson 3
   - Flexible but not specialized

⚡ Tensor Cores = Specialized matrix workers
   - ONLY do matrix multiply operations
   - Do it WAY faster than CUDA cores
   - Essential for AI/ML workloads

BOTH work together in your GPU!
- CUDA cores: data prep, activation functions, etc.
- Tensor cores: matrix multiplications
- This is why modern AI training is so fast!
    """)


if __name__ == "__main__":
    main()

