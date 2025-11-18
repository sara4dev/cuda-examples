"""
Lesson 3: Your First CUDA Kernel
Goal: Understand threads, blocks, and how to launch a kernel
"""

import os
import sys

# Set up library paths for CUDA
venv_path = os.path.dirname(os.path.dirname(sys.executable))
cuda_lib_path = os.path.join(venv_path, "lib/python3.12/site-packages/nvidia")
if os.path.exists(cuda_lib_path):
    lib_dirs = [
        os.path.join(cuda_lib_path, "cuda_nvrtc/lib"),
        os.path.join(cuda_lib_path, "cuda_runtime/lib"),
    ]
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [ld_library_path])

import cupy as cp
import numpy as np


# CUDA kernel source code (written in CUDA C++)
KERNEL_SOURCE = r"""
extern "C" __global__
void simple_kernel(int *output) {
    // threadIdx.x = thread ID within the block (0, 1, 2, 3, ...)
    // blockIdx.x = block ID within the grid (0, 1, 2, ...)
    // blockDim.x = number of threads per block
    
    // Calculate global thread ID
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread writes its ID to the output array
    output[global_id] = global_id;
}
"""


def main():
    print("=" * 60)
    print("LESSON 3: Launch Your First Kernel")
    print("=" * 60)
    
    # Get GPU info
    print("\nGPU Information:")
    device = cp.cuda.Device()
    print(f"   Device ID: {device.id}")
    print(f"   Compute Capability: {device.compute_capability}")
    
    # Set up execution configuration
    num_threads_per_block = 4
    num_blocks = 3
    total_threads = num_threads_per_block * num_blocks
    
    print("\n1. Execution Configuration:")
    print(f"   Threads per block: {num_threads_per_block}")
    print(f"   Number of blocks:  {num_blocks}")
    print(f"   Total threads:     {total_threads}")
    print()
    print("   Visualization:")
    print("   Grid: [Block 0] [Block 1] [Block 2]")
    print("         ├─ Threads 0-3")
    print("         ├─ Threads 4-7")
    print("         └─ Threads 8-11")
    
    # Compile the kernel
    print("\n2. Compiling CUDA kernel...")
    kernel = cp.RawKernel(KERNEL_SOURCE, 'simple_kernel')
    print("   ✓ Kernel compiled successfully")
    
    # Allocate output memory on GPU
    print("\n3. Allocating GPU memory...")
    gpu_output = cp.zeros(total_threads, dtype=cp.int32)
    print(f"   Allocated memory for {total_threads} integers")
    
    # Launch kernel
    print("\n4. Launching kernel...")
    grid = (num_blocks,)      # Number of blocks
    block = (num_threads_per_block,)  # Threads per block
    kernel(grid, block, (gpu_output,))
    print("   Kernel launched!")
    
    # Wait for kernel to finish (CuPy handles this automatically)
    print("\n5. Waiting for kernel to complete...")
    cp.cuda.Stream.null.synchronize()
    print("   Kernel finished!")
    
    # Copy results back to CPU
    print("\n6. Copying results from GPU to CPU...")
    cpu_output = cp.asnumpy(gpu_output)
    
    # Display results
    print("\n7. Results:")
    print(f"   Output array: {cpu_output}")
    print("\n   Explanation:")
    for i in range(total_threads):
        block_id = i // num_threads_per_block
        thread_id = i % num_threads_per_block
        print(f"   output[{i}] = {cpu_output[i]} "
              f"(Block {block_id}, Thread {thread_id})")
    
    print("\n" + "=" * 60)
    print("KEY CONCEPTS:")
    print("=" * 60)
    print("""
Kernel = Function that runs on GPU
Thread = Single execution of the kernel
Block = Group of threads (they share resources)
Grid = All blocks running the kernel

Global Thread ID Formula:
  global_id = blockIdx.x * blockDim.x + threadIdx.x

Example with 3 blocks × 4 threads:
  Block 0: threads 0, 1, 2, 3
  Block 1: threads 4, 5, 6, 7
  Block 2: threads 8, 9, 10, 11

Each thread executes the SAME code but with different IDs!
    """)


if __name__ == "__main__":
    main()

