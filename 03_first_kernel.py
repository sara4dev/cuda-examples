"""
Lesson 3: Your First CUDA Kernel
Goal: Understand threads, blocks, and how to launch a kernel

Note: If you get library errors, run with: ./run_lesson.sh 03_first_kernel.py
"""

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
    num_threads_per_block = 32
    num_blocks = 10000
    total_threads = num_threads_per_block * num_blocks
    
    print("\n1. Execution Configuration:")
    print(f"   Threads per block: {num_threads_per_block}")
    print(f"   Number of blocks:  {num_blocks}")
    print(f"   Total threads:     {total_threads:,}")
    
    # Get SM count and calculate occupancy
    attrs = device.attributes
    num_sms = attrs.get('MultiProcessorCount', 'Unknown')
    max_threads_per_sm = attrs.get('MaxThreadsPerMultiProcessor', 'Unknown')
    
    print(f"\n   GPU Hardware Limits:")
    print(f"   Streaming Multiprocessors (SMs): {num_sms}")
    print(f"   Max concurrent threads per SM:   {max_threads_per_sm:,}")
    if num_sms != 'Unknown' and max_threads_per_sm != 'Unknown':
        max_concurrent = num_sms * max_threads_per_sm
        print(f"   Max concurrent threads (total):  {max_concurrent:,}")
        print()
        print(f"   ⚡ You launched {total_threads:,} threads")
        print(f"   ⚡ GPU can run {max_concurrent:,} concurrently")
        if total_threads > max_concurrent:
            waves = (num_blocks + num_sms - 1) // num_sms
            print(f"   ⚡ Blocks will execute in ~{waves} waves")
    
    print()
    print("   WARP ALLOCATION (32 threads = 1 warp):")
    warps_per_block = (num_threads_per_block + 31) // 32
    print(f"   {num_threads_per_block} threads/block = {warps_per_block} warp(s)")
    if num_threads_per_block % 32 == 0:
        print(f"   ✓ Perfect! All warps fully utilized")
    else:
        partial = num_threads_per_block % 32
        wasted = 32 - partial
        print(f"   ⚠ Last warp only uses {partial}/32 lanes ({wasted} wasted)")
    
    print()
    print("   BLOCK SCHEDULING VISUALIZATION:")
    print("   ┌─────────────────────────────────────────────┐")
    print(f"   │ GPU has {num_sms} SMs (Streaming Multiprocessors)│")
    print("   ├─────────────────────────────────────────────┤")
    if num_blocks > num_sms:
        print(f"   │ Wave 1: Blocks 0-{num_sms-1} (running)        │")
        if num_blocks > 2*num_sms:
            print(f"   │ Wave 2: Blocks {num_sms}-{2*num_sms-1} (queued)       │")
            print("   │ Wave 3: ...                             │")
        print(f"   │ Last wave: Block {num_blocks-1} (queued)         │")
    else:
        print(f"   │ All {num_blocks} blocks fit in 1 wave!          │")
    print("   └─────────────────────────────────────────────┘")
    print("   As blocks finish, new ones start automatically!")
    
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
CUDA EXECUTION HIERARCHY:
┌────────────────────────────────────────────────────┐
│ Grid (all blocks running your kernel)              │
│  ├─ Block 0                                        │
│  │   ├─ Warp 0: Threads 0-31   (execute together) │
│  │   ├─ Warp 1: Threads 32-63  (execute together) │
│  │   └─ ...                                        │
│  ├─ Block 1                                        │
│  └─ Block N                                        │
└────────────────────────────────────────────────────┘

IMPORTANT DISTINCTIONS:

1. WARP SIZE = 32 threads
   - Hardware executes threads in groups of 32
   - This is NOT a limit on threads per block!
   - You can have 64, 128, 256, 512, 1024 threads/block
   - Non-multiples of 32 waste some warp lanes

2. CONCURRENT vs TOTAL threads:
   - RTX 5090: ~348,000 threads can run CONCURRENTLY
   - But you can LAUNCH millions or billions of threads!
   - GPU scheduler runs them in waves automatically

3. BLOCKS vs SMs:
   - You can launch 10,000+ blocks
   - GPU might have ~170 SMs
   - Blocks are queued and scheduled automatically
   - As blocks finish, new ones start

Global Thread ID Formula:
  global_id = blockIdx.x * blockDim.x + threadIdx.x

BEST PRACTICES:
✓ Use multiples of 32 for threads per block (32, 64, 128, 256)
✓ Launch enough blocks to keep GPU busy (>> number of SMs)
✓ Think about total problem size, not hardware limits
✗ Don't worry about exceeding concurrent thread limits
    """)


if __name__ == "__main__":
    main()

