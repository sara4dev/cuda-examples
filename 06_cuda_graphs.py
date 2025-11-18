"""
CUDA Graphs Tutorial
====================

What are CUDA Graphs?
---------------------
CUDA Graphs are a way to define a sequence of CUDA operations (kernels, memory copies, etc.)
and their dependencies as a graph structure. Instead of launching operations one by one from
the CPU, you can:
1. Define all operations and their dependencies once (graph creation)
2. Launch the entire graph as a single unit (graph execution)

Key Advantages:
---------------
1. **Lower CPU Launch Overhead**: Traditional kernel launches have CPU overhead for each launch.
   With graphs, you pay this cost once during graph creation, then subsequent launches are much faster.

2. **Better Performance**: Graph launches can be 5-10x faster than individual kernel launches
   for workloads with many small kernels.

3. **Optimization Opportunities**: The CUDA driver can see the entire workload and optimize
   across kernel boundaries.

4. **Repeatability**: Perfect for workloads that repeat the same sequence of operations
   (e.g., deep learning training loops, simulations).

When to Use CUDA Graphs:
-------------------------
✓ Multiple kernel launches in a sequence
✓ Repeated execution of the same operation pattern
✓ Performance-critical applications with many small kernels
✗ Dynamic workloads where the sequence changes each time
✗ Kernels that depend on CPU-side computations between launches
"""

import cupy as cp
import time
import numpy as np


# Simple CUDA kernel: vector addition
vector_add_kernel = cp.RawKernel(r'''
extern "C" __global__
void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
''', 'vector_add')


# Simple CUDA kernel: vector multiplication
vector_mul_kernel = cp.RawKernel(r'''
extern "C" __global__
void vector_mul(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}
''', 'vector_mul')


# Simple CUDA kernel: vector scale
vector_scale_kernel = cp.RawKernel(r'''
extern "C" __global__
void vector_scale(float* a, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= scale;
    }
}
''', 'vector_scale')


def traditional_approach(a, b, temp1, temp2, result, n, blocks, threads, iterations=1000):
    """
    Traditional approach: Launch each kernel individually from CPU
    Each launch has CPU overhead (driver calls, parameter setup, etc.)
    """
    print("\n" + "="*70)
    print("TRADITIONAL APPROACH: Individual Kernel Launches")
    print("="*70)
    print(f"Running {iterations} iterations with 3 kernels each...")
    
    # Warm-up
    for _ in range(10):
        vector_add_kernel((blocks,), (threads,), (a, b, temp1, n))
        vector_mul_kernel((blocks,), (threads,), (a, b, temp2, n))
        vector_scale_kernel((blocks,), (threads,), (temp1, 2.0, n))
    cp.cuda.Stream.null.synchronize()
    
    # Actual timing
    start_time = time.perf_counter()
    
    for i in range(iterations):
        # Three sequential operations
        vector_add_kernel((blocks,), (threads,), (a, b, temp1, n))
        vector_mul_kernel((blocks,), (threads,), (a, b, temp2, n))
        vector_scale_kernel((blocks,), (threads,), (temp1, 2.0, n))
    
    # Wait for all operations to complete
    cp.cuda.Stream.null.synchronize()
    
    end_time = time.perf_counter()
    elapsed = (end_time - start_time) * 1000  # Convert to ms
    
    print(f"✓ Total time: {elapsed:.3f} ms")
    print(f"✓ Average time per iteration: {elapsed/iterations:.4f} ms")
    print(f"✓ Total kernel launches: {iterations * 3}")
    print(f"✓ CPU overhead per launch: ~{elapsed/iterations/3:.6f} ms")
    
    return elapsed


def cuda_graph_approach(a, b, temp1, temp2, result, n, blocks, threads, iterations=1000):
    """
    CUDA Graph approach: Capture the sequence once, then replay it
    Much lower overhead for repeated execution
    """
    print("\n" + "="*70)
    print("CUDA GRAPH APPROACH: Graph Capture and Replay")
    print("="*70)
    
    # Step 1: Start capturing operations into a graph
    print("\n[Step 1] Capturing operations into a CUDA Graph...")
    stream = cp.cuda.Stream()
    
    with stream:
        # Begin graph capture
        cp.cuda.get_current_stream().begin_capture()
        
        # Record the sequence of operations
        # These don't actually execute during capture, they're just recorded
        vector_add_kernel((blocks,), (threads,), (a, b, temp1, n))
        vector_mul_kernel((blocks,), (threads,), (a, b, temp2, n))
        vector_scale_kernel((blocks,), (threads,), (temp1, 2.0, n))
        
        # End capture and get the graph
        graph = cp.cuda.get_current_stream().end_capture()
    
    print("✓ Graph captured successfully!")
    print(f"✓ Graph contains: 3 kernel operations")
    
    # Step 2: Upload the graph (prepares it for execution)
    print("\n[Step 2] Uploading graph to GPU...")
    graph.upload(stream)
    print("✓ Graph uploaded and ready for execution!")
    
    # Step 3: Execute the graph multiple times
    print(f"\n[Step 3] Executing graph {iterations} times...")
    
    # Warm-up
    for _ in range(10):
        graph.launch(stream)
    stream.synchronize()
    
    # Actual timing
    start_time = time.perf_counter()
    
    for i in range(iterations):
        # Single launch executes all 3 kernels!
        graph.launch(stream)
    
    stream.synchronize()
    
    end_time = time.perf_counter()
    elapsed = (end_time - start_time) * 1000  # Convert to ms
    
    print(f"✓ Total time: {elapsed:.3f} ms")
    print(f"✓ Average time per iteration: {elapsed/iterations:.4f} ms")
    print(f"✓ Total graph launches: {iterations}")
    print(f"✓ CPU overhead per graph launch: ~{elapsed/iterations:.6f} ms")
    
    return elapsed


def main():
    print("\n" + "="*70)
    print(" CUDA GRAPHS TUTORIAL ".center(70, "="))
    print("="*70)
    
    # Setup
    n = 1024 * 1024  # 1M elements
    threads = 256
    blocks = (n + threads - 1) // threads
    
    print(f"\nProblem size: {n:,} elements")
    print(f"Grid configuration: {blocks} blocks × {threads} threads")
    print(f"\nWorkload: 3 kernels (add, multiply, scale) repeated many times")
    
    # Allocate memory
    print("\nAllocating GPU memory...")
    a = cp.random.rand(n, dtype=cp.float32)
    b = cp.random.rand(n, dtype=cp.float32)
    temp1 = cp.empty(n, dtype=cp.float32)
    temp2 = cp.empty(n, dtype=cp.float32)
    result = cp.empty(n, dtype=cp.float32)
    print("✓ Memory allocated")
    
    iterations = 5000  # More iterations to see the difference
    
    # Run traditional approach
    traditional_time = traditional_approach(a, b, temp1, temp2, result, n, blocks, threads, iterations)
    
    # Run CUDA Graph approach
    graph_time = cuda_graph_approach(a, b, temp1, temp2, result, n, blocks, threads, iterations)
    
    # Summary
    print("\n" + "="*70)
    print(" PERFORMANCE COMPARISON ".center(70, "="))
    print("="*70)
    print(f"\nTraditional approach: {traditional_time:.3f} ms")
    print(f"CUDA Graph approach:  {graph_time:.3f} ms")
    print(f"\nSpeedup: {traditional_time/graph_time:.2f}x faster! 🚀")
    print(f"Time saved: {traditional_time - graph_time:.3f} ms ({(1-graph_time/traditional_time)*100:.1f}% reduction)")
    
    # Calculate per-kernel overhead saved
    traditional_per_kernel = traditional_time / (iterations * 3)
    graph_per_kernel = graph_time / (iterations * 3)
    print(f"\nPer-kernel overhead:")
    print(f"  Traditional: {traditional_per_kernel*1000:.3f} μs")
    print(f"  Graph:       {graph_per_kernel*1000:.3f} μs")
    print(f"  Saved:       {(traditional_per_kernel - graph_per_kernel)*1000:.3f} μs per kernel")
    
    print("\n" + "="*70)
    print(" KEY TAKEAWAYS ".center(70, "="))
    print("="*70)
    print("""
1. CUDA Graphs reduce CPU launch overhead dramatically
   → Perfect for workloads with many repeated kernel launches

2. The more kernels you have, the bigger the benefit
   → We saw 3 kernels, but imagine 100+ kernels in a neural network!

3. Graph creation has one-time cost, but execution is much faster
   → Amortized over many iterations in training loops

4. Use cases:
   ✓ Deep learning training (same forward/backward pass repeated)
   ✓ Simulations with fixed computation patterns
   ✓ Signal processing pipelines
   ✓ Any repeated sequence of GPU operations

5. Limitations to remember:
   ✗ Cannot change kernel parameters between graph launches
   ✗ Cannot have CPU computations in the middle of the graph
   ✗ Dynamic control flow is limited (though possible with conditional nodes)
""")
    
    print("="*70)


if __name__ == "__main__":
    main()

