# CUDA Graphs: Complete Guide

## Table of Contents
1. [What Are CUDA Graphs?](#what-are-cuda-graphs)
2. [Why Use CUDA Graphs?](#why-use-cuda-graphs)
3. [How CUDA Graphs Work](#how-cuda-graphs-work)
4. [Performance Benefits](#performance-benefits)
5. [Creating CUDA Graphs](#creating-cuda-graphs)
6. [Advanced Topics](#advanced-topics)
7. [Best Practices](#best-practices)
8. [Common Pitfalls](#common-pitfalls)

---

## What Are CUDA Graphs?

CUDA Graphs are a feature introduced in CUDA 10.0 that allows you to define a sequence of CUDA operations (kernels, memory copies, etc.) and their dependencies as a **graph data structure**. Instead of launching operations one at a time from the CPU, you can:

1. **Define** all operations and their dependencies once (graph creation)
2. **Launch** the entire graph as a single unit (graph execution)
3. **Reuse** the graph for repeated executions with minimal overhead

### Traditional Workflow vs CUDA Graphs

**Traditional approach:**
```
CPU                          GPU
 |                            |
 |-- Launch Kernel 1 -------->|
 |                            |- Execute Kernel 1
 |<--- Kernel 1 Complete -----|
 |-- Launch Kernel 2 -------->|
 |                            |- Execute Kernel 2
 |<--- Kernel 2 Complete -----|
 |-- Launch Kernel 3 -------->|
 |                            |- Execute Kernel 3
 |<--- Kernel 3 Complete -----|
```

Each kernel launch involves:
- CPU-side driver overhead
- Parameter marshaling
- Queue submission
- Synchronization checks

**CUDA Graph approach:**
```
CPU                          GPU
 |                            |
 |-- Launch Graph ----------->|
 |                            |- Execute Kernel 1
 |                            |- Execute Kernel 2
 |                            |- Execute Kernel 3
 |<--- All Complete ----------|
```

One graph launch executes all operations with **minimal CPU overhead**.

---

## Why Use CUDA Graphs?

### 1. **Reduced CPU Launch Overhead**

The primary benefit! Traditional kernel launches have significant CPU overhead:
- Driver API calls
- Parameter validation
- Queue management
- Synchronization checks

With graphs, you pay this cost **once** during graph creation, then subsequent launches are **5-10x faster**.

### 2. **Better GPU Utilization**

CUDA graphs enable:
- **Parallel execution**: Operations without dependencies can run concurrently
- **Optimized scheduling**: The driver can see the entire workload upfront
- **Reduced idle time**: Less time waiting for CPU to issue the next command

### 3. **Optimization Opportunities**

The CUDA driver can optimize across kernel boundaries:
- Merge operations when possible
- Reorder independent operations
- Eliminate redundant synchronizations
- Better resource allocation

### 4. **Perfect for Repeated Workloads**

Common scenarios:
- **Deep Learning**: Same forward/backward pass repeated for thousands of iterations
- **Simulations**: Same timestep computation repeated
- **Signal Processing**: Same pipeline applied to multiple data batches
- **Iterative Solvers**: Same operations in each iteration

---

## How CUDA Graphs Work

### Core Concepts

1. **Graph**: A directed acyclic graph (DAG) of operations and their dependencies
2. **Node**: A single operation (kernel, memcpy, memset, etc.)
3. **Edge**: Dependency between nodes (A must complete before B starts)
4. **Graph Executable**: Optimized, ready-to-launch version of a graph

### The CUDA Graph Lifecycle

```
┌─────────────────┐
│  1. Creation    │  Define nodes and dependencies
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Instantiation│  Optimize and create executable
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Execution   │◄─── Launch repeatedly (fast!)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Update      │  Modify parameters (optional)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. Destruction │  Clean up resources
└─────────────────┘
```

---

## Performance Benefits

### Real-World Speedups

| Workload Type | Typical Speedup | Notes |
|---------------|-----------------|-------|
| Many small kernels | 5-10x | Maximum benefit |
| Medium kernels | 2-4x | Good benefit |
| Few large kernels | 1.2-2x | Modest benefit |
| Single kernel | ~1x | No benefit |

### Where the Time Savings Come From

**Traditional launch overhead breakdown** (approximate):
```
Total per launch: ~5-20 μs
├─ Driver API call: 2-5 μs
├─ Parameter setup: 1-3 μs
├─ Queue submission: 1-5 μs
└─ Synchronization: 1-7 μs
```

**Graph launch overhead**: ~1-3 μs (total for entire graph!)

For a workload with 100 kernel launches:
- Traditional: 100 × 10 μs = **1000 μs overhead**
- Graph: 1 × 2 μs = **2 μs overhead**
- **Savings: 99.8% reduction in launch overhead!**

### Example: Deep Learning Training Loop

A typical neural network forward pass might have:
- 50-200 kernel launches
- Repeated for thousands of iterations

**Without graphs**: 
- 100 kernels × 10 μs overhead = 1 ms overhead per iteration
- 10,000 iterations = **10 seconds wasted on launch overhead**

**With graphs**:
- 1 graph launch × 2 μs = 2 μs overhead per iteration
- 10,000 iterations = **20 ms overhead**
- **Savings: 9.98 seconds! (99.8% reduction)**

---

## Creating CUDA Graphs

### Method 1: Stream Capture (Recommended)

The easiest and most flexible method. Record operations happening on a stream:

**C++ API:**
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// Start capturing
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// Launch operations on the stream (they get recorded, not executed)
kernel1<<<grid, block, 0, stream>>>(...);
kernel2<<<grid, block, 0, stream>>>(...);
cudaMemcpyAsync(..., stream);

// End capture
cudaGraph_t graph;
cudaStreamEndCapture(stream, &graph);

// Create executable
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// Execute graph (can repeat many times)
cudaGraphLaunch(graphExec, stream);
cudaStreamSynchronize(stream);

// Cleanup
cudaGraphExecDestroy(graphExec);
cudaGraphDestroy(graph);
```

**Python (CuPy) API:**
```python
import cupy as cp

stream = cp.cuda.Stream()

with stream:
    # Start capturing
    cp.cuda.get_current_stream().begin_capture()
    
    # Launch operations (they get recorded)
    kernel1((grid,), (block,), (args...))
    kernel2((grid,), (block,), (args...))
    
    # End capture
    graph = cp.cuda.get_current_stream().end_capture()

# Create executable
graph_exec = graph.instantiate()

# Execute graph
graph_exec.launch(stream)
stream.synchronize()
```

### Method 2: Manual Graph Construction

More verbose but gives fine-grained control:

```cpp
// Create empty graph
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// Add kernel node
cudaKernelNodeParams kernelParams = {0};
kernelParams.func = (void*)myKernel;
kernelParams.gridDim = dim3(blocks);
kernelParams.blockDim = dim3(threads);
kernelParams.kernelParams = args;

cudaGraphNode_t kernelNode;
cudaGraphAddKernelNode(&kernelNode, graph, NULL, 0, &kernelParams);

// Add memcpy node (depends on kernel)
cudaMemcpy3DParms memcpyParams = {0};
// ... set parameters ...

cudaGraphNode_t memcpyNode;
cudaGraphNode_t deps[] = {kernelNode};
cudaGraphAddMemcpyNode(&memcpyNode, graph, deps, 1, &memcpyParams);

// Instantiate and execute
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
cudaGraphLaunch(graphExec, stream);
```

### Method 3: Graph Cloning

Create variations of existing graphs:

```cpp
// Clone a graph
cudaGraph_t clonedGraph;
cudaGraphClone(&clonedGraph, originalGraph);

// Modify the cloned graph
// ... update nodes ...

// Instantiate the modified graph
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, clonedGraph, NULL, NULL, 0);
```

---

## Advanced Topics

### 1. Updating Graph Parameters

Instead of recreating graphs, you can update parameters:

```cpp
// Create and instantiate graph
cudaGraph_t graph;
cudaGraphExec_t graphExec;
// ... capture and instantiate ...

// Update kernel parameters
cudaKernelNodeParams params;
cudaGraphExecKernelNodeGetParams(graphExec, node, &params);

// Modify parameters
params.kernelParams[0] = newValue;

// Update the graph executable
cudaGraphExecKernelNodeSetParams(graphExec, node, &params);

// Launch with updated parameters
cudaGraphLaunch(graphExec, stream);
```

This is useful when most of the graph stays the same but some parameters change.

### 2. Conditional Nodes (CUDA 11.0+)

Execute different paths based on runtime conditions:

```cpp
// Create conditional handle
cudaGraphConditionalHandle handle;

// Add conditional node
cudaGraphAddGraphNode(...);  // Default path
cudaGraphAddConditionalNode(...);  // Conditional path

// At runtime, the condition determines which path executes
```

### 3. Child Graphs

Nest graphs for modularity:

```cpp
// Create child graph
cudaGraph_t childGraph;
// ... populate child graph ...

// Add child as a node in parent graph
cudaGraphNode_t childNode;
cudaGraphAddChildGraphNode(&childNode, parentGraph, deps, numDeps, childGraph);
```

### 4. Graph Debugging

CUDA provides tools to visualize and debug graphs:

```cpp
// Export graph to DOT format
cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose);

// View with Graphviz:
// dot -Tpng graph.dot -o graph.png
```

### 5. Multiple Streams in Graphs

Capture operations across multiple streams:

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Begin capture on main stream
cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);

// Operations on stream1
kernel1<<<grid, block, 0, stream1>>>(...);

// Fork to stream2
cudaEventRecord(event, stream1);
cudaStreamWaitEvent(stream2, event);

// Parallel operations
kernel2<<<grid, block, 0, stream1>>>(...);  // Parallel
kernel3<<<grid, block, 0, stream2>>>(...);  // Parallel

// Join streams
cudaEventRecord(event2, stream2);
cudaStreamWaitEvent(stream1, event2);

// Continue on stream1
kernel4<<<grid, block, 0, stream1>>>(...);

cudaStreamEndCapture(stream1, &graph);
```

---

## Best Practices

### 1. When to Use CUDA Graphs

✅ **Good use cases:**
- Repeated execution of the same operation sequence
- Many small kernel launches
- Performance-critical hot loops
- Fixed computation patterns (e.g., neural network layers)
- Pipelines with clear dependencies

❌ **Poor use cases:**
- Single execution (no amortization of graph creation cost)
- Highly dynamic workloads where the sequence changes
- Operations requiring CPU computations in between
- Simple workloads with 1-2 large kernels

### 2. Graph Creation Strategy

**Option A: Create once, use forever**
```cpp
// At initialization
cudaGraph_t graph = createMyGraph();
cudaGraphExec_t exec = instantiate(graph);

// In training/simulation loop (repeat many times)
for (int iter = 0; iter < 10000; iter++) {
    cudaGraphLaunch(exec, stream);  // Fast!
}
```

**Option B: Lazy instantiation**
```cpp
// First call: create and cache
cudaGraphExec_t getGraphExec() {
    static cudaGraphExec_t cached = nullptr;
    if (cached == nullptr) {
        cudaGraph_t graph = createMyGraph();
        cudaGraphInstantiate(&cached, graph, ...);
    }
    return cached;
}
```

### 3. Memory Considerations

- Graphs capture **pointers**, not values
- The data at those pointers can change between launches
- Useful for input/output buffers that change each iteration

```cpp
// Graph captures the pointer d_input, not its contents
kernel<<<grid, block, 0, stream>>>(d_input, d_output);

// You can change what d_input points to (or its contents) between launches
cudaMemcpy(d_input, new_data, size, cudaMemcpyHostToDevice);
cudaGraphLaunch(exec, stream);  // Uses new data!
```

### 4. Warm-up Runs

Always do warm-up runs before timing:

```cpp
// Warm-up (first launch may be slower)
for (int i = 0; i < 10; i++) {
    cudaGraphLaunch(exec, stream);
}
cudaStreamSynchronize(stream);

// Now time actual runs
auto start = high_resolution_clock::now();
for (int i = 0; i < 1000; i++) {
    cudaGraphLaunch(exec, stream);
}
cudaStreamSynchronize(stream);
auto end = high_resolution_clock::now();
```

### 5. Error Checking

Check graph-related errors:

```cpp
cudaGraphExec_t exec;
cudaGraphNode_t errorNode;
char errorMsg[256];

cudaError_t err = cudaGraphInstantiate(&exec, graph, &errorNode, errorMsg, 256);
if (err != cudaSuccess) {
    fprintf(stderr, "Graph instantiation failed at node %p: %s\n", 
            errorNode, errorMsg);
    // Handle error
}
```

---

## Common Pitfalls

### 1. ❌ Capturing Host Functions

**Problem:**
```cpp
cudaStreamBeginCapture(stream, ...);
kernel1<<<grid, block, 0, stream>>>(...);
someHostFunction();  // ❌ NOT captured!
kernel2<<<grid, block, 0, stream>>>(...);
cudaStreamEndCapture(stream, &graph);
```

**Solution:** Only GPU operations are captured. Host code runs immediately.

### 2. ❌ Changing Graph Structure

**Problem:**
```cpp
// Create graph for 3 kernels
cudaGraphExec_t exec = instantiate(graph);

// Later: try to add a 4th kernel ❌
// Can't change graph structure after instantiation!
```

**Solution:** Recreate the graph or use conditional nodes for dynamic behavior.

### 3. ❌ Synchronization During Capture

**Problem:**
```cpp
cudaStreamBeginCapture(stream, ...);
kernel1<<<grid, block, 0, stream>>>(...);
cudaStreamSynchronize(stream);  // ❌ Error!
cudaStreamEndCapture(stream, &graph);
```

**Solution:** No synchronization during capture. The graph defines dependencies.

### 4. ❌ Capturing Wrong Stream

**Problem:**
```cpp
cudaStream_t stream1, stream2;
cudaStreamBeginCapture(stream1, ...);
kernel1<<<grid, block, 0, stream2>>>(...);  // ❌ Wrong stream!
cudaStreamEndCapture(stream1, &graph);
```

**Solution:** All operations must use the captured stream (or forked streams).

### 5. ❌ Memory Allocation During Capture

**Problem:**
```cpp
cudaStreamBeginCapture(stream, ...);
cudaMalloc(&ptr, size);  // ❌ Can't allocate during capture!
cudaStreamEndCapture(stream, &graph);
```

**Solution:** Allocate memory before graph capture.

### 6. ❌ Not Handling Graph Creation Overhead

**Problem:**
```cpp
// In a tight loop:
for (int i = 0; i < 1000; i++) {
    cudaGraph_t graph = captureGraph();  // ❌ Creating graph every time!
    cudaGraphExec_t exec = instantiate(graph);
    cudaGraphLaunch(exec, stream);
}
```

**Solution:** Create graph once, reuse many times:
```cpp
cudaGraph_t graph = captureGraph();  // ✓ Once
cudaGraphExec_t exec = instantiate(graph);
for (int i = 0; i < 1000; i++) {
    cudaGraphLaunch(exec, stream);  // ✓ Fast repeated launches
}
```

---

## Performance Tuning Tips

### 1. Minimize Graph Creation

- Create graphs at initialization, not in hot loops
- Cache graph executables for reuse
- Consider graph pooling for dynamic scenarios

### 2. Batch Operations

If you have many tiny kernels, consider:
- Fusing them into larger kernels
- Using CUDA graphs to reduce launch overhead
- Both: fused kernels + graphs = maximum performance

### 3. Measure Everything

```cpp
// Measure graph creation overhead
auto t1 = now();
cudaGraph_t graph = createGraph();
auto t2 = now();
printf("Graph creation: %.3f ms\n", elapsed(t1, t2));

// Measure instantiation overhead
auto t3 = now();
cudaGraphExec_t exec = instantiate(graph);
auto t4 = now();
printf("Instantiation: %.3f ms\n", elapsed(t3, t4));

// Measure launch overhead
auto t5 = now();
for (int i = 0; i < 1000; i++) {
    cudaGraphLaunch(exec, stream);
}
cudaStreamSynchronize(stream);
auto t6 = now();
printf("1000 launches: %.3f ms (%.4f ms each)\n", 
       elapsed(t5, t6), elapsed(t5, t6) / 1000);
```

### 4. Profile with Nsight

Use NVIDIA Nsight Systems to visualize:
- Graph creation vs. execution time
- CPU overhead reduction
- GPU utilization improvements

```bash
nsys profile -o profile ./my_cuda_program
```

---

## Real-World Example: Deep Learning

### Without Graphs
```cpp
// Forward pass (repeated 10,000 times in training)
for (int iter = 0; iter < 10000; iter++) {
    // Layer 1
    matmul<<<grid, block>>>(input, weights1, hidden1);
    relu<<<grid, block>>>(hidden1);
    
    // Layer 2
    matmul<<<grid, block>>>(hidden1, weights2, hidden2);
    relu<<<grid, block>>>(hidden2);
    
    // Layer 3
    matmul<<<grid, block>>>(hidden2, weights3, output);
    softmax<<<grid, block>>>(output);
    
    // Backward pass
    // ... 20 more kernel launches ...
}
// Total: 10,000 iterations × 30 kernels = 300,000 kernel launches!
// At 10 μs overhead each = 3 seconds wasted
```

### With Graphs
```cpp
// Create graph once
cudaGraph_t graph;
cudaStreamBeginCapture(stream, ...);
// Forward pass
matmul<<<grid, block, 0, stream>>>(input, weights1, hidden1);
relu<<<grid, block, 0, stream>>>(hidden1);
matmul<<<grid, block, 0, stream>>>(hidden1, weights2, hidden2);
relu<<<grid, block, 0, stream>>>(hidden2);
matmul<<<grid, block, 0, stream>>>(hidden2, weights3, output);
softmax<<<grid, block, 0, stream>>>(output);
// Backward pass
// ... 20 more kernels ...
cudaStreamEndCapture(stream, &graph);
cudaGraphExec_t exec = instantiate(graph);

// Training loop
for (int iter = 0; iter < 10000; iter++) {
    cudaGraphLaunch(exec, stream);  // One launch for all 30 kernels!
}
// Total: 10,000 graph launches
// At 2 μs overhead each = 20 ms overhead
// Savings: 2.98 seconds! (99.3% reduction)
```

---

## Summary

### Key Takeaways

1. **What**: CUDA Graphs capture a sequence of GPU operations as a reusable unit
2. **Why**: Dramatically reduces CPU launch overhead (5-10x speedup possible)
3. **When**: Best for repeated execution of fixed operation sequences
4. **How**: Use stream capture API for easiest implementation

### Decision Tree

```
Do you have a repeated sequence of GPU operations?
├─ NO → Don't use graphs
└─ YES
    └─ Do you have many (5+) kernel launches per sequence?
        ├─ NO → Marginal benefit, probably not worth it
        └─ YES
            └─ Does the sequence change dynamically?
                ├─ YES → Use conditional nodes or recreate graphs
                └─ NO → Perfect use case! Use CUDA Graphs!
```

### Next Steps

1. **Try the examples**: Run `06_cuda_graphs.py` and `06_cuda_graphs.cu`
2. **Profile your code**: Identify repeated kernel sequences
3. **Implement graphs**: Start with stream capture API
4. **Measure improvements**: Compare before/after performance
5. **Iterate**: Fine-tune based on profiling results

---

## Resources

- [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [CUDA Graphs Blog Post](https://developer.nvidia.com/blog/cuda-graphs/)
- [GTC Presentation on CUDA Graphs](https://developer.nvidia.com/gtc/2020/video/s21833)
- Example code: `06_cuda_graphs.py` and `06_cuda_graphs.cu` in this repository

---

*Happy optimizing! 🚀*


