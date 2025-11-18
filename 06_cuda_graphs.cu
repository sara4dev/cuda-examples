/**
 * CUDA Graphs Tutorial - Native CUDA C++ Version
 * ==============================================
 * 
 * This demonstrates CUDA Graphs using the native CUDA C++ API.
 * Compare with 06_cuda_graphs.py to see the Python equivalent.
 * 
 * Compile with:
 *   nvcc -o cuda_graphs 06_cuda_graphs.cu -O3
 * 
 * Run with:
 *   ./cuda_graphs
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Simple kernel: Vector addition
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Simple kernel: Vector multiplication
__global__ void vectorMul(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// Simple kernel: Vector scaling
__global__ void vectorScale(float* a, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= scale;
    }
}

// Initialize array with random values
void initArray(float* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Traditional approach: Launch each kernel individually
double traditionalApproach(float* d_a, float* d_b, float* d_temp1, float* d_temp2,
                          int n, int blocks, int threads, int iterations) {
    printf("\n======================================================================\n");
    printf("TRADITIONAL APPROACH: Individual Kernel Launches\n");
    printf("======================================================================\n");
    printf("Running %d iterations with 3 kernels each...\n", iterations);
    
    // Warm-up
    for (int i = 0; i < 10; i++) {
        vectorAdd<<<blocks, threads>>>(d_a, d_b, d_temp1, n);
        vectorMul<<<blocks, threads>>>(d_a, d_b, d_temp2, n);
        vectorScale<<<blocks, threads>>>(d_temp1, 2.0f, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        vectorAdd<<<blocks, threads>>>(d_a, d_b, d_temp1, n);
        vectorMul<<<blocks, threads>>>(d_a, d_b, d_temp2, n);
        vectorScale<<<blocks, threads>>>(d_temp1, 2.0f, n);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("✓ Total time: %.3f ms\n", elapsed);
    printf("✓ Average time per iteration: %.4f ms\n", elapsed / iterations);
    printf("✓ Total kernel launches: %d\n", iterations * 3);
    printf("✓ CPU overhead per launch: ~%.6f ms\n", elapsed / iterations / 3);
    
    return elapsed;
}

// CUDA Graph approach: Capture once, replay many times
double cudaGraphApproach(float* d_a, float* d_b, float* d_temp1, float* d_temp2,
                        int n, int blocks, int threads, int iterations) {
    printf("\n======================================================================\n");
    printf("CUDA GRAPH APPROACH: Graph Capture and Replay\n");
    printf("======================================================================\n");
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Step 1: Capture the sequence of operations into a graph
    printf("\n[Step 1] Capturing operations into a CUDA Graph...\n");
    
    cudaGraph_t graph;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    
    // Record the operations (they don't execute during capture)
    vectorAdd<<<blocks, threads, 0, stream>>>(d_a, d_b, d_temp1, n);
    vectorMul<<<blocks, threads, 0, stream>>>(d_a, d_b, d_temp2, n);
    vectorScale<<<blocks, threads, 0, stream>>>(d_temp1, 2.0f, n);
    
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    printf("✓ Graph captured successfully!\n");
    printf("✓ Graph contains: 3 kernel operations\n");
    
    // Step 2: Instantiate the graph into an executable form
    printf("\n[Step 2] Creating executable graph instance...\n");
    cudaGraphExec_t graphExec;
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    printf("✓ Graph instantiated and ready for execution!\n");
    
    // Step 3: Execute the graph many times
    printf("\n[Step 3] Executing graph %d times...\n", iterations);
    
    // Warm-up
    for (int i = 0; i < 10; i++) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Timing
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        // Single launch executes all 3 kernels!
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("✓ Total time: %.3f ms\n", elapsed);
    printf("✓ Average time per iteration: %.4f ms\n", elapsed / iterations);
    printf("✓ Total graph launches: %d\n", iterations);
    printf("✓ CPU overhead per graph launch: ~%.6f ms\n", elapsed / iterations);
    
    // Cleanup
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    return elapsed;
}

int main() {
    printf("\n======================================================================\n");
    printf("                    CUDA GRAPHS TUTORIAL                             \n");
    printf("======================================================================\n");
    
    // Configuration
    const int n = 1024 * 1024;  // 1M elements
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int iterations = 5000;
    
    printf("\nProblem size: %d elements\n", n);
    printf("Grid configuration: %d blocks × %d threads\n", blocks, threads);
    printf("\nWorkload: 3 kernels (add, multiply, scale) repeated many times\n");
    
    // Allocate host memory
    printf("\nAllocating memory...\n");
    size_t bytes = n * sizeof(float);
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    
    // Initialize arrays
    srand(12345);
    initArray(h_a, n);
    initArray(h_b, n);
    
    // Allocate device memory
    float *d_a, *d_b, *d_temp1, *d_temp2;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_temp1, bytes));
    CUDA_CHECK(cudaMalloc(&d_temp2, bytes));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    printf("✓ Memory allocated and data copied to GPU\n");
    
    // Run both approaches
    double traditional_time = traditionalApproach(d_a, d_b, d_temp1, d_temp2, 
                                                  n, blocks, threads, iterations);
    
    double graph_time = cudaGraphApproach(d_a, d_b, d_temp1, d_temp2, 
                                         n, blocks, threads, iterations);
    
    // Summary
    printf("\n======================================================================\n");
    printf("                    PERFORMANCE COMPARISON                           \n");
    printf("======================================================================\n");
    printf("\nTraditional approach: %.3f ms\n", traditional_time);
    printf("CUDA Graph approach:  %.3f ms\n", graph_time);
    printf("\nSpeedup: %.2fx faster! 🚀\n", traditional_time / graph_time);
    printf("Time saved: %.3f ms (%.1f%% reduction)\n", 
           traditional_time - graph_time, 
           (1 - graph_time / traditional_time) * 100);
    
    double traditional_per_kernel = traditional_time / (iterations * 3);
    double graph_per_kernel = graph_time / (iterations * 3);
    printf("\nPer-kernel overhead:\n");
    printf("  Traditional: %.3f μs\n", traditional_per_kernel * 1000);
    printf("  Graph:       %.3f μs\n", graph_per_kernel * 1000);
    printf("  Saved:       %.3f μs per kernel\n", 
           (traditional_per_kernel - graph_per_kernel) * 1000);
    
    printf("\n======================================================================\n");
    printf("                        KEY TAKEAWAYS                                \n");
    printf("======================================================================\n");
    printf("\n1. CUDA Graphs reduce CPU launch overhead dramatically\n");
    printf("   → Perfect for workloads with many repeated kernel launches\n\n");
    printf("2. The more kernels you have, the bigger the benefit\n");
    printf("   → We saw 3 kernels, but imagine 100+ kernels in a neural network!\n\n");
    printf("3. Graph creation has one-time cost, but execution is much faster\n");
    printf("   → Amortized over many iterations in training loops\n\n");
    printf("4. API Flow:\n");
    printf("   cudaStreamBeginCapture()  → Start recording\n");
    printf("   Launch kernels...         → Record operations\n");
    printf("   cudaStreamEndCapture()    → Stop recording, get graph\n");
    printf("   cudaGraphInstantiate()    → Create executable\n");
    printf("   cudaGraphLaunch()         → Execute graph (fast!)\n\n");
    printf("5. Use cases:\n");
    printf("   ✓ Deep learning training (same forward/backward pass repeated)\n");
    printf("   ✓ Simulations with fixed computation patterns\n");
    printf("   ✓ Signal processing pipelines\n");
    printf("   ✓ Any repeated sequence of GPU operations\n\n");
    
    printf("======================================================================\n\n");
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_temp1));
    CUDA_CHECK(cudaFree(d_temp2));
    free(h_a);
    free(h_b);
    
    return 0;
}


