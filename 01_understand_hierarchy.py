"""
CUDA Execution Model: Understanding Threads, Blocks, Grids, and SMs

CUDA Hierarchy (from smallest to largest):
1. Thread: Single execution unit (like a single core)
2. Warp: 32 threads that execute together in lockstep (hardware level)
3. Block: Group of threads that can cooperate and share memory
4. Grid: Collection of blocks that execute a kernel

Streaming Multiprocessor (SM):
- Physical hardware unit on the GPU
- Multiple blocks can be assigned to an SM
- SMs execute warps (groups of 32 threads)
"""

from cuda.bindings import driver as cuda
from cuda.bindings import nvrtc
import numpy as np


# CUDA kernel that prints thread/block information
CUDA_SOURCE = r"""
extern "C" __global__
void print_thread_info(int *data) {
    // Thread identifiers within a block
    int threadX = threadIdx.x;  // Thread ID in X dimension
    int threadY = threadIdx.y;  // Thread ID in Y dimension
    int threadZ = threadIdx.z;  // Thread ID in Z dimension
    
    // Block identifiers within a grid
    int blockX = blockIdx.x;
    int blockY = blockIdx.y;
    int blockZ = blockIdx.z;
    
    // Block dimensions
    int blockDimX = blockDim.x;  // Number of threads per block in X
    int blockDimY = blockDim.y;
    int blockDimZ = blockDim.z;
    
    // Grid dimensions
    int gridDimX = gridDim.x;   // Number of blocks in X
    int gridDimY = gridDim.y;
    int gridDimZ = gridDim.z;
    
    // Calculate global thread ID (unique across entire grid)
    int globalThreadId = blockX * blockDimX + threadX;
    
    // Only print from a few threads to avoid overwhelming output
    if (globalThreadId < 8) {
        printf("Global Thread %d: Block(%d,%d,%d) Thread(%d,%d,%d) | BlockDim(%d,%d,%d) GridDim(%d,%d,%d)\\n",
               globalThreadId, blockX, blockY, blockZ, 
               threadX, threadY, threadZ,
               blockDimX, blockDimY, blockDimZ,
               gridDimX, gridDimY, gridDimZ);
    }
    
    // Write global thread ID to output array
    data[globalThreadId] = globalThreadId;
}

extern "C" __global__
void vector_add(float *a, float *b, float *c, int n) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread handles one element
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
        
        // Print from a few threads to show work distribution
        if (idx < 5) {
            printf("Thread %d: a[%d]=%.1f + b[%d]=%.1f = c[%d]=%.1f\\n",
                   idx, idx, a[idx], idx, b[idx], idx, c[idx]);
        }
    }
}
"""


def check_cuda_errors(result):
    """Helper to check CUDA errors"""
    if result[0].value != 0:
        raise RuntimeError(f"CUDA Error: {result[0]}")
    return result[1] if len(result) > 1 else None


def compile_cuda_kernel(source):
    """Compile CUDA source code"""
    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(
        str.encode(source), b"kernel.cu", 0, [], []
    )
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"Failed to create program: {err}")
    
    # Compile
    opts = [b'--gpu-architecture=compute_75']
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        # Get compilation log
        err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b' ' * log_size
        err, = nvrtc.nvrtcGetProgramLog(prog, log)
        raise RuntimeError(f"Compilation failed:\n{log.decode()}")
    
    # Get PTX
    err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b' ' * ptx_size
    err, = nvrtc.nvrtcGetPTX(prog, ptx)
    
    return ptx


def get_device_info():
    """Get CUDA device information"""
    err, = cuda.cuInit(0)
    check_cuda_errors((err,))
    
    err, device = cuda.cuDeviceGet(0)
    check_cuda_errors((err,))
    
    # Get device name
    err, name = cuda.cuDeviceGetName(128, device)
    name = name.decode() if isinstance(name, bytes) else name
    
    # Get number of SMs
    err, num_sms = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device
    )
    
    # Get max threads per block
    err, max_threads_per_block = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device
    )
    
    # Get max threads per SM
    err, max_threads_per_sm = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device
    )
    
    # Get warp size
    err, warp_size = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE, device
    )
    
    print("=" * 70)
    print("GPU DEVICE INFORMATION")
    print("=" * 70)
    print(f"Device Name: {name}")
    print(f"Number of SMs (Streaming Multiprocessors): {num_sms}")
    print(f"Max Threads per Block: {max_threads_per_block}")
    print(f"Max Threads per SM: {max_threads_per_sm}")
    print(f"Warp Size: {warp_size} threads")
    print(f"Max Warps per SM: {max_threads_per_sm // warp_size}")
    print("=" * 70)
    print()
    
    return device


def example1_print_thread_info():
    """Example 1: Print thread/block information"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Understanding Thread and Block IDs")
    print("=" * 70)
    print("Configuration: 2 blocks × 4 threads per block = 8 total threads")
    print("=" * 70)
    
    # Initialize CUDA
    err, = cuda.cuInit(0)
    check_cuda_errors((err,))
    
    err, device = cuda.cuDeviceGet(0)
    check_cuda_errors((err,))
    
    err, context = cuda.cuCtxCreate(None, 0, device)
    check_cuda_errors((err,))
    
    # Compile kernel
    ptx = compile_cuda_kernel(CUDA_SOURCE)
    
    # Load module
    err, module = cuda.cuModuleLoadData(ptx)
    check_cuda_errors((err,))
    
    err, kernel = cuda.cuModuleGetFunction(module, b"print_thread_info")
    check_cuda_errors((err,))
    
    # Allocate memory
    num_threads = 8
    err, d_data = cuda.cuMemAlloc(num_threads * 4)  # 4 bytes per int
    check_cuda_errors((err,))
    
    # Launch configuration
    threads_per_block = 4  # 4 threads per block
    num_blocks = 2         # 2 blocks
    
    print(f"\nLaunching kernel with:")
    print(f"  Blocks: {num_blocks}")
    print(f"  Threads per block: {threads_per_block}")
    print(f"  Total threads: {num_blocks * threads_per_block}")
    print("\nKernel output:")
    
    # Launch kernel
    args = [d_data]
    args_ptr = np.array([arg for arg in args], dtype=np.uint64)
    
    err, = cuda.cuLaunchKernel(
        kernel,
        num_blocks, 1, 1,           # Grid dimensions (blocks)
        threads_per_block, 1, 1,    # Block dimensions (threads)
        0,                           # Shared memory
        0,                           # Stream
        args_ptr.ctypes.data,        # Kernel arguments
        0                            # Extra options
    )
    check_cuda_errors((err,))
    
    # Synchronize
    err, = cuda.cuCtxSynchronize()
    check_cuda_errors((err,))
    
    # Read results
    h_data = np.zeros(num_threads, dtype=np.int32)
    err, = cuda.cuMemcpyDtoH(h_data, d_data, num_threads * 4)
    check_cuda_errors((err,))
    
    print(f"\nThread IDs written to array: {h_data}")
    
    # Cleanup
    cuda.cuMemFree(d_data)
    cuda.cuCtxDestroy(context)


def example2_vector_addition():
    """Example 2: Vector addition to demonstrate work distribution"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Vector Addition - Work Distribution Across Threads")
    print("=" * 70)
    
    # Initialize CUDA
    err, = cuda.cuInit(0)
    check_cuda_errors((err,))
    
    err, device = cuda.cuDeviceGet(0)
    check_cuda_errors((err,))
    
    err, context = cuda.cuCtxCreate(None, 0, device)
    check_cuda_errors((err,))
    
    # Compile kernel
    ptx = compile_cuda_kernel(CUDA_SOURCE)
    
    # Load module
    err, module = cuda.cuModuleLoadData(ptx)
    check_cuda_errors((err,))
    
    err, kernel = cuda.cuModuleGetFunction(module, b"vector_add")
    check_cuda_errors((err,))
    
    # Create test vectors
    n = 16
    h_a = np.arange(n, dtype=np.float32)
    h_b = np.arange(n, dtype=np.float32) * 2
    h_c = np.zeros(n, dtype=np.float32)
    
    print(f"Vector size: {n} elements")
    print(f"Vector A: {h_a[:8]}...")
    print(f"Vector B: {h_b[:8]}...")
    
    # Allocate device memory
    bytes_size = n * 4
    err, d_a = cuda.cuMemAlloc(bytes_size)
    check_cuda_errors((err,))
    err, d_b = cuda.cuMemAlloc(bytes_size)
    check_cuda_errors((err,))
    err, d_c = cuda.cuMemAlloc(bytes_size)
    check_cuda_errors((err,))
    
    # Copy to device
    err, = cuda.cuMemcpyHtoD(d_a, h_a, bytes_size)
    check_cuda_errors((err,))
    err, = cuda.cuMemcpyHtoD(d_b, h_b, bytes_size)
    check_cuda_errors((err,))
    
    # Launch configuration
    threads_per_block = 4
    num_blocks = (n + threads_per_block - 1) // threads_per_block
    
    print(f"\nLaunch configuration:")
    print(f"  Total elements: {n}")
    print(f"  Threads per block: {threads_per_block}")
    print(f"  Number of blocks: {num_blocks}")
    print(f"  Total threads launched: {num_blocks * threads_per_block}")
    print(f"\nEach thread processes ONE element of the array.")
    print("\nKernel output (first few threads):")
    
    # Launch kernel
    args = np.array([d_a, d_b, d_c, n], dtype=np.uint64)
    
    err, = cuda.cuLaunchKernel(
        kernel,
        num_blocks, 1, 1,           # Grid dimensions
        threads_per_block, 1, 1,    # Block dimensions
        0, 0,
        args.ctypes.data,
        0
    )
    check_cuda_errors((err,))
    
    # Synchronize
    err, = cuda.cuCtxSynchronize()
    check_cuda_errors((err,))
    
    # Copy result back
    err, = cuda.cuMemcpyDtoH(h_c, d_c, bytes_size)
    check_cuda_errors((err,))
    
    print(f"\nResult vector C: {h_c[:8]}...")
    print(f"Verification: All correct = {np.allclose(h_c, h_a + h_b)}")
    
    # Cleanup
    cuda.cuMemFree(d_a)
    cuda.cuMemFree(d_b)
    cuda.cuMemFree(d_c)
    cuda.cuCtxDestroy(context)


def explain_concepts():
    """Print detailed explanation of CUDA concepts"""
    print("\n" + "=" * 70)
    print("CUDA EXECUTION MODEL EXPLAINED")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────┐
│                         CUDA HIERARCHY                          │
└─────────────────────────────────────────────────────────────────┘

GRID (All blocks executing a kernel)
├─ BLOCK 0 (Can have up to 1024 threads)
│  ├─ WARP 0 (32 threads: Thread 0-31)
│  ├─ WARP 1 (32 threads: Thread 32-63)
│  └─ ...
├─ BLOCK 1
│  ├─ WARP 0
│  └─ ...
└─ BLOCK N

┌─────────────────────────────────────────────────────────────────┐
│                  KEY CONCEPTS EXPLAINED                         │
└─────────────────────────────────────────────────────────────────┘

1. THREAD
   - Smallest unit of execution
   - Has unique threadIdx (x, y, z) within its block
   - Executes the kernel code once
   - Each thread typically processes one data element

2. WARP (Hardware level)
   - Group of 32 threads
   - Threads in a warp execute in lockstep (SIMT - Single Instruction Multiple Thread)
   - The fundamental execution unit on an SM
   - If threads diverge (if/else), both paths execute serially

3. BLOCK
   - Group of threads (up to 1024 threads)
   - Has unique blockIdx (x, y, z) within the grid
   - Threads in a block can:
     * Share memory (__shared__)
     * Synchronize (__syncthreads())
   - Blocks execute independently (any order)
   - All threads in a block execute on the same SM

4. GRID
   - Collection of all blocks executing a kernel
   - Can have millions of blocks
   - gridDim specifies number of blocks

5. STREAMING MULTIPROCESSOR (SM) - Hardware
   - Physical processor on the GPU
   - Can execute multiple blocks concurrently
   - Has:
     * CUDA cores (ALUs)
     * Shared memory
     * Registers
     * Warp schedulers
   - Executes one warp at a time, but can switch between warps

┌─────────────────────────────────────────────────────────────────┐
│                  HOW THEY WORK TOGETHER                         │
└─────────────────────────────────────────────────────────────────┘

Software View:               Hardware View:
Grid (all blocks)     →      Distributed across GPU
└─ Blocks             →      Assigned to SMs
   └─ Threads         →      Execute as Warps (32 threads)

Example: Processing 10,000 elements
- Launch 100 blocks × 100 threads = 10,000 threads
- GPU with 40 SMs might assign 2-3 blocks per SM
- Each SM processes its blocks by executing warps
- Warp scheduler switches between warps to hide latency

┌─────────────────────────────────────────────────────────────────┐
│                  CALCULATING GLOBAL THREAD ID                   │
└─────────────────────────────────────────────────────────────────┘

For 1D configuration:
  globalThreadId = blockIdx.x * blockDim.x + threadIdx.x

For 2D configuration:
  x = blockIdx.x * blockDim.x + threadIdx.x
  y = blockIdx.y * blockDim.y + threadIdx.y
  globalThreadId = y * (gridDim.x * blockDim.x) + x

This maps each thread to a unique data element!
    """)


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("CUDA PROGRAMMING: Understanding SMs, Threads, Blocks, and Grids")
    print("=" * 70)
    
    # Get device info
    get_device_info()
    
    # Run examples
    example1_print_thread_info()
    example2_vector_addition()
    
    # Explain concepts
    explain_concepts()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. One thread = One piece of work (usually one array element)
2. Threads are grouped into blocks (for cooperation)
3. Blocks are grouped into grids (entire kernel)
4. SMs are hardware that execute blocks as warps
5. Always calculate global thread ID to map threads to data
6. Threads execute in warps of 32 (avoid warp divergence)
    """)


if __name__ == "__main__":
    main()

