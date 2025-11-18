"""
Lesson 1: CUDA Device Information
Goal: Understand what a GPU device is and query its properties
"""

from cuda.bindings import driver as cuda


def main():
    print("=" * 60)
    print("LESSON 1: Query GPU Device Information")
    print("=" * 60)
    
    # Step 1: Initialize CUDA driver API
    # This MUST be called before any other CUDA functions
    print("\n1. Initializing CUDA...")
    err, = cuda.cuInit(0)  # 0 = flags (always 0)
    print(f"   cuInit() returned: {err} (0 = success)")
    
    if err != 0:
        print("   ERROR: Failed to initialize CUDA!")
        return
    
    # Step 2: Get number of available GPUs
    print("\n2. How many GPUs do we have?")
    err, device_count = cuda.cuDeviceGetCount()
    print(f"   Found {device_count} GPU(s)")
    
    # Step 3: Get a handle to the first GPU (device 0)
    print("\n3. Getting device handle...")
    err, device = cuda.cuDeviceGet(0)  # 0 = first device
    print(f"   Device handle: {device}")
    
    # Step 4: Query device properties
    print("\n4. Device Properties:")
    print("-" * 60)
    
    # Device name
    err, name = cuda.cuDeviceGetName(256, device)
    name_str = name.decode() if isinstance(name, bytes) else name
    print(f"   Name: {name_str}")
    
    # Number of Streaming Multiprocessors (SMs)
    # SMs are the actual processors that execute your code
    err, num_sms = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        device
    )
    print(f"   Streaming Multiprocessors (SMs): {num_sms}")
    
    # Compute capability (like GPU version)
    err, major = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        device
    )
    err, minor = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        device
    )
    print(f"   Compute Capability: {major}.{minor}")
    
    # Max threads per block
    err, max_threads = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        device
    )
    print(f"   Max threads per block: {max_threads}")
    
    # Warp size (threads execute in groups of this size)
    err, warp_size = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        device
    )
    print(f"   Warp size: {warp_size} threads")
    
    # Total memory
    err, total_mem = cuda.cuDeviceTotalMem(device)
    print(f"   Total memory: {total_mem / (1024**3):.2f} GB")
    
    print("\n" + "=" * 60)
    print("KEY CONCEPTS:")
    print("=" * 60)
    print("""
- Device = Your GPU (you can have multiple)
- SM (Streaming Multiprocessor) = Physical processor on GPU
- Warp = 32 threads that execute together
- Block = Group of threads (max 1024)
- Compute Capability = GPU generation/features
    """)


if __name__ == "__main__":
    main()

