"""
Lesson 2: CUDA Memory Management
Goal: Understand how to allocate and copy data to/from GPU
"""

from cuda.bindings import driver as cuda
import numpy as np


def main():
    print("=" * 60)
    print("LESSON 2: GPU Memory Basics")
    print("=" * 60)
    
    # Initialize CUDA
    err, = cuda.cuInit(0)
    err, device = cuda.cuDeviceGet(0)
    
    # Create a context (runtime environment for GPU operations)
    print("\n1. Creating CUDA context...")
    err, context = cuda.cuCtxCreate(None, 0, device)
    print(f"   Context created: {context}")
    print("   (Think of context like a process - it manages GPU resources)")
    
    try:
        # Create some data on CPU (host)
        print("\n2. Creating data on CPU (host)...")
        host_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        print(f"   Host data: {host_data}")
        print(f"   Size: {host_data.nbytes} bytes")
        
        # Allocate memory on GPU (device)
        print("\n3. Allocating memory on GPU (device)...")
        num_bytes = host_data.nbytes
        err, device_ptr = cuda.cuMemAlloc(num_bytes)
        print(f"   Device pointer: {device_ptr}")
        print(f"   Allocated {num_bytes} bytes on GPU")
        
        # Copy data from CPU to GPU (Host to Device)
        print("\n4. Copying data from CPU → GPU (HtoD)...")
        err, = cuda.cuMemcpyHtoD(device_ptr, host_data, num_bytes)
        print(f"   Copy complete! Data is now on GPU")
        
        # Allocate memory to receive data back
        print("\n5. Creating empty array to receive data back...")
        result = np.zeros_like(host_data)
        print(f"   Result (before): {result}")
        
        # Copy data back from GPU to CPU (Device to Host)
        print("\n6. Copying data from GPU → CPU (DtoH)...")
        err, = cuda.cuMemcpyDtoH(result, device_ptr, num_bytes)
        print(f"   Result (after):  {result}")
        
        # Verify
        print("\n7. Verification:")
        if np.array_equal(host_data, result):
            print("   ✓ Success! Data matches perfectly")
        else:
            print("   ✗ Error! Data doesn't match")
        
        # Free GPU memory
        print("\n8. Freeing GPU memory...")
        err, = cuda.cuMemFree(device_ptr)
        print("   GPU memory freed")
        
    finally:
        # Always destroy context when done
        print("\n9. Destroying context...")
        err, = cuda.cuCtxDestroy(context)
        print("   Context destroyed")
    
    print("\n" + "=" * 60)
    print("KEY CONCEPTS:")
    print("=" * 60)
    print("""
Memory Locations:
- Host = CPU memory (your regular RAM)
- Device = GPU memory (separate from CPU)

Memory Operations:
- cuMemAlloc() = Allocate GPU memory
- cuMemcpyHtoD() = Copy Host to Device (CPU → GPU)
- cuMemcpyDtoH() = Copy Device to Host (GPU → CPU)
- cuMemFree() = Free GPU memory

Important:
- GPU can't access CPU memory directly
- CPU can't access GPU memory directly
- You must explicitly copy data between them
    """)


if __name__ == "__main__":
    main()

