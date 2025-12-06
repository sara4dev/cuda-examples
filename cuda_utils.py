"""
CUDA Utility Functions
Common helper functions for CUDA programming examples.
"""

from cuda.bindings import driver as cuda
from cuda.bindings import nvrtc

# Constants
DEVICE_NAME_BUFFER_SIZE = 256


def check_cuda_errors(result):
    """
    Check CUDA errors and raise RuntimeError if an error occurred.
    
    Args:
        result: A tuple where the first element is the error code.
    
    Returns:
        The second element of the result tuple if present, otherwise None.
    
    Raises:
        RuntimeError: If the error code is non-zero.
    """
    if result[0].value != 0:
        raise RuntimeError(f"CUDA Error: {result[0]}")
    return result[1] if len(result) > 1 else None


def init_cuda():
    """
    Initialize CUDA and get the first device.
    
    Returns:
        A tuple of (device, device_name, num_sms) containing the device handle,
        device name string, and number of streaming multiprocessors.
    
    Raises:
        RuntimeError: If CUDA initialization fails.
    """
    err, = cuda.cuInit(0)
    check_cuda_errors((err,))
    
    err, device = cuda.cuDeviceGet(0)
    check_cuda_errors((err,))
    
    # Get device name
    err, name = cuda.cuDeviceGetName(DEVICE_NAME_BUFFER_SIZE, device)
    check_cuda_errors((err,))
    device_name = name.decode() if isinstance(name, bytes) else name
    
    # Get number of SMs
    err, num_sms = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device
    )
    check_cuda_errors((err,))
    
    return device, device_name, num_sms


def create_cuda_context(device):
    """
    Create a CUDA context for the given device.
    
    Args:
        device: The CUDA device handle.
    
    Returns:
        The created context.
    
    Raises:
        RuntimeError: If context creation fails.
    """
    err, context = cuda.cuCtxCreate(None, 0, device)
    check_cuda_errors((err,))
    return context


def compile_cuda_kernel(source, arch='compute_75'):
    """
    Compile CUDA source code to PTX.
    
    Args:
        source: CUDA source code as a string.
        arch: Target GPU architecture (default: 'compute_75').
    
    Returns:
        PTX bytecode as bytes.
    
    Raises:
        RuntimeError: If compilation fails.
    """
    # Create program
    err, prog = nvrtc.nvrtcCreateProgram(
        str.encode(source), b"kernel.cu", 0, [], []
    )
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"Failed to create program: {err}")
    
    # Compile
    opts = [f'--gpu-architecture={arch}'.encode()]
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


def get_device_attributes(device):
    """
    Get common device attributes.
    
    Args:
        device: The CUDA device handle.
    
    Returns:
        A dictionary containing device attributes:
        - num_sms: Number of streaming multiprocessors
        - max_threads_per_block: Maximum threads per block
        - max_threads_per_sm: Maximum threads per SM
        - warp_size: Warp size (threads per warp)
        - compute_major: Compute capability major version
        - compute_minor: Compute capability minor version
        - total_memory: Total memory in bytes
    """
    attrs = {}
    
    err, attrs['num_sms'] = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device
    )
    check_cuda_errors((err,))
    
    err, attrs['max_threads_per_block'] = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device
    )
    check_cuda_errors((err,))
    
    err, attrs['max_threads_per_sm'] = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device
    )
    check_cuda_errors((err,))
    
    err, attrs['warp_size'] = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE, device
    )
    check_cuda_errors((err,))
    
    err, attrs['compute_major'] = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device
    )
    check_cuda_errors((err,))
    
    err, attrs['compute_minor'] = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device
    )
    check_cuda_errors((err,))
    
    err, attrs['total_memory'] = cuda.cuDeviceTotalMem(device)
    check_cuda_errors((err,))
    
    return attrs


def parse_compute_capability(compute_cap):
    """
    Parse a compute capability value into major and minor versions.
    
    Args:
        compute_cap: Compute capability as an integer (e.g., 75 for 7.5).
    
    Returns:
        A tuple of (major, minor) versions.
    """
    return divmod(compute_cap, 10)
