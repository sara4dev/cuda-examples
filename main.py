#!/usr/bin/env python3
"""
CUDA Examples - Learn GPU Programming Step by Step

Start with Lesson 1 and work your way up!
"""


def main():
    print("=" * 70)
    print("CUDA Learning Examples - One Concept at a Time")
    print("=" * 70)
    print()
    print("Lessons (start with #1):")
    print()
    print("  1️⃣  Device Info      → python 01_device_info.py")
    print("      Learn: What is a GPU device and its properties")
    print()
    print("  2️⃣  Memory Basics    → python 02_memory_basics.py")
    print("      Learn: How to allocate and copy data to/from GPU")
    print()
    print("  3️⃣  First Kernel     → ./run_lesson.sh 03_first_kernel.py")
    print("      Learn: Threads, blocks, and launching kernels")
    print()
    print("  4️⃣  Tensor Cores     → ./run_lesson.sh 04_tensor_cores.py")
    print("      Learn: What are tensor cores and their performance")
    print()
    print("  5️⃣  cuDNN Selection  → ./run_lesson.sh 05_cudnn_tensor_cores.py")
    print("      Learn: How cuDNN auto-selects tensor vs CUDA cores")
    print()
    print("=" * 70)
    print("Quick start:")
    print("  python 01_device_info.py")
    print("  python 02_memory_basics.py")
    print("  ./run_lesson.sh 03_first_kernel.py")
    print("  ./run_lesson.sh 04_tensor_cores.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

