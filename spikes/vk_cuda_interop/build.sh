#!/usr/bin/env bash
# Build the Vulkan<->CUDA interop spike.
# Pure C++ (no .cu): Vulkan writes, CUDA reads via runtime API. Link vulkan+cudart.
set -e
cd "$(dirname "$0")"

CXX=${CXX:-g++}
CUDA_INC=${CUDA_INC:-/usr/include}
CUDA_LIBDIR=${CUDA_LIBDIR:-/usr/lib/x86_64-linux-gnu}

"$CXX" -std=c++17 -O2 \
    vk_cuda_interop.cpp \
    -I"$CUDA_INC" \
    -L"$CUDA_LIBDIR" \
    -lvulkan -lcudart \
    -o vk_cuda_interop

echo "built ./spikes/vk_cuda_interop/vk_cuda_interop"
