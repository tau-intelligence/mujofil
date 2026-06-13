#!/usr/bin/env bash
# Build the EGL/GL <-> CUDA interop spike. Pure C++ (no .cu).
set -e
cd "$(dirname "$0")"

CXX=${CXX:-g++}
CUDA_INC=${CUDA_INC:-/usr/include}
CUDA_LIBDIR=${CUDA_LIBDIR:-/usr/lib/x86_64-linux-gnu}

"$CXX" -std=c++17 -O2 \
    gl_cuda_interop.cpp \
    -I"$CUDA_INC" \
    -L"$CUDA_LIBDIR" \
    -lEGL -lGLESv2 -lcudart \
    -o gl_cuda_interop

echo "built ./spikes/gl_cuda_interop/gl_cuda_interop"
