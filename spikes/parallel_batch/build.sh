#!/usr/bin/env bash
# Build the parallel-batch (gl_Layer single-pass vs N serial draws) spike.
# Pure EGL + desktop GL 4.1 + GLEW. No Filament, no CUDA.
set -e
cd "$(dirname "$0")"
CXX=${CXX:-g++}
"$CXX" -std=c++17 -O2 parallel_batch.cpp -lGLEW -lEGL -lGL -o parallel_batch
echo "built ./spikes/parallel_batch/parallel_batch"
