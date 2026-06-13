#!/usr/bin/env bash
# Build spike 2: Filament OpenGL backend -> our GL texture -> CUDA (zero-copy).
set -e
cd "$(dirname "$0")"

FILAMENT_DIR=${FILAMENT_DIR:-/home/mumuksh/Visual-Fidelity-Mujoco/deps/filament}
SRC=../../third_party/filament-src

INC=(
  -I"$SRC/libs/utils/include"
  -I"$FILAMENT_DIR/include"
  -I/usr/include
)

# Filament static libs (same set as native build) + GL/EGL + CUDA.
FIL_LIBS="-lfilament -lbackend -lbluegl -lbluevk -lfilabridge -lfilaflat \
  -lutils -lgeometry -lsmol-v -libl -limage -lcamutils -lfilameshio -lgltfio \
  -lgltfio_core -luberarchive -lmeshoptimizer -lktxreader -lbasis_transcoder \
  -ldracodec -lstb -luberzlib -lzstd -lvkshaders"

clang++ -O2 -fPIC -std=c++17 -stdlib=libc++ -DNDEBUG "${INC[@]}" \
    filament_gl_export.cpp \
    -L"$FILAMENT_DIR/lib/x86_64" -L/usr/lib/x86_64-linux-gnu \
    -Wl,--start-group $FIL_LIBS -Wl,--end-group \
    -lGL -lX11 -lcudart -lpthread -ldl -lz \
    -o filament_gl_export

echo "built ./spikes/filament_gl_export/filament_gl_export"
