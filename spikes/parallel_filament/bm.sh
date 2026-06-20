#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
FILAMENT_DIR=$HOME/filament-build/out/release/filament
SRC=../../third_party/filament-src
FIL_LIBS="-lfilament -lbackend -lbluegl -lbluevk -lfilabridge -lfilaflat -lutils -lgeometry -lsmol-v -libl -limage -lcamutils -lfilameshio -lgltfio -lgltfio_core -luberarchive -lmeshoptimizer -lktxreader -lbasis_transcoder -ldracodec -lstb -luberzlib -lzstd -lvkshaders"
clang++ -O2 -fPIC -std=c++17 -stdlib=libc++ -DNDEBUG \
  -I"$SRC/libs/utils/include" -I"$SRC/libs/math/include" -I"$FILAMENT_DIR/include" -I/usr/include \
  multiworld.cpp -L"$FILAMENT_DIR/lib/x86_64" -L/usr/lib/x86_64-linux-gnu \
  -Wl,--start-group $FIL_LIBS -Wl,--end-group \
  -lEGL -lGL -lcudart -lpthread -ldl -lz -lc++ -lc++abi -o multiworld
echo built
