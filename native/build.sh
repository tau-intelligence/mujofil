#!/usr/bin/env bash
# Build the mujofil-warp native module (milestone 5a).
# Filament source headers (bluevk + bundled vulkan + utils) come from the
# v1.56.3 sparse checkout; we LINK the prebuilt Filament static libs. mujofil's
# PyPI release is NOT touched.
set -e
cd "$(dirname "$0")"

FILAMENT_DIR=${FILAMENT_DIR:-/home/mumuksh/Visual-Fidelity-Mujoco/deps/filament}
SRC=../third_party/filament-src
DLPACK_INC=../spikes/dlpack_torch/include

PY=$(python -c "import sys; print(sys.executable)")
PYINC=$($PY -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INC=$($PY -c "import pybind11; print(pybind11.get_include())")
EXT_SUFFIX=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Filament static libs (same set + order mujofil uses).
FIL_LIBS="-lfilament -lbackend -lbluegl -lbluevk -lfilabridge -lfilaflat \
  -lutils -lgeometry -lsmol-v -libl -limage -lcamutils -lvkshaders"

clang++ -O2 -fPIC -shared -std=c++17 -stdlib=libc++ \
    render_module.cpp \
    -I"$SRC/libs/bluevk/include" \
    -I"$SRC/libs/utils/include" \
    -I"$FILAMENT_DIR/include" \
    -I"$DLPACK_INC" \
    -I/usr/include \
    -I"$PYINC" -I"$PYBIND_INC" \
    -L"$FILAMENT_DIR/lib/x86_64" \
    -L/usr/lib/x86_64-linux-gnu \
    $FIL_LIBS -lcudart -lpthread -ldl \
    -o "_mujofil_warp${EXT_SUFFIX}"

echo "built native/_mujofil_warp${EXT_SUFFIX}"
