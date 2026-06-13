#!/usr/bin/env bash
# Build mujofil-warp's native module (M5b): own shared-device Filament renderer +
# mujofil's SceneBridge/material/light source (vendored unchanged) -> torch CUDA.
# mujofil's PyPI release is NOT touched.
set -e
cd "$(dirname "$0")"

FILAMENT_DIR=${FILAMENT_DIR:-/home/mumuksh/Visual-Fidelity-Mujoco/deps/filament}
MUJOFIL_SRC=${MUJOFIL_SRC:-/home/mumuksh/MuJoCo-Filament}
SRC=../third_party/filament-src
DLPACK_INC=../spikes/dlpack_torch/include
MUJOCO_INC="$MUJOFIL_SRC/third_party/mujoco/include"

PY=$(python -c "import sys; print(sys.executable)")
PYINC=$($PY -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INC=$($PY -c "import pybind11; print(pybind11.get_include())")
EXT_SUFFIX=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

INC=(
  -Ivendor
  -I"$SRC/libs/bluevk/include"
  -I"$SRC/libs/utils/include"
  -I"$FILAMENT_DIR/include"
  -I"$MUJOCO_INC"
  -I"$DLPACK_INC"
  -I/usr/include
  -I"$PYINC" -I"$PYBIND_INC"
)

CXXFLAGS=(-O2 -fPIC -std=c++17 -stdlib=libc++ -DNDEBUG)

SRCS=(
  render_module.cpp
  renderer_warp.cpp
  vendor/core/scene_bridge.cpp
  vendor/core/material_manager.cpp
  vendor/core/light_manager.cpp
)

mkdir -p obj
OBJS=()
for s in "${SRCS[@]}"; do
  o="obj/$(echo "$s" | tr '/' '_').o"
  echo "  CC $s"
  clang++ "${CXXFLAGS[@]}" "${INC[@]}" -c "$s" -o "$o"
  OBJS+=("$o")
done

# Filament static libs (mujofil's full set) + CUDA.
FIL_LIBS="-lfilament -lbackend -lbluegl -lbluevk -lfilabridge -lfilaflat \
  -lutils -lgeometry -lsmol-v -libl -limage -lcamutils -lfilameshio -lgltfio \
  -lgltfio_core -luberarchive -lmeshoptimizer -lktxreader -lbasis_transcoder \
  -ldracodec -lstb -luberzlib -lzstd -lvkshaders"

echo "  LD _mujofil_warp${EXT_SUFFIX}"
clang++ -shared -stdlib=libc++ "${OBJS[@]}" \
    -L"$FILAMENT_DIR/lib/x86_64" -L/usr/lib/x86_64-linux-gnu \
    -Wl,--start-group $FIL_LIBS -Wl,--end-group \
    -lcudart -lpthread -ldl -lz \
    -o "_mujofil_warp${EXT_SUFFIX}"

echo "built native/_mujofil_warp${EXT_SUFFIX}"
