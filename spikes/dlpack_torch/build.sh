#!/usr/bin/env bash
# Build the DLPack->torch pybind module against the venv's pybind11 + system CUDA.
set -e
cd "$(dirname "$0")"

PY=$(python -c "import sys; print(sys.executable)")
PYINC=$($PY -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INC=$($PY -c "import pybind11; print(pybind11.get_include())")
EXT_SUFFIX=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

clang++ -O2 -fPIC -shared -std=c++17 \
    dlpack_cuda.cpp \
    -I"$PYINC" -I"$PYBIND_INC" -Iinclude -I/usr/include \
    -L/usr/lib/x86_64-linux-gnu -lcudart \
    -o "dlpack_cuda${EXT_SUFFIX}"

echo "built dlpack_cuda${EXT_SUFFIX}"
