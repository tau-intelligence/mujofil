#!/usr/bin/env bash
# Build the forked EGL Filament artifact INSIDE the same manylinux_2_34 container
# the wheels are built in, so the static libs' libc++/glibc ABI matches exactly.
# Mirrors the cibuildwheel before-all toolchain setup from pyproject.toml.
set -euo pipefail

REPO=/io                      # mujofil-warp source, bind-mounted
OUT=/io/_filament_egl_manylinux

echo "::: installing build deps :::"
dnf install -y wget tar xz gzip git cmake ninja-build libatomic \
    libpng-devel zlib-devel libglvnd-devel mesa-libGL-devel mesa-libEGL-devel \
    >/tmp/dnf.log 2>&1 || yum install -y wget tar xz gzip git cmake ninja-build \
    libatomic libpng-devel zlib-devel libglvnd-devel mesa-libGL-devel mesa-libEGL-devel

echo "::: fetching LLVM 18.1.8 (clang + bundled libc++) :::"
if [ ! -x /opt/llvm/bin/clang++ ]; then
  wget -q https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz -O /tmp/llvm.tar.xz
  mkdir -p /opt/llvm && tar -xf /tmp/llvm.tar.xz -C /opt/llvm --strip-components=1
  find /opt/llvm \( -name 'libc++.so*' -o -name 'libc++abi.so*' -o -name 'libunwind.so*' \) -exec cp -P {} /usr/lib64/ \; && ldconfig
  dnf install -y ncurses-compat-libs >/dev/null 2>&1 || true
  [ -e /usr/lib64/libtinfo.so.5 ] || ln -s "$(ls /usr/lib64/libtinfo.so.6* | head -1)" /usr/lib64/libtinfo.so.5
  ldconfig
fi

export CC=/opt/llvm/bin/clang
export CXX=/opt/llvm/bin/clang++
export CFLAGS="--gcc-toolchain=/opt/rh/gcc-toolset-14/root/usr"
export CXXFLAGS="--gcc-toolchain=/opt/rh/gcc-toolset-14/root/usr"
export LDFLAGS="--gcc-toolchain=/opt/rh/gcc-toolset-14/root/usr"
# Filament's build.sh hardcodes clang; ensure our LLVM clang is first on PATH.
export PATH=/opt/llvm/bin:/opt/rh/gcc-toolset-14/root/usr/bin:$PATH

echo "::: building forked Filament (applies patches 0001+0002) :::"
rm -rf "$OUT" /tmp/fil_src_ml
FILAMENT_SRC_DIR=/tmp/fil_src_ml bash "$REPO/packaging/build_filament_egl.sh" "$OUT"

echo "::: verify the layered fork is present :::"
n=$(strings "$OUT/lib/x86_64/libfilamat.a" 2>/dev/null | grep -c "FILAMENT_LAYERED_BATCH" || true)
echo "FILAMENT_LAYERED_BATCH markers in libfilamat.a: $n"
test "$n" -ge 1 || { echo "ERROR: forked layered code missing from artifact"; exit 1; }

echo "::: package tarball (filament/ at top, --strip-components=1 layout) :::"
cd "$(dirname "$OUT")"
rm -rf /tmp/fil_pkg && mkdir -p /tmp/fil_pkg/filament
cp -a "$OUT/." /tmp/fil_pkg/filament/
cd /tmp/fil_pkg && tar czf /io/filament-egl-v1.56.3-linux-x86_64.tgz filament
echo "::: DONE -> /io/filament-egl-v1.56.3-linux-x86_64.tgz :::"
ls -la /io/filament-egl-v1.56.3-linux-x86_64.tgz
