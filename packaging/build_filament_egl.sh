#!/usr/bin/env bash
# Build a custom EGL-enabled Filament for mujofil-warp's headless OpenGL backend.
#
# Google's prebuilt Linux Filament is GLX-only (needs an X server). mujofil-warp's
# GL backend renders HEADLESS via surfaceless EGL, which requires Filament built
# with FILAMENT_SUPPORTS_EGL_ON_LINUX (the `-e` flag). This script clones the
# pinned Filament source, applies our EGL desktop-GL fixes (see
# packaging/filament-patches/), and builds the static libs + headers.
#
# Usage:
#   packaging/build_filament_egl.sh [INSTALL_DIR]
# Result: $INSTALL_DIR/{include,lib/x86_64,...}  (default: ./_filament_egl)
#
# Requirements: git, clang/clang++, libc++ dev headers, cmake, ninja.
set -euo pipefail

FILAMENT_VERSION="${FILAMENT_VERSION:-v1.56.3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="$HERE/filament-patches"
INSTALL_DIR="${1:-$PWD/_filament_egl}"
SRC_DIR="${FILAMENT_SRC_DIR:-$PWD/_filament_src}"

# Reuse an existing successful build.
if [ -d "$INSTALL_DIR/include" ] && [ -d "$INSTALL_DIR/lib/x86_64" ]; then
    echo "EGL Filament already built at $INSTALL_DIR — skipping."
    exit 0
fi

command -v clang >/dev/null  || { echo "error: clang not found"; exit 1; }
command -v ninja >/dev/null  || { echo "error: ninja not found"; exit 1; }

# 1. Clone the pinned Filament source (shallow).
if [ ! -d "$SRC_DIR/.git" ]; then
    echo "Cloning Filament $FILAMENT_VERSION ..."
    git clone --depth 1 --branch "$FILAMENT_VERSION" \
        https://github.com/google/filament.git "$SRC_DIR"
fi

# 2. Apply our EGL desktop-GL / headless fixes (idempotent).
cd "$SRC_DIR"
for p in "$PATCH_DIR"/*.patch; do
    [ -e "$p" ] || continue
    if git apply --check "$p" 2>/dev/null; then
        echo "Applying $(basename "$p")"
        git apply "$p"
    elif git apply --reverse --check "$p" 2>/dev/null; then
        echo "$(basename "$p") already applied — skipping"
    else
        echo "error: cannot apply $(basename "$p")"; exit 1
    fi
done

# 3. Build with EGL-on-Linux enabled (release, install layout).
echo "Building Filament (EGL on Linux) — this takes a while ..."
CC=clang CXX=clang++ ./build.sh -e -i release

# 4. Stage the install tree where the caller asked for it.
BUILT="$SRC_DIR/out/release/filament"
[ -d "$BUILT/include" ] || { echo "error: build did not produce $BUILT"; exit 1; }
mkdir -p "$INSTALL_DIR"
cp -a "$BUILT/." "$INSTALL_DIR/"
echo "EGL Filament installed to $INSTALL_DIR"
