cmake_minimum_required(VERSION 3.20)
project(mujofil_warp VERSION 0.1.0 LANGUAGES CXX)

# ============================================================================
# Toolchain: Filament's prebuilt libs are built with libc++, and our native
# code uses the Vulkan shared-context path that requires Clang + libc++. We
# therefore REQUIRE a Clang compiler and force -stdlib=libc++ everywhere.
# (scikit-build passes CMAKE_CXX_COMPILER from the CC/CXX env; CI sets clang.)
# ============================================================================
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
        "mujofil-warp must be built with Clang (Filament uses libc++).\n"
        "Re-run with:  CC=clang CXX=clang++ pip install . \n"
        "Detected compiler: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-stdlib=libc++ -O2 -DNDEBUG)
add_link_options(-stdlib=libc++)

# ============================================================================
# Filament — prebuilt static libs, auto-downloaded if FILAMENT_DIR isn't given.
# ============================================================================
# Filament — a CUSTOM EGL-enabled build (Google's prebuilt Linux Filament is
# GLX-only and cannot render headless). One build serves both the GL (surfaceless
# EGL) and Vulkan backends; its static libs are baked into the modules.
# Resolution order:
#   1. FILAMENT_DIR (env or -D) -> use as-is (fastest; CI sets this).
#   2. Download a prebuilt EGL Filament artifact (built from the same source +
#      patches via packaging/build_filament_egl.sh, hosted as a release asset).
#   3. Build it from source (slow ~20-30 min; needs clang/libc++/cmake/ninja).
# ============================================================================
set(FILAMENT_VERSION "1.56.3")
# Prebuilt EGL Filament artifact (overridable). Empty disables the download step.
set(MUJOFIL_WARP_FILAMENT_URL
    "https://github.com/tau-intelligence/mujofil/releases/download/filament-egl-v${FILAMENT_VERSION}/filament-egl-v${FILAMENT_VERSION}-linux-x86_64.tgz"
    CACHE STRING "URL of the prebuilt EGL-enabled Filament artifact")
if(NOT FILAMENT_DIR)
    set(FILAMENT_DIR "$ENV{FILAMENT_DIR}")
endif()
if(NOT FILAMENT_DIR)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "mujofil-warp currently supports Linux only.")
    endif()
    set(FILAMENT_DIR "${CMAKE_BINARY_DIR}/_filament_egl")
    if(NOT EXISTS "${FILAMENT_DIR}/include")
        set(_got_filament FALSE)
        # 2. Try the prebuilt artifact first (fast path).
        if(MUJOFIL_WARP_FILAMENT_URL)
            message(STATUS "Downloading prebuilt EGL Filament: ${MUJOFIL_WARP_FILAMENT_URL}")
            file(DOWNLOAD "${MUJOFIL_WARP_FILAMENT_URL}" "${CMAKE_BINARY_DIR}/filament-egl.tgz"
                STATUS _dl_status)
            list(GET _dl_status 0 _dl_rc)
            if(_dl_rc EQUAL 0)
                file(MAKE_DIRECTORY "${FILAMENT_DIR}")
                execute_process(
                    COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/filament-egl.tgz"
                    WORKING_DIRECTORY "${FILAMENT_DIR}")
                # Tarball extracts to a 'filament/' subdir — flatten it.
                if(EXISTS "${FILAMENT_DIR}/filament/include")
                    file(GLOB _fc "${FILAMENT_DIR}/filament/*")
                    foreach(_c ${_fc})
                        get_filename_component(_n "${_c}" NAME)
                        file(RENAME "${_c}" "${FILAMENT_DIR}/${_n}")
                    endforeach()
                endif()
                if(EXISTS "${FILAMENT_DIR}/include")
                    set(_got_filament TRUE)
                endif()
            else()
                message(STATUS "Prebuilt EGL Filament not available (${_dl_status}); will build from source.")
            endif()
        endif()
        # 3. Fall back to building from source.
        if(NOT _got_filament)
            message(STATUS "Building custom EGL Filament v${FILAMENT_VERSION} from source (slow ~20-30 min)...")
            execute_process(
                COMMAND bash "${CMAKE_SOURCE_DIR}/packaging/build_filament_egl.sh" "${FILAMENT_DIR}"
                RESULT_VARIABLE _fil_rc)
            if(NOT _fil_rc EQUAL 0)
                message(FATAL_ERROR "EGL Filament build failed (rc=${_fil_rc}). "
                    "See packaging/build_filament_egl.sh; it needs git, clang, libc++ dev, cmake, ninja.")
            endif()
        endif()
    endif()
endif()
if(NOT EXISTS "${FILAMENT_DIR}/include")
    message(FATAL_ERROR "FILAMENT_DIR='${FILAMENT_DIR}' has no include/ — bad Filament path.")
endif()
set(FILAMENT_INCLUDE_DIR "${FILAMENT_DIR}/include")
set(FILAMENT_LIB_DIR "${FILAMENT_DIR}/lib/x86_64")
message(STATUS "Filament: ${FILAMENT_DIR}")

# Full static-lib set (circular deps -> wrapped in --start-group at link time).
set(FILAMENT_LIBS
    filament backend bluegl bluevk filabridge filaflat utils geometry smol-v
    ibl image camutils filameshio gltfio gltfio_core uberarchive meshoptimizer
    ktxreader basis_transcoder dracodec stb uberzlib zstd vkshaders)

# ============================================================================
# MuJoCo headers — VENDORED (third_party/mujoco/include, Apache-2.0). The bindings
# only read mjModel/mjData via pointers and call no MuJoCo functions, so we never
# link or ship libmujoco; `mujoco` stays a pure RUNTIME pip dependency. Vendoring
# the headers means the BUILD needs no `mujoco` install (build isolation has none).
# Falls back to a pip `mujoco` install if the vendored copy is absent.
# ============================================================================
if(NOT Python_EXECUTABLE)
    find_package(Python COMPONENTS Interpreter REQUIRED)
endif()
if(NOT MUJOCO_INCLUDE_DIR)
    if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/mujoco/include/mujoco/mujoco.h")
        set(MUJOCO_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/mujoco/include")
    else()
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c
                "import os,mujoco;print(os.path.join(os.path.dirname(mujoco.__file__),'include'))"
            OUTPUT_VARIABLE MUJOCO_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _mj_rc)
        if(NOT _mj_rc EQUAL 0 OR NOT EXISTS "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h")
            message(FATAL_ERROR "MuJoCo headers not found (vendored copy missing and "
                "pip `mujoco` not importable). Restore third_party/mujoco/include or "
                "pass -DMUJOCO_INCLUDE_DIR=...")
        endif()
    endif()
endif()
message(STATUS "MuJoCo headers: ${MUJOCO_INCLUDE_DIR}")

# ============================================================================
# CUDA — headers + STATIC cudart so the wheel needs only the NVIDIA driver
# (libcuda.so) at runtime, not a CUDA toolkit install.
# ============================================================================
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    set(CUDA_INCLUDE_DIRS "${CUDAToolkit_INCLUDE_DIRS}")
    set(CUDA_RT_STATIC CUDA::cudart_static)
else()
    # Fallback: system CUDA headers + static cudart in the default multiarch dir.
    find_path(CUDA_INCLUDE_DIRS cuda_runtime.h PATHS /usr/include /usr/local/cuda/include)
    find_library(_cudart_static NAMES cudart_static
        PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)
    find_library(_culibos NAMES culibos PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)
    if(NOT CUDA_INCLUDE_DIRS OR NOT _cudart_static)
        message(FATAL_ERROR "CUDA toolkit not found (need cuda_runtime.h + libcudart_static.a). "
            "Install the CUDA toolkit or pass -DCUDAToolkit_ROOT=...")
    endif()
    set(CUDA_RT_STATIC ${_cudart_static} ${_culibos} rt)
endif()
message(STATUS "CUDA headers: ${CUDA_INCLUDE_DIRS}")

find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

set(VENDOR_CORE "${CMAKE_SOURCE_DIR}/native/vendor/core")
set(VENDOR_INC "${CMAKE_SOURCE_DIR}/native/vendor")
set(DLPACK_INC "${CMAKE_SOURCE_DIR}/third_party/dlpack/include")
# The Vulkan backend includes Filament's <backend/platforms/VulkanPlatform.h>,
# which pulls in bluevk + internal utils headers (e.g. utils/Hash.h) that the
# PREBUILT Filament package strips. We vendor those source headers here.
set(BLUEVK_INC "${CMAKE_SOURCE_DIR}/third_party/bluevk/include")
set(UTILS_SRC_INC "${CMAKE_SOURCE_DIR}/third_party/utils_include")

# Shared mujofil bridge source (vendored UNCHANGED) — compiled once per module
# (each module needs its own renderer.h substitution, so we list the sources).
set(BRIDGE_SOURCES
    ${VENDOR_CORE}/scene_bridge.cpp
    ${VENDOR_CORE}/material_manager.cpp
    ${VENDOR_CORE}/light_manager.cpp)

# Common include dirs / link libs shared by both backends. The vendored source
# includes "core/renderer.h" etc., so the include ROOT is native/vendor.
set(COMMON_INCLUDES
    ${CMAKE_SOURCE_DIR}/native ${VENDOR_INC}
    ${FILAMENT_INCLUDE_DIR} ${MUJOCO_INCLUDE_DIR} ${DLPACK_INC} ${CUDA_INCLUDE_DIRS})

# Helper: build one pybind module with a given renderer impl, module name, extra
# include dirs (PREPENDED, so vendored headers win over the prebuilt subset) and
# extra system link libs.
function(add_warp_backend tgt module_name renderer_src extra_inc extra_libs)
    pybind11_add_module(${tgt}
        native/render_module.cpp ${renderer_src} ${BRIDGE_SOURCES})
    target_compile_definitions(${tgt} PRIVATE MUJOFIL_WARP_MODULE=${module_name})
    target_include_directories(${tgt} PRIVATE ${extra_inc} ${COMMON_INCLUDES})
    target_link_directories(${tgt} PRIVATE ${FILAMENT_LIB_DIR})
    target_link_libraries(${tgt} PRIVATE
        -Wl,--start-group ${FILAMENT_LIBS} -Wl,--end-group
        ${CUDA_RT_STATIC} ${extra_libs}
        Threads::Threads c++ c++abi dl z)
    set_target_properties(${tgt} PROPERTIES
        OUTPUT_NAME ${module_name}
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH ON)
    install(TARGETS ${tgt} DESTINATION mujofil)
endfunction()

# OpenGL single-sync backend (default) — headless via surfaceless EGL (no X).
add_warp_backend(_mujofil_warp_gl _mujofil_warp_gl
    native/renderer_gl.cpp "" "GL;EGL")

# Vulkan shared-device backend — needs vendored bluevk + utils source headers;
# links bluegl/bluevk (already in FILAMENT_LIBS), no external -lvulkan (bluevk
# owns the symbols). MUST also link EGL: the EGL-from-source Filament
# libbackend.a references eglCreateWindowSurface (dead code on the Vulkan path),
# so the static link fails with 'undefined symbol: eglCreateWindowSurface'
# unless EGL is provided -- the symptom seen in the published 0.2.0 Vulkan module.
add_warp_backend(_mujofil_warp _mujofil_warp
    native/renderer_warp.cpp "${BLUEVK_INC};${UTILS_SRC_INC}" "EGL")

# ============================================================================
# Package data: compiled Filament materials (loaded via VF_MUJOCO_MATERIALS_DIR,
# which mujofil/__init__.py points at this install dir).
# ============================================================================
file(GLOB PREBUILT_MATERIALS "${CMAKE_SOURCE_DIR}/mujofil/materials/*.filamat")
if(PREBUILT_MATERIALS)
    install(FILES ${PREBUILT_MATERIALS} DESTINATION mujofil/materials)
endif()

# Layered (gl_Layer) material set for the parallel-batch / egocentric path. These
# are compiled with the FORKED matc (-g) so they carry the gl_Layer write + the
# textured_pbr normal/MR/emissive maps; ship them alongside the standard set
# (mujofil/__init__.py points VF_MUJOCO_MATERIALS_DIR here when layered=True).
file(GLOB PREBUILT_MATERIALS_LAYERED "${CMAKE_SOURCE_DIR}/mujofil/materials_layered/*.filamat")
if(PREBUILT_MATERIALS_LAYERED)
    install(FILES ${PREBUILT_MATERIALS_LAYERED} DESTINATION mujofil/materials_layered)
endif()
