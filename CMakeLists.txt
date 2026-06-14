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
# Static libs are baked into the module; nothing to ship separately.
# ============================================================================
set(FILAMENT_VERSION "1.56.3")
if(NOT FILAMENT_DIR)
    set(FILAMENT_DIR "$ENV{FILAMENT_DIR}")
endif()
if(NOT FILAMENT_DIR)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "mujofil-warp currently supports Linux only.")
    endif()
    set(FILAMENT_DIR "${CMAKE_BINARY_DIR}/_filament")
    if(NOT EXISTS "${FILAMENT_DIR}/include")
        set(_fil_url "https://github.com/google/filament/releases/download/v${FILAMENT_VERSION}/filament-v${FILAMENT_VERSION}-linux.tgz")
        message(STATUS "FILAMENT_DIR not set — downloading Filament v${FILAMENT_VERSION}...")
        file(MAKE_DIRECTORY "${FILAMENT_DIR}")
        file(DOWNLOAD "${_fil_url}" "${CMAKE_BINARY_DIR}/filament.tgz" SHOW_PROGRESS STATUS _dl)
        list(GET _dl 0 _dl_rc)
        if(NOT _dl_rc EQUAL 0)
            message(FATAL_ERROR "Failed to download Filament from ${_fil_url} (${_dl}).")
        endif()
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/filament.tgz"
            WORKING_DIRECTORY "${FILAMENT_DIR}")
        # The tarball extracts into a 'filament/' subdir — flatten it.
        if(EXISTS "${FILAMENT_DIR}/filament/include")
            file(GLOB _fil_children "${FILAMENT_DIR}/filament/*")
            foreach(_c ${_fil_children})
                get_filename_component(_n "${_c}" NAME)
                file(RENAME "${_c}" "${FILAMENT_DIR}/${_n}")
            endforeach()
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
# MuJoCo headers — from the pip `mujoco` package (no install of libmujoco; the
# bindings only read mjModel/mjData via pointers). `mujoco` is a runtime dep.
# ============================================================================
if(NOT Python_EXECUTABLE)
    find_package(Python COMPONENTS Interpreter REQUIRED)
endif()
if(NOT MUJOCO_INCLUDE_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
            "import os,mujoco;print(os.path.join(os.path.dirname(mujoco.__file__),'include'))"
        OUTPUT_VARIABLE MUJOCO_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _mj_rc)
    if(NOT _mj_rc EQUAL 0 OR NOT EXISTS "${MUJOCO_INCLUDE_DIR}/mujoco/mujoco.h")
        message(FATAL_ERROR "Could not locate MuJoCo headers via pip `mujoco`. "
            "Install it (pip install mujoco) or pass -DMUJOCO_INCLUDE_DIR=...")
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
    install(TARGETS ${tgt} DESTINATION mujofil_warp)
endfunction()

# OpenGL single-sync backend (default) — needs GL + X11.
add_warp_backend(_mujofil_warp_gl _mujofil_warp_gl
    native/renderer_gl.cpp "" "GL;X11")

# Vulkan shared-device backend — needs vendored bluevk + utils source headers;
# links bluegl/bluevk (already in FILAMENT_LIBS), no external -lvulkan (bluevk
# owns the symbols).
add_warp_backend(_mujofil_warp _mujofil_warp
    native/renderer_warp.cpp "${BLUEVK_INC};${UTILS_SRC_INC}" "")

# ============================================================================
# Package data: compiled Filament materials (loaded via VF_MUJOCO_MATERIALS_DIR,
# which mujofil_warp/__init__.py points at this install dir).
# ============================================================================
file(GLOB PREBUILT_MATERIALS "${CMAKE_SOURCE_DIR}/mujofil_warp/materials/*.filamat")
if(PREBUILT_MATERIALS)
    install(FILES ${PREBUILT_MATERIALS} DESTINATION mujofil_warp/materials)
endif()
