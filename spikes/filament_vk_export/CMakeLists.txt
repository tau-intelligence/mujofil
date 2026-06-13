cmake_minimum_required(VERSION 3.16)
project(filament_vk_export CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prebuilt Filament location (override with -DFILAMENT_DIR=...)
if(NOT DEFINED FILAMENT_DIR)
    set(FILAMENT_DIR "/home/mumuksh/Visual-Fidelity-Mujoco/deps/filament")
endif()

set(FIL_INC "${FILAMENT_DIR}/include")
set(FIL_LIB "${FILAMENT_DIR}/lib/x86_64")

# Filament source headers (bluevk, utils, bundled Vulkan headers) — the prebuilt
# package strips backend-internal headers, so we pull them from a source checkout
# of the SAME tag (v1.56.3). Order matters: source bluevk include (with bundled
# vulkan/) must come BEFORE the system include path.
set(SRC "${CMAKE_SOURCE_DIR}/../../third_party/filament-src")
set(SRC_INC
    "${SRC}/libs/bluevk/include"      # bluevk/BlueVK.h + bundled vulkan/vulkan_core.h
    "${SRC}/libs/utils/include")      # utils/Hash.h etc.

add_executable(filament_vk_export main.cpp)
target_include_directories(filament_vk_export PRIVATE ${SRC_INC} ${FIL_INC})
target_link_directories(filament_vk_export PRIVATE ${FIL_LIB})

# Filament static libs (order matters for static linking).
set(FIL_LIBS
    filament backend bluegl bluevk filabridge filaflat utils geometry
    smol-v ibl image camutils vkshaders)

find_package(Threads REQUIRED)
target_link_libraries(filament_vk_export PRIVATE
    ${FIL_LIBS}
    Threads::Threads dl
    c++ c++abi)

# Filament is built with libc++; match it.
target_compile_options(filament_vk_export PRIVATE -stdlib=libc++)
target_link_options(filament_vk_export PRIVATE -stdlib=libc++)

# ---- Milestone 3a/3b: Filament renders into our exportable swapchain -> CUDA -
add_executable(render_export render_export.cpp)
target_include_directories(render_export PRIVATE ${SRC_INC} ${FIL_INC} /usr/include)
target_link_directories(render_export PRIVATE ${FIL_LIB} /usr/lib/x86_64-linux-gnu)
target_link_libraries(render_export PRIVATE
    ${FIL_LIBS}
    cudart
    Threads::Threads dl
    c++ c++abi)
target_compile_options(render_export PRIVATE -stdlib=libc++)
target_link_options(render_export PRIVATE -stdlib=libc++)
