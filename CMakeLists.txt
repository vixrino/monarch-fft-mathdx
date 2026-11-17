cmake_minimum_required(VERSION 3.26...3.31)
project(dxbench LANGUAGES CXX CUDA)

set(CUDA_ARCH 75 CACHE STRING "Target SM, e.g. 75 for T4")
set(MATHDX_ROOT "" CACHE PATH "Unpacked MathDx, e.g. <dir>/nvidia-mathdx-25.12.1-cuda12/nvidia/mathdx/25.12")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

if(NOT MATHDX_ROOT)
  message(FATAL_ERROR "set -DMATHDX_ROOT=<dir>/nvidia/mathdx/YY.MM")
endif()
find_package(mathdx REQUIRED COMPONENTS cufftdx cublasdx CONFIG
             PATHS ${MATHDX_ROOT} NO_DEFAULT_PATH)
message(STATUS "MathDx ${mathdx_VERSION} at ${MATHDX_ROOT}, sm_${CUDA_ARCH}")

add_executable(dx_smoke cuda/common/dx_smoke.cu)
target_include_directories(dx_smoke PRIVATE cuda/common)
# header-only GEMM path; 25.12 has no _no_lto target, see docs/decisions.md
if(TARGET mathdx::cublasdx_no_lto)
  set(cublasdx_target mathdx::cublasdx_no_lto)
else()
  set(cublasdx_target mathdx::cublasdx)
endif()

target_link_libraries(dx_smoke PRIVATE mathdx::cufftdx ${cublasdx_target})
