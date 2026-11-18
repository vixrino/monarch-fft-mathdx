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
# source correlation for ncu and SASS; unlike -G it keeps optimizations on
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

if(NOT MATHDX_ROOT)
  message(FATAL_ERROR "set -DMATHDX_ROOT=<dir>/nvidia/mathdx/YY.MM")
endif()
find_package(mathdx REQUIRED COMPONENTS cufftdx cublasdx CONFIG
             PATHS ${MATHDX_ROOT} NO_DEFAULT_PATH)
message(STATUS "MathDx ${mathdx_VERSION} at ${MATHDX_ROOT}, sm_${CUDA_ARCH}")

# header-only GEMM path; 25.12 has no _no_lto target, see docs/decisions.md
if(TARGET mathdx::cublasdx_no_lto)
  set(cublasdx_target mathdx::cublasdx_no_lto)
else()
  set(cublasdx_target mathdx::cublasdx)
endif()

add_executable(dx_smoke cuda/common/dx_smoke.cu)
target_include_directories(dx_smoke PRIVATE cuda/common)
target_link_libraries(dx_smoke PRIVATE mathdx::cufftdx ${cublasdx_target})

find_package(CUDAToolkit REQUIRED)

# provenance baked in at configure time; the runner always configures a fresh clone
execute_process(COMMAND git rev-parse --short HEAD WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE git_commit OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
execute_process(COMMAND git status --porcelain --untracked-files=no WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE git_status OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT git_commit)
  set(git_commit unknown)
endif()
if(git_status)
  set(git_dirty 1)
else()
  set(git_dirty 0)
endif()

add_executable(cufftdx_block_fft cuda/cufftdx/block_fft.cu)
target_include_directories(cufftdx_block_fft PRIVATE cuda/common)
target_compile_definitions(cufftdx_block_fft PRIVATE
  DXB_SM=${CUDA_ARCH}0
  DXB_GIT_COMMIT="${git_commit}"
  DXB_GIT_DIRTY=${git_dirty}
  DXB_MATHDX_VERSION="${mathdx_VERSION}")
target_link_libraries(cufftdx_block_fft PRIVATE mathdx::cufftdx CUDA::cufft CUDA::nvml)

# cuFFTDx tuning grid, one binary per N so the ~200 instantiations build in parallel; see docs/decisions.md
foreach(n 256 512 1024 2048 4096 8192 16384)
  add_executable(cufftdx_grid_${n} cuda/cufftdx/block_fft.cu)
  target_include_directories(cufftdx_grid_${n} PRIVATE cuda/common)
  target_compile_definitions(cufftdx_grid_${n} PRIVATE
    DXB_GRID_N=${n}
    DXB_SM=${CUDA_ARCH}0
    DXB_GIT_COMMIT="${git_commit}"
    DXB_GIT_DIRTY=${git_dirty}
    DXB_MATHDX_VERSION="${mathdx_VERSION}")
  target_link_libraries(cufftdx_grid_${n} PRIVATE mathdx::cufftdx CUDA::cufft CUDA::nvml)
  set_target_properties(cufftdx_grid_${n} PROPERTIES EXCLUDE_FROM_ALL ON)
  list(APPEND grid_targets cufftdx_grid_${n})
endforeach()
add_custom_target(cufftdx_grid DEPENDS ${grid_targets})

add_executable(monarch_fft cuda/monarch/monarch_fft.cu)
target_include_directories(monarch_fft PRIVATE cuda/common)
target_compile_definitions(monarch_fft PRIVATE
  DXB_SM=${CUDA_ARCH}0
  DXB_GIT_COMMIT="${git_commit}"
  DXB_GIT_DIRTY=${git_dirty}
  DXB_MATHDX_VERSION="${mathdx_VERSION}")
target_link_libraries(monarch_fft PRIVATE ${cublasdx_target} CUDA::cufft CUDA::nvml)

# which cuBLASDx configuration reaches the tensor cores on this SM; see docs/decisions.md
add_executable(mma_probe cuda/monarch/mma_probe.cu)
target_include_directories(mma_probe PRIVATE cuda/common)
target_compile_definitions(mma_probe PRIVATE DXB_SM=${CUDA_ARCH}0)
target_link_libraries(mma_probe PRIVATE ${cublasdx_target})

enable_testing()
add_test(NAME cufftdx_accuracy COMMAND cufftdx_block_fft --check)
add_test(NAME monarch_accuracy COMMAND monarch_fft --check)
