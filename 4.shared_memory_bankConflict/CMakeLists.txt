cmake_minimum_required(VERSION 3.10)

set(ENV{CUDACXX} "/usr/local/cuda/bin/nvcc")
project(CUDA_TEST LANGUAGES CUDA CXX)

# 通过CheckLanuage判断CUDA是否可用
include(CheckLanguage)
check_language(CUDA)

# Add CUDA files to the project
set(CUDA_FILES
    src/matrix_mul_gpu.cu
    src/matrix_mul_gpu_shared.cu
    src/matrix_mul_gpu_shared_conflict.cu
    src/matrix_mul_gpu_shared_conflict_pad.cu
)

# Add CPP files to the project
set(CXX_FILES
    main.cpp
    src/matrix_mul_cpu.cpp
    src/utils.cpp
)

# 该方式不需要使用 cuda_add_executable()
add_executable(matrix_multiply ${CUDA_FILES} ${CXX_FILES})

target_include_directories(matrix_multiply PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)


