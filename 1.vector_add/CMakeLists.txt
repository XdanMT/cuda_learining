cmake_minimum_required(VERSION 3.10)

set(ENV{CUDACXX} "/usr/local/cuda/bin/nvcc")
project(CUDA_TEST LANGUAGES CUDA CXX)

# 通过CheckLanuage判断CUDA是否可用
include(CheckLanguage)
check_language(CUDA)

# Add CUDA files to the project
set(CUDA_FILES
    vector_addition.cu
)

# 该方式不需要使用 cuda_add_executable()
add_executable(vector_addition ${CUDA_FILES})
