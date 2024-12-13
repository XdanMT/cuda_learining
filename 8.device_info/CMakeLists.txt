cmake_minimum_required(VERSION 3.10)

set(ENV{CUDACXX} "/usr/local/cuda/bin/nvcc")
project(CUDA_TEST LANGUAGES CUDA CXX)

# 通过CheckLanuage判断CUDA是否可用
include(CheckLanguage)
check_language(CUDA)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# Add CUDA files to the project
set(CUDA_FILES
    # src/image_resize_gpu.cu
)

# Add CPP files to the project
set(CXX_FILES
    main.cpp
    src/utils.cpp
)

# 该方式不需要使用 cuda_add_executable()
add_executable(device_info ${CUDA_FILES} ${CXX_FILES})
target_link_libraries(device_info ${OpenCV_LIBS} ${CUDA_LIBRARIES})

target_include_directories(device_info PUBLIC 
                            ${CMAKE_CURRENT_SOURCE_DIR}/include 
                            ${CUDA_INCLUDE_DIRS} 
                            ${OpenCV_INCLUDE_DIRS})


