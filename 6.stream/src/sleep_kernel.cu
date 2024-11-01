#include "cuda_runtime.h"
#include "utils.hpp"

#define MAX_ITER 10000000   // 决定了kernel运行的时长



__global__ void sleep_kernel(int64_t run_nums){
    int64_t current_nums = 0;
    int64_t start = clock64();
    while (current_nums < run_nums){
        current_nums = clock64() - start;
    }
}

void SleepSingleStream(float *src_host, int width, int block_size, int cout) {
    // 计算在device上申请内存的大小
    int size = width * width * sizeof(float);

    // 申请device上的内存
    float *src_device;
    float *target_device;
    CUDACHECK(cudaMalloc((void**)&src_device, size));
    CUDACHECK(cudaMalloc((void**)&target_device, size));

    for (int i=0; i<cout; i++){
        CUDACHECK(cudaMemcpy(src_device, src_host, size, cudaMemcpyHostToDevice));
        dim3 dim_block = (block_size, block_size);
        dim3 dim_grid = ((width + dim_block.x - 1)/dim_block.x, (width + dim_block.y - 1)/dim_block.y);
        sleep_kernel<<<dim_grid, dim_block>>>(MAX_ITER);
        // 而在默认流中，操作是按顺序执行的。因此，cudaMemcpy 会等待 sleep_kernel 内核执行结束后再开始执行
        CUDACHECK(cudaMemcpy(src_host, src_device, size, cudaMemcpyDeviceToHost));
    }

    CUDACHECK(cudaDeviceSynchronize());
    PEEK_LAST_KERNEL_CHECK()
    cudaFree(target_device);
    cudaFree(src_device); 
}

void SleepMultiStream(float *src_host, int width, int block_size, int cout) {
    // 计算在device上申请内存的大小
    int size = width * width * sizeof(float);

    // 申请device上的内存
    float *src_device;
    float *target_device;
    CUDACHECK(cudaMalloc((void**)&src_device, size));
    CUDACHECK(cudaMalloc((void**)&target_device, size));

    // 创建需要的流
    cudaStream_t stream[cout];
    for (int i = 0; i < cout; i++) {
        CUDACHECK(cudaStreamCreate(&stream[i]));  // 注意要传入指针类型
    }
    
    for (int i=0; i<cout; i++){
        CUDACHECK(cudaMemcpyAsync(src_device, src_host, size, cudaMemcpyHostToDevice, stream[i]));
        dim3 dim_block = (block_size, block_size);
        dim3 dim_grid = ((width + dim_block.x - 1)/dim_block.x, (width + dim_block.y - 1)/dim_block.y);
        sleep_kernel<<<dim_grid, dim_block, 0, stream[i]>>>(MAX_ITER);   // 0表示shared_mem
        CUDACHECK(cudaMemcpyAsync(src_host, src_device, size, cudaMemcpyDeviceToHost, stream[i]));  // 更换为异步的API
    }

    CUDACHECK(cudaDeviceSynchronize());
    PEEK_LAST_KERNEL_CHECK()
    

    // 释放流，以及在device上申请的其他指针的空间
    for(int i=0; i < cout; i++){
        cudaStreamDestroy(stream[i]);
    }
    cudaFree(target_device);
    cudaFree(src_device); 


}