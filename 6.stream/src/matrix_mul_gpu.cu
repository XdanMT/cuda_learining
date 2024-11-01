#include <iostream>
#include "utils.hpp"


__global__ void matrix_mul_kernel(float *M_device, float *N_device, float *P_device, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < width) {
        float element_sum = 0.0;
        for (int k = 0; k < width; k++) {
            float a = M_device[y * width + k];
            float b = N_device[k * width + x];
            element_sum += a * b;
        }
        P_device[y * width + x] = element_sum;
    }
}


void matrix_mul_gpu(float* M_host, float* N_host, float* P_host, int width, int blk_size) {
    float *M_device, *N_device, *P_device;   // device上的输入矩阵和输出矩阵

    // 申请device上的内存
    cudaMalloc((void**)&M_device, width * width * sizeof(float));
    cudaMalloc((void**)&N_device, width * width * sizeof(float));
    cudaMalloc((void**)&P_device, width * width * sizeof(float));

    // 将host上的输入矩阵拷贝到device上
    cudaMemcpy(M_device, M_host, width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N_device, N_host, width * width * sizeof(float), cudaMemcpyHostToDevice);

    // 调用kernel函数
#if 0
    if (blk_size*blk_size > 1024){
        std::cout << "The total blk_size is " << "" << blk_size*blk_size << ", which is too large, please set it to 1024 or less." << std::endl;
        return;
    }
#endif
    dim3 block_size(blk_size, blk_size);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (width + block_size.y - 1) / block_size.y);
    LAST_KERNEL_CHECK();
    matrix_mul_kernel<<<grid_size, block_size>>>(M_device, N_device, P_device, width);
    LAST_KERNEL_CHECK();                  // 检查同步错误，即内核函数matrix_mul_kernel()调用时刻的错误
    CUDACHECK(cudaDeviceSynchronize());   // 检查异步错误，即内核函数matrix_mul_kernel()执行过程中的错误
    LAST_KERNEL_CHECK();   


    // 将device上的输出矩阵拷贝到host上
    CUDACHECK(cudaMemcpy(P_host, P_device, width * width * sizeof(float), cudaMemcpyDeviceToHost));
    LAST_KERNEL_CHECK();

    // 释放device上的内存
    cudaFree(M_device);
    cudaFree(N_device); 
    cudaFree(P_device);

}


// int main() {
//     // 申请host上的内存
//     int width = 4;   // 矩阵的宽度
//     float* M_host = (float*)malloc(width * width * sizeof(float));
//     float* N_host = (float*)malloc(width * width * sizeof(float));
//     float* P_host = (float*)malloc(width * width * sizeof(float));

//     // 随机生成输入矩阵
//     int seed = 1;
//     init_matrix(M_host, width, width, 1, 5);
//     init_matrix(N_host, width, width, 1, 5);

//     // gpu计算
//     matrix_mul_gpu(M_host, N_host, P_host, width);

//     // 输出结果
//     std::cout << "Matrix P from GPU:" << std::endl;
//     print_matrix(P_host, width, width);

//     // 释放host上的内存
//     free(M_host);
//     free(N_host);
//     free(P_host);

//     return 0;
// }