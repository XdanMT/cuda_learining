#include "utils.hpp"
#include <iostream>
#include "matrix_mul.hpp"
#include "timer.hpp"
#include "sleep_kernel.hpp"
#include "cuda_runtime.h"

int seed;

void check_overlap_support(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    if (!prop.deviceOverlap) {
        std::cout << "device does not support overlap!!" <<std::endl;
    } else {
        std::cout << "device supports overlap!" << std::endl;
    }
}




int main() {
    // 需要先确认自己的GPU是否支持overlap计算
    check_overlap_support();

    std::cout << "------------------------- preparation -------------------------" << std::endl;
    Timer timer;
    timer.start_cpu();

    // 申请host上的内存
    int width = 1 << 10;   // 1024
    int block_size;     // 块大小
    int count = 5;     // 决定了stream的个数

    // 分配主机端的内存，由于是使用stream加速方式，因此需要申请锁页内存的空间
    bool use_shared_memory;    
    int size = width * width * sizeof(float);
    float* M_host;
    float* N_host;
    float* P_by_device;
    cudaMallocHost(&M_host, size * sizeof(float));   // 这里只能通过这种方式来申请锁页内存
    cudaMallocHost(&N_host, size * sizeof(float));   // 虽然成功运行，但是很有可能造成多流计算的时候无法并行化 
    cudaMallocHost(&P_by_device, size * sizeof(float)); 

    // 随机生成输入矩阵
    seed = 1;
    init_matrix_float(M_host, width, width, 1, 5, seed);
    init_matrix_float(N_host, width, width, 1, 5, seed);
    timer.stop_cpu();
    timer.duration_cpu("init duration(host)");

    // gpu计算 --- warmup
    std::cout << "--------------------------- warmup ---------------------------" << std::endl;
    block_size = 1;
    timer.start_gpu();
    matrix_mul_gpu(M_host, N_host, P_by_device, width, block_size);
    timer.stop_gpu();
    timer.duration_gpu("matmul in gpu(warmup)");

    // run single stream  --- general
    std::cout << "--------------------------- bs = 16 ---------------------------" << std::endl;
    block_size = 16;
    timer.start_gpu();
    SleepSingleStream(M_host, width, block_size, count);
    timer.stop_gpu();
    timer.duration_gpu("run single stream(block size=16)");

    // run multi stream  --- general
    block_size = 16;
    timer.start_gpu();
    SleepMultiStream(M_host, width, block_size, count);
    timer.stop_gpu();
    timer.duration_gpu("run multi stream(block size=16)");

    // 释放host上的内存
    cudaFreeHost(M_host);
    cudaFreeHost(N_host);
    cudaFreeHost(P_by_device);
    
}