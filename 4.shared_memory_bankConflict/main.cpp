#include <iostream>
#include "utils.hpp"
#include "matrix_mul.hpp"
#include "timer.hpp"

int seed;

int main() {
    Timer timer;
    timer.start();

    // 申请host上的内存
    int width = 1 << 9;   // 1024
    int block_size;     // 块大小
    bool use_shared_memory;
    float* M_host = (float*)malloc(width * width * sizeof(float));
    float* N_host = (float*)malloc(width * width * sizeof(float));
    float* P_by_host = (float*)malloc(width * width * sizeof(float));
    float* P_by_device = (float*)malloc(width * width * sizeof(float));

    // 随机生成输入矩阵
    seed = 1;
    init_matrix_float(M_host, width, width, 1, 5, seed);
    init_matrix_float(N_host, width, width, 1, 5, seed);
    timer.stop();
    timer.duration("init duration(host)");

    // cpu计算 --- general
    timer.start();
    matrix_mul_cpu(M_host, N_host, P_by_host, width);
    timer.stop();
    timer.duration("matmul in cpu");

    // gpu计算 --- warmup
    std::cout << "--------------------------- warmup ---------------------------" << std::endl;
    block_size = 1;
    timer.start();
    matrix_mul_gpu(M_host, N_host, P_by_device, width, block_size);
    timer.stop();
    timer.duration("matmul in gpu(warmup)");

    // gpu计算 --- general
    std::cout << "--------------------------- bs = 16 ---------------------------" << std::endl;
    block_size = 16;
    timer.start();
    matrix_mul_gpu(M_host, N_host, P_by_device, width, block_size);
    timer.stop();
    timer.duration("matmul in gpu general(block size=16)");
    compare_matrix(P_by_host, P_by_device, width, width);

    // gpu计算 --- static shared memory
    timer.start();
    block_size = 16;
    use_shared_memory = true;
    matrix_mul_gpu_shared(M_host, N_host, P_by_device, width, block_size, use_shared_memory);
    timer.stop();
    timer.duration("matmul in gpu with static shared memory(block size=16)");
    compare_matrix(P_by_host, P_by_device, width, width);

    // gpu计算 --- static shared memory with bank conflict
    timer.start();
    block_size = 16;
    use_shared_memory = true;
    matrix_mul_gpu_shared_with_bank_conflict(M_host, N_host, P_by_device, width, block_size, use_shared_memory);
    timer.stop();
    timer.duration("matmul in gpu with static shared memory with bank conflict(block size=16)");
    compare_matrix(P_by_host, P_by_device, width, width);

    // gpu计算 --- static shared memory with bank conflict pad
    timer.start();
    block_size = 16;
    use_shared_memory = true;
    matrix_mul_gpu_shared_with_conflict_pad(M_host, N_host, P_by_device, width, block_size, use_shared_memory);
    timer.stop();
    timer.duration("matmul in gpu with static shared memory with bank conflict pad(block size=16)");
    compare_matrix(P_by_host, P_by_device, width, width);

    // gpu计算 --- dynamic shared memory 
    timer.start();
    block_size = 16;
    use_shared_memory = false;
    matrix_mul_gpu_shared(M_host, N_host, P_by_device, width, block_size, use_shared_memory);
    timer.stop();
    timer.duration("matmul in gpu with dynamic shared memory(block size=16)");
    compare_matrix(P_by_host, P_by_device, width, width);

    // gpu计算 --- dynamic shared memory with bank conflict
    timer.start();
    block_size = 16;
    use_shared_memory = false;
    matrix_mul_gpu_shared_with_bank_conflict(M_host, N_host, P_by_device, width, block_size, use_shared_memory);
    timer.stop();
    timer.duration("matmul in gpu with dynamic shared memory with bank conflict(block size=16)");
    compare_matrix(P_by_host, P_by_device, width, width);

    // gpu计算 --- dynamic shared memory with bank conflict pad
    timer.start();
    block_size = 16;
    use_shared_memory = false;
    matrix_mul_gpu_shared_with_conflict_pad(M_host, N_host, P_by_device, width, block_size, use_shared_memory);
    timer.stop();
    timer.duration("matmul in gpu with dynamic shared memory with bank conflict pad(block size=16)");
    compare_matrix(P_by_host, P_by_device, width, width);

    // gpu计算 --- general
    std::cout << "--------------------------- bs = 32 ---------------------------" << std::endl;
    block_size = 32;
    timer.start();
    matrix_mul_gpu(M_host, N_host, P_by_device, width, block_size);
    timer.stop();
    timer.duration("matmul in gpu general(block size=32)");
    compare_matrix(P_by_host, P_by_device, width, width);

    // gpu计算 --- dynamic shared memory 
    timer.start();
    block_size = 32;   // 不能超过32，因为32*32=1024，超过了一个block的最大线程数
    use_shared_memory = false;
    matrix_mul_gpu_shared(M_host, N_host, P_by_device, width, block_size, use_shared_memory);
    timer.stop();
    timer.duration("matmul in gpu with dynamic shared memory(block size=32)");
    compare_matrix(P_by_host, P_by_device, width, width);

    // gpu计算 --- dynamic shared memory with bank conflict
    block_size = 32;
    timer.start();
    matrix_mul_gpu_shared_with_bank_conflict(M_host, N_host, P_by_device, width, block_size, use_shared_memory);
    timer.stop();
    timer.duration("matmul in gpu with dynamic shared memory with bank conflict(block size=32)");
    compare_matrix(P_by_host, P_by_device, width, width);

    // gpu计算 --- dynamic shared memory with bank conflict pad
    timer.start();
    block_size = 32;
    use_shared_memory = false;
    matrix_mul_gpu_shared_with_conflict_pad(M_host, N_host, P_by_device, width, block_size, use_shared_memory);
    timer.stop();
    timer.duration("matmul in gpu with dynamic shared memory with bank conflict pad(block size=32)");
    compare_matrix(P_by_host, P_by_device, width, width);

    /* 
     * 注意，这里blockSize=64导致一个block中的thread数量超过了1024，最终使得kernel无法启动
     * 这个错误属于参数设定的错误。类似的错误比如说还有设置过大的shared_memory
     * 如果没有使用error handler进行错误排查的话是无法发现错误的
    */
    // gpu计算 --- general
    // timer.start();
    // block_size = 64;   // 不能超过32，因为32*32=1024，超过了一个block的最大线程数
    // matrix_mul_gpu(M_host, N_host, P_by_device, width, block_size);
    // timer.stop();
    // timer.duration("matmul in gpu (block size=64)");

    // 输出结果
    // std::cout << "Matrix P from CPU:" << std::endl;
    // print_matrix(P_by_host, width, width);
    // std::cout << "Matrix P from GPU:" << std::endl;
    // print_matrix(P_by_device, width, width);

    // 释放host上的内存
    free(M_host);
    free(N_host);
    free(P_by_host);
    free(P_by_device);
    
}