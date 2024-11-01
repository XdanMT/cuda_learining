#include <iostream>
#include <thread>  // 包含用于休眠的头文件  
#include <chrono>  // 包含时间库的头文件  


#define CUDACHECK(call)                                                    \
do                                                                         \
{                                                                          \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                            \
        printf("ERROR: %s:%d, ", __FILE__, __LINE__);                      \
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(error), cudaGetErrorString(error));  \
    }                                                                      \
} while(0)

#define PEEK_LAST_KERNEL_CHECK() {                                                       \
    cudaError_t err = cudaPeekAtLastError();                                             \
    if (err != cudaSuccess) {                                                            \
        printf("PEEK_LAST_KERNEL_CHECK ERROR: %s:%d, ", __FILE__, __LINE__);             \
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));  \
    }                                                                                    \   
}   

#define LAST_KERNEL_CHECK() {                                                            \
    cudaError_t err = cudaGetLastError();                                                \
    if (err != cudaSuccess) {                                                            \
        printf("LAST_KERNEL_CHECK ERROR: %s:%d, ", __FILE__, __LINE__);                  \
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));  \
    }                                                                                    \   
}


__global__ void print_dim_kernel() {
    printf("BlockIdx(%d, %d, %d), ThreadIdx(%d, %d, %d), BlockDim(%d, %d, %d), GridDim(%d, %d, %d)\n",
    blockIdx.x, blockIdx.y, blockIdx.z,threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

void print_one_dim() {
    int data_len = 8;
    int block_size = 4;
    int grid_size = (data_len + block_size - 1) / block_size;
    dim3 block(block_size);
    dim3 grid(grid_size);
    print_dim_kernel<<<grid_size, block_size>>>();
}

// 尝试访问非法内存
__global__ void kernel() {
    printf("kernel start\n");
    int* ptr = nullptr;
    *ptr = 0; // 尝试访问非法内存
}


int main() {
    print_one_dim();

    // cudaDeviceSynchronize();         // 等待所有线程完成，再执行后续代码
    printf("main() end\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));     // 如果没有cudaDeviceSynchronize()，这里可以添加延迟，延迟期间，GPU可以打印信息

    float* p;
    CUDACHECK(cudaMalloc(&p, 1000000000000000 * sizeof(float)));
    PEEK_LAST_KERNEL_CHECK();                    // 捕获到错误1，不清除
    kernel<<<1, 1>>>();  // 尝试访问非法内存，会导致程序崩溃
    PEEK_LAST_KERNEL_CHECK();
    CUDACHECK(cudaDeviceSynchronize());
    PEEK_LAST_KERNEL_CHECK();
    LAST_KERNEL_CHECK();  
    LAST_KERNEL_CHECK();  

    return 0;
}


// 当前机器维RTX3060 laptop当前代码的编译命令为：
// nvcc -o print_test print_test.cu -arch=sm_86