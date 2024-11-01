#include <stdio.h>
#include <utils.hpp>

#define BLOCKSIZE 16

__global__ void matrix_mul_kernel_shared_static_with_conflict_pad(float *M_device, float *N_device, float *P_device, int width,int blk_size) {
    // pad操作，在分配的时候给共享矩阵最后一列后增加了一行空列，让原本conflict的位置错开，但是并不是所有的都错开了，只是得到了一定的缓解
    __shared__ float M_shared[BLOCKSIZE][BLOCKSIZE + 1];   
    __shared__ float N_shared[BLOCKSIZE][BLOCKSIZE + 1];

    // 在之前的方法的基础上，只要将tx和ty的索引交换位置，就产生了bank conflict
    // 通俗理解：一个block中有 blk_size*blk_size 个线程，现在让每个线程负责计算其对应位置的转置位置的运算，只需要交换block内部的局部索引即可
    int ty = threadIdx.x;     
    int tx = threadIdx.y;

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    float element_sum = 0.0;     // 定义一个变量来存放计算结果

    if (x < width && y < width) {
        for (int tile_idx = 0; tile_idx < width / blk_size; tile_idx++) {
            // printf("tile_num: %d\n", width / BLOCKSIZE);
            M_shared[ty][tx] = M_device[y * width + (tile_idx * blk_size + tx)];
            N_shared[ty][tx] = N_device[((tile_idx * blk_size + ty)) * width + x];
            __syncthreads();

            for (int k = 0; k < blk_size; k++){
                element_sum += M_shared[ty][k] * N_shared[k][tx];
            }
            __syncthreads();
        }
        P_device[y * width + x] = element_sum;
    }   
}


__global__ void matrix_mul_kernel_shared_dynamic_with_conflict_pad(float *M_device, float *N_device, float *P_device, int width,int blk_size) {
    /* 
        声明动态共享变量的时候需要加extern，同时需要是一维的 
        注意这里有个坑, 不能够像这样定义： 
            __shared__ float M_deviceShared[];
            __shared__ float N_deviceShared[];
        因为在cuda中定义动态共享变量的话，无论定义多少个他们的地址都是一样的。
        所以如果想要像上面这样使用的话，需要用两个指针分别指向shared memory的不同位置才行
    */
   
    //    这个变量实际上是在内核调用时由外部（例如，在主机代码中或在调用内核之前）分配的，而不是在内核函数内部自动分配的。
    extern __shared__ float shared_mem[];    // 注意动态的shared memory矩阵是一维的
    // 等价于在分配的时候给共享矩阵最后一列后增加了一行空列，让原本conflict的位置错开
    int shared_mem_stride = (blk_size + 1) * blk_size;

    int ty = threadIdx.x;     
    int tx = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    
    float element_sum = 0.0;     // 定义一个变量来存放计算结果
    if (x < width && y < width) {
        for (int tile_idx = 0; tile_idx < width / blk_size; tile_idx++) {
            // printf("tile_num: %d\n", width / BLOCKSIZE);
            // 动态的就将静态的两个shared_mem矩阵放在一起，然后展平，同一个索引对应的两个矩阵的元素的位置相差shared_mem_stride
            shared_mem[ty * (blk_size + 1) + tx] = M_device[y * width + (tile_idx * blk_size + tx)];
            shared_mem[(ty * (blk_size + 1) + tx) + shared_mem_stride] = N_device[((tile_idx * blk_size + ty)) * width + x];
            __syncthreads();

            for (int k = 0; k < blk_size; k++){
                element_sum += shared_mem[ty * (blk_size + 1) + k] * shared_mem[(k * (blk_size + 1) + tx) + shared_mem_stride];
            }
            __syncthreads();
        }
        P_device[y * width + x] = element_sum;
    }   
}


void matrix_mul_gpu_shared_with_conflict_pad(float* M_host, float* N_host, float* P_host, int width, int blk_size, bool use_static_shared_memory) {
    float *M_device, *N_device, *P_device;   // device上的输入矩阵和输出矩阵

    // 申请device上的内存
    cudaMalloc((void**)&M_device, width * width * sizeof(float));
    cudaMalloc((void**)&N_device, width * width * sizeof(float));
    cudaMalloc((void**)&P_device, width * width * sizeof(float));

    // 将host上的输入矩阵拷贝到device上
    cudaMemcpy(M_device, M_host, width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(N_device, N_host, width * width * sizeof(float), cudaMemcpyHostToDevice);

    // 调用kernel函数
    dim3 block_size(blk_size, blk_size);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, (width + block_size.y - 1) / block_size.y);
    LAST_KERNEL_CHECK();
    if (use_static_shared_memory){
        matrix_mul_kernel_shared_static_with_conflict_pad<<<grid_size, block_size>>>(M_device, N_device, P_device, width, blk_size);
    }
    else{
        matrix_mul_kernel_shared_dynamic_with_conflict_pad<<<grid_size, block_size, (blk_size + 1) * blk_size * sizeof(float)*2, nullptr>>>(M_device, N_device, P_device, width, blk_size);
    }
    

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