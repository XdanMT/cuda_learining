#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
// #include <cuda_runtime.h>

// #define CUDACHECK(call) {                                                  \
//     cudaError_t error = call;                                              \
//     if (error != cudaSuccess) {                                            \
//         printf("ERROR: %s:%d, ", __FILE__, __LINE__);                      \
//         printf("CODE:%d, DETAIL:%s\n", error, cudaGetErrorString(error));  \
//         exit(1);                                                           \
//     }                                                                      \
// }

#define CUDACHECK(call)                                                    \
do                                                                         \
{                                                                          \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                            \
        printf("ERROR: %s:%d, ", __FILE__, __LINE__);                      \
        printf("CODE:%d, DETAIL:%s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                           \
    }                                                                      \
} while(0)

#define PEEK_LAST_KERNEL_CHECK() {                                                       \
    cudaError_t err = cudaPeekAtLastError();                                             \
    if (err != cudaSuccess) {                                                            \
        printf("PEEK_LAST_KERNEL_CHECK ERROR: %s:%d, ", __FILE__, __LINE__);             \
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));  \
        exit(1);                                                                         \
    }                                                                                    \   
}   

#define LAST_KERNEL_CHECK() {                                                            \
    cudaError_t err = cudaGetLastError();                                                \
    if (err != cudaSuccess) {                                                            \
        printf("LAST_KERNEL_CHECK ERROR: %s:%d, ", __FILE__, __LINE__);                  \
        printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));  \
        exit(1);                                                                         \
    }                                                                                    \   
}



// #define CUDA_CHECK(call)             __cudaCheck(call, __FILE__, __LINE__)
// #define LAST_KERNEL_CHECK()          __kernelCheck(__FILE__, __LINE__)
// #define BLOCKSIZE 16

// inline static void __cudaCheck(cudaError_t err, const char* file, const int line) {
//     if (err != cudaSuccess) {
//         printf("ERROR: %s:%d, ", file, line);
//         printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
//         exit(1);
//     }
// }

// inline static void __kernelCheck(const char* file, const int line) {
//     /* 
//      * 在编写CUDA是，错误排查非常重要，默认的cuda runtime API中的函数都会返回cudaError_t类型的结果，
//      * 但是在写kernel函数的时候，需要通过cudaPeekAtLastError或者cudaGetLastError来获取错误
//      */
//     cudaError_t err = cudaPeekAtLastError();
//     if (err != cudaSuccess) {
//         printf("ERROR: %s:%d, ", file, line);
//         printf("CODE:%s, DETAIL:%s\n", cudaGetErrorName(err), cudaGetErrorString(err));
//         exit(1);
//     }
// }


void init_matrix_int(int* A, int m, int n, int min_val, int max_val, int seed);          // 初始化介于a, b之间的随机整数
void init_matrix_double(double* A, int m, int n, float min_val, float max_val, int seed);  // 初始化介于a, b之间的随机浮点数
void print_matrix(double* A, int m, int n);    // 打印矩阵
void compare_matrix(double* A, double* B, int m, int n);    // 比较两个矩阵是否相同

#endif