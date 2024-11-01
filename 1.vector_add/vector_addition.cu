// rector_addition.cu
#include <iostream>

__global__ void addVectors(int* a, int* b, int* c, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    const int size = 32;
    int a[size], b[size], c[size];
    int *d_a, *d_b, *d_c;

    // Allocate memory on device
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMalloc((void**)&d_c, size * sizeof(int));

    // Initialize arrays a and b, and copy them to device
    for (int i = 0; i < size; ++i) {
        a[i] = i;
        b[i] = i * 2;
    }
    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int blocksize = 1024;    // 设置超过1024时，会报错
    int girdsize = (size + blocksize - 1) / blocksize;
    addVectors<<<girdsize, blocksize>>>(d_a, d_b, d_c, size);

    // Copy result back to host and free device memory
    // Copy result back to host                
    cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < size; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
} 
