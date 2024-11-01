这个工程将原本使用chrono方式进行的运行时间计算，改成了使用cuda内部event方式的计算



#include <iostream>
#include <cuda_runtime.h>

__global__ void dummyKernel() {
    // This is a dummy kernel that just does some work
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = 0; i < 1000; ++i) {
        // Dummy computation
    }
}

int main() {
    cudaEvent_t start, stop;
    float milliseconds = 0;

    // Create CUDA events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Launch kernel (with some dummy workload)
    dummyKernel<<<10, 256>>>();

    // Record stop event
    cudaEventRecord(stop);

    // Synchronize events to ensure kernel completion
    cudaEventSynchronize(stop);

    // Calculate time elapsed
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution took: " << milliseconds << " ms\n";

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
