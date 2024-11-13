#include <iostream>
#include <cuda_runtime.h>

#include "image_preprocess.hpp"
#include "opencv2/opencv.hpp"
#include "timer.hpp"
#include "image_resize_gpu.hpp"





void image_preprocess_cpu(cv::Mat &input, cv::Mat &output, const int &tar_h, const int &tar_w, Timer &timer){
    timer.start_cpu();
    cv::cvtColor(input, output, cv::COLOR_BGR2RGB);   // BGR >> RGB
    cv::resize(output, output, cv::Size(tar_w, tar_h), 0, 0, cv::INTER_LINEAR);   // resize
    timer.stop_cpu();
    timer.duration_cpu("Preprocess image in cpu");
}


void image_preprocess_gpu(cv::Mat &host_src, cv::Mat &host_tar, const int &tar_h, const int &tar_w, Timer &timer, const std::string &letterbox_type){
    /**1.先写在CPU上做的事情：
        * 明确的数据类型（float32、uint8）创建 d_tar
        * 分配device端变量的内存d_src, d_tar
     * */ 

    // 准备输入数据，即host_src >> device_src
    uint8_t *device_src;
    int src_h = host_src.rows;
    int src_w = host_src.cols;
    int size_input = src_h * src_w * 3 * sizeof(uint8_t);  // 输入图像的元素个数
    cudaMalloc(&device_src, size_input);
    cudaMemcpy(device_src, host_src.data, size_input, cudaMemcpyHostToDevice);   // 注意 cv::Mat.data()是取了矩阵数据部分的指针

    // 准备输出数据，不需要拷贝内存，分配内存就可以了，这里假定输出的数据类型是uint8类型的
    uint8_t *device_tar;
    int size_output = tar_h * tar_w * 3 * sizeof(uint8_t);
    cudaMalloc(&device_tar, size_output);   // 这个device_tar在计算好之后，会将数据copy到host_tar变量中
    

    // 2. GPU处理部分，调用.cu文件中的接口即可
    timer.start_cpu();
    resize_bilinear_letterbox_gpu(device_src, device_tar, tar_h, tar_w, src_h, src_w, letterbox_type);
    timer.stop_cpu();

    
    // 3. 将结果拷贝到host_tar变量中
    cudaMemcpy(host_tar.data, device_tar, size_output, cudaMemcpyDeviceToHost);


    // 4. 释放空间
    cudaFree(device_src);
    cudaFree(device_tar);
}