#include <iostream>

#include "opencv2/opencv.hpp"
#include "cuda_runtime.h"
#include "timer.hpp"
#include "utils.hpp"
#include "image_preprocess.hpp"



int main(){
    Timer timer;
    std::string img_path = "./data/fox.png";
    std::string img_save_prefix = "./results/";
    std::string img_output_name = "";

    // 读到的图像的layout是 NHWC
    cv::Mat input = cv::imread(img_path, 1);   // 1表示“3 channel BGR color image”
    cv::imwrite("./results/fox_saved_by_opencv.png", input);
    cv::Mat output_cpu;
    int tar_h = 500;
    int tar_w = 500;
    cv::Mat output_gpu(cv::Size(tar_w, tar_h), CV_8UC3);
    
    // cpu进行图像的resize操作
    image_preprocess_cpu(input, output_cpu, tar_h, tar_w, timer);
    cv::cvtColor(output_cpu, output_cpu, cv::COLOR_RGB2BGR);
    img_output_name = img_save_prefix + "preprocess_fox_cpu.png";
    cv::imwrite(img_output_name, output_cpu);

    // gpu进行图像的letterbox的resize操作，位置top
    image_preprocess_gpu(input, output_gpu, tar_h, tar_w, timer, "top");
    timer.duration_cpu("gpu (warm up)");
    image_preprocess_gpu(input, output_gpu, tar_h, tar_w, timer, "top");
    timer.duration_cpu("Preprocess image letterbox_top resize in gpu");
    cv::cvtColor(output_gpu, output_gpu, cv::COLOR_RGB2BGR);
    img_output_name = img_save_prefix + "preprocess_fox_gpu_top.png";
    cv::imwrite(img_output_name, output_gpu);

    // gpu进行图像的letterbox的resize操作，位置top
    image_preprocess_gpu(input, output_gpu, tar_h, tar_w, timer, "center");
    timer.duration_cpu("Preprocess image letterbox_top resize in gpu");
    cv::cvtColor(output_gpu, output_gpu, cv::COLOR_RGB2BGR);
    img_output_name = img_save_prefix + "preprocess_fox_gpu_center.png";
    cv::imwrite(img_output_name, output_gpu);



    return 0;
}
