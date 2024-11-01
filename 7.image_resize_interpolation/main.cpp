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

    cv::Mat input = cv::imread(img_path, 1);   // 1表示“3 channel BGR color image”
    cv::Mat output;
    int tar_h = 500;
    int tar_w = 500;
    
    // cpu进行图像的resize操作
    image_preprocess_cpu(input, output, tar_h, tar_w, timer);
    cv::cvtColor(output, output, cv::COLOR_RGB2BGR);
    img_output_name = img_save_prefix + "preprocess_fox_cpu.png";
    cv::imwrite(img_output_name, output);

    // 


    return 0;
}
