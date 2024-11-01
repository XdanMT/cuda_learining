#include <iostream>

#include "image_preprocess.hpp"
#include "opencv2/opencv.hpp"
#include "timer.hpp"





void image_preprocess_cpu(cv::Mat &input, cv::Mat &output, const int &tar_h, const int &tar_w, Timer &timer){
    timer.start_cpu();
    cv::cvtColor(input, input, cv::COLOR_BGR2RGB);   // BGR >> RGB
    cv::resize(input, output, cv::Size(tar_w, tar_h), 0, 0, cv::INTER_LINEAR);   // resize
    timer.stop_cpu();
    timer.duration_cpu("Preprocess image in cpu");
}


void image_preprocess_gpu(cv::Mat input, cv::Mat output, int tar_h, int tar_w, Timer timer){

}