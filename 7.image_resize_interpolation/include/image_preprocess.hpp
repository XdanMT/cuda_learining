#ifndef IMAGE_PREPROCESS_HPP
#define IMAGE_PREPROCESS_HPP
#include "opencv2/opencv.hpp"
#include "timer.hpp"


void image_preprocess_cpu(cv::Mat &input, cv::Mat &output, const int &tar_h, const int &tar_w, Timer &timer);
void image_preprocess_gpu(cv::Mat input, cv::Mat output, int tar_h, int tar_w, Timer timer);



#endif