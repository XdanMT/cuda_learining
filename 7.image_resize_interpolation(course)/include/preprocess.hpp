#ifndef __PREPROCESS_HPP__
#define __PREPROCESS_HPP__

#include "opencv2/opencv.hpp"
#include "timer.hpp"

cv::Mat preprocess_cpu(cv::Mat &src, const int& tarH, const int& tarW, Timer timer);
// 因为这里没有模板函数的具体实现，因此需要再preprocess.cpp文件中进行模板显示实例化
template<typename T> cv::Mat preprocess_gpu(cv::Mat &h_src, const int& tarH, const int& tarW, Timer timer);
template<typename T> void resize_bilinear_gpu(T* d_tar, uint8_t* d_src, int tarW, int tarH, int srcH, int srcW);
#endif //__PREPROCESS_HPP__
