#ifndef TIMER_HPP
#define TIMER_HPP

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include "utils.hpp"


class Timer{
public:
    Timer(){
        _cStart = std::chrono::high_resolution_clock::now();
        _cEnd = std::chrono::high_resolution_clock::now();
        cudaEventCreate(&_gStart);
        cudaEventCreate(&_gEnd);
    }

    ~Timer(){
        cudaEventDestroy(_gStart);
        cudaEventDestroy(_gEnd);
    }

public:
    void start_cpu(){_cStart = std::chrono::high_resolution_clock::now();}

    void stop_cpu(){_cEnd = std::chrono::high_resolution_clock::now();}

    void start_gpu(){
        cudaEventRecord(_gStart, 0);   // 0表示stream_id
    }

    void stop_gpu(){
        cudaEventRecord(_gEnd, 0);
    }

    void duration_gpu(std::string msg){
        CUDACHECK(cudaEventSynchronize(_gStart));
        CUDACHECK(cudaEventSynchronize(_gEnd));
        cudaEventElapsedTime(&_timeElasped_milliseconds, _gStart, _gEnd);
        std::cout << msg << " uses " << _timeElasped_milliseconds << " ms" << std::endl;
    }

    void duration_cpu(std::string msg){
        auto duration_value_nano = _cEnd - _cStart;
        auto duration_value_mili = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(duration_value_nano);
        std::cout << msg << " uses " << duration_value_mili.count() << " ms" << std::endl;
    }


private:
    std::chrono::high_resolution_clock::time_point _cStart;
    std::chrono::high_resolution_clock::time_point _cEnd;
    cudaEvent_t _gStart;
    cudaEvent_t _gEnd;
    float _timeElasped_milliseconds;  // 表示GPU运行时间的变量
};







#endif // TIMER_HPP