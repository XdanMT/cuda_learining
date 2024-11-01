#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <iostream>

class Timer {
public:
    Timer() {};

public:
    void start() {mStart = std::chrono::high_resolution_clock::now();}      // get start timepoint
    void stop() {mEnd = std::chrono::high_resolution_clock::now();}         // get end timepoint
    void duration(std::string msg);        // get running time duration

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
    std::chrono::time_point<std::chrono::high_resolution_clock> mEnd;

};


void Timer::duration(std::string msg){
    auto duration_value_nano = mEnd - mStart;
    auto duration_value_mili = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(duration_value_nano);
    std::cout << msg << " uses " << duration_value_mili.count() << " ms" << std::endl;
}


#endif