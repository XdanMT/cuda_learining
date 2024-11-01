#include <iostream>
#include <random>


int main(int argc, char const *argv[])
{
    /* code */
    float min = 5, max = 5.5;
    float rdn_value;
    for (int i = 0; i < 10; i++) {
        rdn_value = (float)rand() / RAND_MAX * (max - min) + min;
        std::cout << rdn_value << " ";
    }
    // 输出int指针的长度
    std::cout << "\n";
    std::cout << sizeof(float) << std::endl;
    std::cout << sizeof(int) << std::endl;
    std::cout << sizeof(double) << std::endl;
    return 0;
}
