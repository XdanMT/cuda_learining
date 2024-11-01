#include <iostream>
#include <random>


// 初始化介于a, b之间的随机整数
void init_matrix_float(float* A, int m, int n, float min_val, float max_val, int seed) {
    srand(seed);
    if (min_val > max_val) {
        std::swap(min_val, max_val);
    }

    for (int i = 0; i < m * n; i++) {
        A[i] = min_val + (float)rand() / RAND_MAX * (max_val - min_val);
    }
}

void init_matrix_int(int* A, int m, int n, int min_val, int max_val, int seed) {
    srand(seed);
    if (min_val > max_val) {
        std::swap(min_val, max_val);
    }

    for (int i = 0; i < m * n; i++) {
        A[i] = min_val + rand() % (max_val - min_val + 1);
    }
}


// 打印矩阵
void print_matrix(float* A, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << A[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << '\n';
}

// 比较两个矩阵是否相同
void compare_matrix(float* A, float* B, int m, int n){
    float precision = 1e-2;    // 误差在1e-2以内是可以接受的
    int count = 0;
    for (int i = 0; i < m * n; i++) {
        float error_value = std::abs(A[i] - B[i]);
        if (error_value > precision) {
            std::cout << "Matmul result is different" << std::endl;
            std::cout << "The different element are " << A[i] << " and " << B[i];
            std::cout << ", and the error value is " << error_value;
            std::cout << ", and the index is " << i << std::endl;

            if (count++ > 10)
            {
                return;
            }
            
            continue;
        }
    }
    std::cout << "Matrices are equal!\n" << std::endl;
}