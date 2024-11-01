# include <iostream>


void matrix_mul_cpu(double* M, double* N, double* P, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            double sum = 0;
            for (int k = 0; k < width; k++) {
                double a = M[i * width + k];
                double b = N[k * width + j];
                sum += a * b;
            }
            P[i * width + j] = sum;
        }
    }
}


// int main() {
//     int width = 4;
//     float *A = (float*)malloc(width * width * sizeof(float));
//     float *B = (float*)malloc(width * width * sizeof(float));
//     float *C = (float*)malloc(width * width * sizeof(float));
//     // initialize A and B randomly
//     int seed = 1;
//     init_matrix(A, width, width, 1, 5);
//     init_matrix(B, width, width, 1, 5);

//     // cpu计算
//     matrix_mul_cpu(A, B, C, width);

//     // 输出结果
//     std::cout << "Matrix P from CPU:" << std::endl;
//     print_matrix(C, width, width);

//     // 释放内存
//     free(A);
//     free(B);    
//     free(C);

//     return 0;
// }

