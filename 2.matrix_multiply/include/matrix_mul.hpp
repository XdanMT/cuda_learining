#ifndef MATMUL_HPP
#define MATMUL_HPP

void matrix_mul_cpu(double* M, double* N, double* P, int width);
void matrix_mul_gpu(double* M, double* N, double* P, int width, int blk_size);

#endif // MATMUL_HPP