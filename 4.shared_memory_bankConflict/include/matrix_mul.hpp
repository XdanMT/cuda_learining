#ifndef MATMUL_HPP
#define MATMUL_HPP

void matrix_mul_cpu(float* M_host, float* N_host, float* P_host, int width);
void matrix_mul_gpu(float* M_host, float* N_host, float* P_host, int width, int blk_size);
void matrix_mul_gpu_shared(float* M_host, float* N_host, float* P_host, int width, int blk_size, bool use_static_shared_memory);
void matrix_mul_gpu_shared_with_bank_conflict(float* M_host, float* N_host, float* P_host, int width, int blk_size, bool use_static_shared_memory);
void matrix_mul_gpu_shared_with_conflict_pad(float* M_host, float* N_host, float* P_host, int width, int blk_size, bool use_static_shared_memory);

#endif // MATMUL_HPP