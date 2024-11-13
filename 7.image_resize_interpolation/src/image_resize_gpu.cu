#include <iostream>


__global__ void resize_bilinear_kernel_top(uint8_t *d_src, uint8_t *d_tar, int tar_h, int tar_w, int src_h, int src_w, float scaled_h, float scaled_w){
    /**
     * d_src 是原图的数据，d_tar是空的
    */

    // 1. 计算当前的kernel的坐标，代表tar的thread的坐标
    int tarX = blockDim.x * blockIdx.x + threadIdx.x;
    int tarY = blockDim.y * blockIdx.y + threadIdx.y;


    // 2. 将tar的坐标映射到src中去，得到离散的坐标
    float srcX = ((float)tarX + 0.5) * scaled_w - 0.5;    // 这一步的原理请查看有道云笔记
    float srcY = ((float)tarY + 0.5) * scaled_h - 0.5;

    // 3. 计算与离散的坐标相邻的4个点的坐标，并计算出局部偏移
    int srcX_low = floor(srcX);
    int srcX_high = srcX_low + 1;
    int srcY_low = floor(srcY);
    int srcY_high = srcY_low + 1;
    float delta_w = srcX - srcX_low;  // 局部偏移
    float delta_h = srcY - srcY_low;

    if (srcX_low < 0 || srcX_low > (src_w - 1) || srcY_low < 0 || srcY_low > (src_h - 1))
        return;

    // 4. 通过双线性差值的公式计算得到加权参数，用a_xy来表示，计算方法是求对角的矩形的面积
    float a_11 = (1 - delta_w) * (1 - delta_h);
    float a_12 = (1 - delta_w) * delta_h;
    float a_21 = delta_h * (1 - delta_h);
    float a_22 = delta_h * delta_w;

    // 5. 在d_src数据中取出数据，用加权参数计算加权后的像素值，得到目标的像素值


    // 6. 将结果存放到d_tar内存中，可以在赋值的这一步骤将BGR>>RGB的操作一起完成
    // ** d_src的layout是 NHWC，按照从后往前展开的规则，RGB三个元素在原图中的位置是连续的
    int stride = 3;
    int src_idx_11 = (srcY_low * src_w + srcX_low) * stride;
    int src_idx_12 = (srcY_high * src_w + srcX_low) * stride;
    int src_idx_21 = (srcY_low * src_w + srcX_high) * stride;
    int src_idx_22 = (srcY_high * src_w + srcX_high) * stride;
    int tar_idx = (tarY * tar_w + tarX) * stride;

    d_tar[tar_idx + 0] = round(d_src[src_idx_11 + 2] * a_11 + 
                               d_src[src_idx_12 + 2] * a_12 + 
                               d_src[src_idx_21 + 2] * a_21 + 
                               d_src[src_idx_22 + 2] * a_22);

    d_tar[tar_idx + 1] = round(d_src[src_idx_11 + 1] * a_11 + 
                               d_src[src_idx_12 + 1] * a_12 + 
                               d_src[src_idx_21 + 1] * a_21 + 
                               d_src[src_idx_22 + 1] * a_22);

    d_tar[tar_idx + 2] = round(d_src[src_idx_11 + 0] * a_11 + 
                               d_src[src_idx_12 + 0] * a_12 + 
                               d_src[src_idx_21 + 0] * a_21 + 
                               d_src[src_idx_22 + 0] * a_22);
}


__global__ void resize_bilinear_kernel_center(uint8_t *d_src, uint8_t *d_tar, int tar_h, int tar_w, int src_h, int src_w, float scaled_h, float scaled_w){
    /**
     * d_src 是原图的数据，d_tar是空的
    */

    // 1. 计算当前的kernel的坐标，代表tar的thread的坐标
    int tarX = blockDim.x * blockIdx.x + threadIdx.x;
    int tarY = blockDim.y * blockIdx.y + threadIdx.y;


    // 2. 将tar的坐标映射到src中去，得到离散的坐标
    float srcX = ((float)tarX + 0.5) * scaled_w - 0.5;    // 这一步的原理请查看有道云笔记
    float srcY = ((float)tarY + 0.5) * scaled_h - 0.5;

    // 3. 计算与离散的坐标相邻的4个点的坐标，并计算出局部偏移
    int srcX_low = floor(srcX);
    int srcX_high = srcX_low + 1;
    int srcY_low = floor(srcY);
    int srcY_high = srcY_low + 1;
    float delta_w = srcX - srcX_low;  // 局部偏移
    float delta_h = srcY - srcY_low;

    if (srcX_low < 0 || srcX_low > (src_w - 1) || srcY_low < 0 || srcY_low > (src_h - 1))
        return;

    // 4. 通过双线性差值的公式计算得到加权参数，用a_xy来表示，计算方法是求对角的矩形的面积
    float a_11 = (1 - delta_w) * (1 - delta_h);
    float a_12 = (1 - delta_w) * delta_h;
    float a_21 = delta_h * (1 - delta_h);
    float a_22 = delta_h * delta_w;

    // 5. 在d_src数据中取出数据，用加权参数计算加权后的像素值，得到目标的像素值


    // 6. 将结果存放到d_tar内存中，可以在赋值的这一步骤将BGR>>RGB的操作一起完成
    // ** d_src的layout是 NHWC，按照从后往前展开的规则，RGB三个元素在原图中的位置是连续的
    int stride = 3;
    int src_idx_11 = (srcY_low * src_w + srcX_low) * stride;
    int src_idx_12 = (srcY_high * src_w + srcX_low) * stride;
    int src_idx_21 = (srcY_low * src_w + srcX_high) * stride;
    int src_idx_22 = (srcY_high * src_w + srcX_high) * stride;

    // 7. 将目标图中的坐标位置从top位置调整到center为止 ，这是和top的kernel中不同之处
    tarY = tarY - int((src_h / scaled_h) / 2) + int(tar_h / 2);
    tarX = tarX - int((src_w / scaled_w) / 2) + int(tar_w / 2);
    int tar_idx = (tarY * tar_w + tarX) * stride;

    d_tar[tar_idx + 0] = round(d_src[src_idx_11 + 2] * a_11 + 
                               d_src[src_idx_12 + 2] * a_12 + 
                               d_src[src_idx_21 + 2] * a_21 + 
                               d_src[src_idx_22 + 2] * a_22);

    d_tar[tar_idx + 1] = round(d_src[src_idx_11 + 1] * a_11 + 
                               d_src[src_idx_12 + 1] * a_12 + 
                               d_src[src_idx_21 + 1] * a_21 + 
                               d_src[src_idx_22 + 1] * a_22);

    d_tar[tar_idx + 2] = round(d_src[src_idx_11 + 0] * a_11 + 
                               d_src[src_idx_12 + 0] * a_12 + 
                               d_src[src_idx_21 + 0] * a_21 + 
                               d_src[src_idx_22 + 0] * a_22);
}


void resize_bilinear_letterbox_gpu(uint8_t *device_src, uint8_t *device_tar, int tar_h, int tar_w, int src_h, int src_w, const std::string &letterbox_type){
    // 1. 设置grid和block大小
    int gridSize = 16;
    dim3 grid_dim(gridSize, gridSize, 1);
    dim3 block_dim((tar_h + gridSize - 1) / gridSize, (tar_h + gridSize - 1) / gridSize);

    // 2. 计算scale
    float scaled_h = src_h / tar_h;
    float scaled_w = src_w / tar_w;
    float scaled = (scaled_h > scaled_w ? scaled_h : scaled_w);
    scaled_h = scaled;
    scaled_w = scaled;

    // 3. kernel执行
    if (letterbox_type == "top")
        resize_bilinear_kernel_top <<<grid_dim, block_dim>>> (device_src, device_tar, tar_h, tar_w, src_h, src_w, scaled_h, scaled_w);
    else if (letterbox_type == "center")
    {
        resize_bilinear_kernel_center <<<grid_dim, block_dim>>> (device_src, device_tar, tar_h, tar_w, src_h, src_w, scaled_h, scaled_w);
    }
    
}