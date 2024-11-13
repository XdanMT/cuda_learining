#ifndef IMAGE_RESIZE_HPP
#define IMAGE_RESIZE_HPP

#include <iostream>


void resize_bilinear_letterbox_gpu(uint8_t *device_src, uint8_t *device_tar, int tar_h, int tar_w, int src_h, int src_w, const std::string &letterbox_type);


#endif