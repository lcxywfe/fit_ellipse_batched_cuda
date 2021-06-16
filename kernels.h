#pragma once
#include <cuda.h>
#include <cuda_runtime.h>


namespace kernels {

void get_centers(float* points, float* centers, int batch_size,
                 int sample_szie);

void fill_param(float* points, float* centers, float* A, float* b,
                int batch_size, int sample_szie);

void element_wise_div(float* A, float* B, int size);

void fill_param2(float* x, float* A, float* b, int batch_size);

void fill_param3(float* points, float* centers, float* r, float* A, float* b,
                 int batch_size, int sample_size);

}  // namespace kernels
