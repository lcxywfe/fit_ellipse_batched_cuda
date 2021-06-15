#pragma once
#include <cuda.h>
#include <cuda_runtime.h>


namespace kernels {

void get_centers(float* points, float* centers, int batch_size,
                 int sample_szie);

void fill_param(float* points, float* centers, double* A, double* b,
                int batch_size, int sample_szie);

void element_wise_div(double* A, double* B, int size);

void fill_param2(double* x, double* A, double* b, int batch_size);

void fill_param3(float* points, float* centers, double* r, double* A, double* b,
                 int batch_size, int sample_size);

}  // namespace kernels
