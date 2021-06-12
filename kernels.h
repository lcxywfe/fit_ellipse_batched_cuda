#pragma once
#include <cuda.h>
#include <cuda_runtime.h>


namespace kernels {

void get_centers(float* points, float* centers, int batch_size,
                 int sample_szie);

void fill_param(float* points, float* centers, double* A, double* b,
                int batch_size, int sample_szie);
}

