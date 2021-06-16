#include "kernels.h"


using namespace kernels;

#define CUDA_BLOCK_SIZE 256

namespace {

__global__ void get_centers_kernel(float* points, float* centers,
                                   int batch_size, int sample_size) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    while (batch < batch_size) {
        float x = 0, y = 0;
        for (int i = 0; i < sample_size; ++i) {
            x += points[batch * sample_size * 2 + i * 2];
            y += points[batch * sample_size * 2 + i * 2 + 1];
        }
        centers[batch * 2] = x / sample_size;
        centers[batch * 2 + 1] = y / sample_size;
        batch += gridDim.x * blockDim.x;
    }
}

__global__ void fill_param_kernel(float* points, float* centers, float* A,
                                  float* b, int batch_size, int sample_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < batch_size * sample_size) {
        int batch_id = index / sample_size;
        int sample_id = index - batch_id * sample_size;
        float x = points[index * 2] - centers[batch_id * 2];
        float y = points[index * 2 + 1] - centers[batch_id * 2 + 1];
        int sample_offset = batch_id * sample_size * 5 + sample_id;
        A[sample_offset] = -(float)x * (float)x;
        A[sample_offset + sample_size] = -(float)y * (float)y;
        A[sample_offset + sample_size * 2] = -(float)x * (float)y;
        A[sample_offset + sample_size * 3] = x;
        A[sample_offset + sample_size * 4] = y;
        b[index] = 10000.0;

        index += gridDim.x * blockDim.x;
    }
}

__global__ void fill_param_kernel_2(float* x, float* A, float* b,
                                    int batch_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < batch_size) {
        A[index * 4] = 2 * x[index * 5];
        A[index * 4 + 1] = A[index * 4 + 2] = x[index * 5 + 2];
        A[index * 4 + 3] = 2 * x[index * 5 + 1];
        b[index * 2] = x[index * 5 + 3];
        b[index * 2 + 1] = x[index * 5 + 4];

        index += gridDim.x * blockDim.x;
    }
}

__global__ void fill_param_kernel_3(float* points, float* centers, float* r,
                                    float* A, float* b, int batch_size,
                                    int sample_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < batch_size * sample_size) {
        int batch_id = index / sample_size;
        int sample_id = index - batch_id * sample_size;
        float x = points[index * 2] - centers[batch_id * 2];
        float y = points[index * 2 + 1] - centers[batch_id * 2 + 1];
        int sample_offset = batch_id * sample_size * 3 + sample_id;
        float r0 = r[batch_id * 2];
        float r1 = r[batch_id * 2 + 1];
        A[sample_offset] = (x - r0) * (x - r0);
        A[sample_offset + sample_size] = (y - r1) * (y - r1);
        A[sample_offset + sample_size * 2] = (x - r0) * (y - r1);
        b[index] = 1.0;

        index += gridDim.x * blockDim.x;
    }
}

__global__ void element_wise_div_kernel(float* A, float* B, int size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= size)
        return;
    float val = B[index];
    if (val == 0)
        A[index] = 0;
    else
        A[index] /= val;
}

}

void kernels::get_centers(float* points, float* centers, int batch_size,
                          int sample_size) {
    int grid = (batch_size + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    get_centers_kernel<<<grid, CUDA_BLOCK_SIZE>>>(points, centers, batch_size,
                                                  sample_size);
}

void kernels::fill_param(float* points, float* centers, float* A, float* b,
                         int batch_size, int sample_size) {
    int grid =
            (batch_size * sample_size + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    fill_param_kernel<<<grid, CUDA_BLOCK_SIZE>>>(points, centers, A, b,
                                                 batch_size, sample_size);
}


void kernels::fill_param2(float* x, float* A, float* b, int batch_size) {
    int grid = (batch_size + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    fill_param_kernel_2<<<grid, CUDA_BLOCK_SIZE>>> (x, A, b, batch_size);
}

void kernels::fill_param3(float* points, float* centers, float* r, float* A,
                          float* b, int batch_size, int sample_size) {
    int grid =
            (batch_size * sample_size + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    fill_param_kernel_3<<<grid, CUDA_BLOCK_SIZE>>>(points, centers, r, A, b,
                                                   batch_size, sample_size);
}

void kernels::element_wise_div(float* A, float* B, int size) {
    int grid = (size + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    element_wise_div_kernel<<<grid, CUDA_BLOCK_SIZE>>>(A, B, size);
}

