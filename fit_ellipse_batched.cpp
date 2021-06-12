#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <iostream>
#include "fit_ellipse_batched.h"
#include "kernels.h"
#include "common.h"

using namespace cv;


namespace {
template <typename T>
void print_cuda(T* data, int size, int stride) {
    std::vector<T>  d;
    d.resize(size);
    cudaMemcpy(d.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i) {
        if (i % stride == 0)
            printf("\n");
        std::cout << d[i] << " ";
    }
    printf("\n");
}
}


std::vector<cv::RotatedRect> fit_ellipse_batched(
        std::vector<std::vector<cv::Point2f>> batched_points) {
    CHECK(batched_points.size() <= 32);
    int batch_size = batched_points.size();
    int sample_size = batched_points[0].size();
    std::vector<float> points_data(0);
    for (int i = 0; i < batch_size; ++i) {
        CHECK(batched_points[i].size() == sample_size);
        for (int j = 0; j < sample_size; ++j) {
            points_data.push_back(batched_points[i][j].x);
            points_data.push_back(batched_points[i][j].y);
        }
    }
    float* points;
    cudaSafeCall(cudaMalloc((void**)(&points),
                            batch_size * sample_size * 2 * sizeof(float)));
    cudaMemcpy(points, points_data.data(),
               batch_size * sample_size * 2 * sizeof(float),
               cudaMemcpyHostToDevice);

    const double min_eps = 1e-8;

    float* centers;
    cudaSafeCall(
            cudaMalloc((void**)(&centers), batch_size * 2 * sizeof(float)));
    cudaMemset((void*)centers, 0, batch_size * 2 * sizeof(float));

    kernels::get_centers(points, centers, batch_size, sample_size);
    // print_cuda(centers, batch_size * 2, 2);

    double* A;
    double* b;
    double* x;
    cudaSafeCall(cudaMalloc((void**)(&A),
                            batch_size * sample_size * 5 * sizeof(double)));
    cudaSafeCall(cudaMalloc((void**)(&b),
                            batch_size * sample_size * sizeof(double)));
    cudaSafeCall(cudaMalloc((void**)(&x), batch_size * 5 * sizeof(double)));

    kernels::fill_param(points, centers, A, b, batch_size, sample_size);









    cudaFree(A);
    cudaFree(b);
    cudaFree(x);
    cudaFree(centers);
    cudaFree(points);
}
