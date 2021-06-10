#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "fit_ellipse_batched.h"
#include "kernels.h"

using namespace cv;

#ifndef cudaSafeCall
#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError_t err, const char* file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA error in file '%s', line %d,  error: %s\n", file,
                line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#endif

#ifndef cublasSafeCall
#define cublasSafeCall(err) __cublasSafeCall(err, __FILE__, __LINE__)
inline void __cublasSafeCall(cublasStatus_t err, const char* file,
                             const int line) {
    if (CUBLAS_STATUS_SUCCESS != err) {
        fprintf(stderr,
                "CUBLAS error in file '%s', line %d, error %d terminating!\n",
                file, line, err);
        exit(EXIT_FAILURE);
    }
}
#endif


std::vector<cv::RotatedRect> fit_ellipse_batched(
        std::vector<std::vector<cv::Point2f>> _batched_points) {
    CHECK(_batched_points.size() <= 32);
    std::vector<cv::Mat> batched_points(0);
    for (int i = 0; i < _batched_points.size(); ++i) {
        CHECK(_batched_points.size() == _batched_points[0].size());
        batched_points.emplace_back(_batched_points[i]);
    }
    int batch_size = batched_points.size();
    int sample_size = batched_points[0].checkVector(2);
    const double min_eps = 1e-8;

    float* centers;
    cudaSafeCall(
            cudaMalloc((void**)(&centers), batch_size * 2 * sizeof(float)));
    cudaMemset((void*)centers, 0, batch_size * 2 * sizeof(float));

    // for (int i = 0; i < batch_size; ++i) {
    //     for (int j = 0; j < sample_size; ++j) {
    //         Point2f p(float(batched_points[i].ptr<Point2f>()[j].x),
    //                   float(batched_points[i].ptr<Point2f>()[j].y));
    //         centers[i] += p;
    //     }
    //     centers[i].x /= sample_size;
    //     centers[i].y /= sample_size;
    // }
}
