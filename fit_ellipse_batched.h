#include <opencv2/opencv.hpp>
#include <vector>
#include <cublas_v2.h>
#include <cusolverDn.h>

class BatchedEllipseFitter {
public:
    BatchedEllipseFitter();
    ~BatchedEllipseFitter();
    std::vector<cv::RotatedRect> fit(
            std::vector<std::vector<cv::Point2f>> batched_points);
private:
    void svd_with_col_major_input(double* src, double* U, double* S, double* V,
                                  int m, int n, const int batchSize,
                                  int* d_info);

    void solve(double* A, double* b, double* x, int x_num, int batch_size,
               int sample_size);

private:
    cusolverDnHandle_t cusolver_handle_;
    gesvdjInfo_t gesvdj_params_;
    cublasHandle_t cublas_handle_;

};
