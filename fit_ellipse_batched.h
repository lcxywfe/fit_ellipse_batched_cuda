#include <opencv2/opencv.hpp>
#include <vector>
#include <cublas_v2.h>
#include <cusolverDn.h>

class BatchedEllipseFitter {
public:
    BatchedEllipseFitter() { init(); }
    ~BatchedEllipseFitter();
    std::vector<cv::RotatedRect> fit(
            std::vector<std::vector<cv::Point2f>> batched_points);
private:
    void svd_with_col_major_input(double* src, double* U, double* S, double* V,
                                  int m, int n, const int batchSize,
                                  int* d_info);

    void solve(double* A, double* b, double* x, int x_num, int batch_size,
               int sample_size);

    static void init();

private:
    static cusolverDnHandle_t cusolver_handle_;
    static cublasHandle_t cublas_handle_;
    static gesvdjInfo_t gesvdj_params_;
    static bool inited_;
};
