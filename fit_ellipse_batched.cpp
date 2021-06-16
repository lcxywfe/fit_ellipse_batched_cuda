#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <iostream>
#include <mutex>
#include <cublas_v2.h>
#include <cusolverDn.h>
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

class Timer {
public:
    Timer() { reset(); }
    void reset() {
        m_started = false;
        m_stopped = false;
    }
    void start() {
        assert(!m_started);
        m_started = true;
        m_start = std::chrono::high_resolution_clock::now();
    }
    void stop() {
        assert(m_started);
        assert(!m_stopped);
        m_stopped = true;
        m_stop = std::chrono::high_resolution_clock::now();
    }
    size_t get_time_in_ms() const {
        assert(m_stopped);
        return std::chrono::duration_cast<std::chrono::milliseconds>(m_stop -
                                                                     m_start)
                .count();
    }

private:
    using time_point = std::chrono::high_resolution_clock::time_point;
    time_point m_start, m_stop;
    bool m_started, m_stopped;
};

}

cusolverDnHandle_t BatchedEllipseFitter::cusolver_handle_ = NULL;
cublasHandle_t BatchedEllipseFitter::cublas_handle_ = NULL;
gesvdjInfo_t BatchedEllipseFitter::gesvdj_params_ = NULL;
bool BatchedEllipseFitter::inited_ = false;


void BatchedEllipseFitter::init() {
    static std::mutex mutex;
    mutex.lock();

    if (inited_) {
        mutex.unlock();
        return;
    }
    cusolverDnCreate(&cusolver_handle_);

    const float tol = 1.e-7;
    const int max_sweeps = 15;
    const int sort_svd = 1;

    cusolverDnCreateGesvdjInfo(&gesvdj_params_);
    checkCUDAError("Could not create GesvdjInfo");
    cusolverDnXgesvdjSetTolerance(gesvdj_params_, tol);
    checkCUDAError("Could not SetTolerance");
    cusolverDnXgesvdjSetMaxSweeps(gesvdj_params_, max_sweeps);
    checkCUDAError("Could not SetMaxSweeps");
    cusolverDnXgesvdjSetSortEig(gesvdj_params_, sort_svd);
    checkCUDAError("Could not SetSortEigs");

    cublasCreate(&cublas_handle_);

    inited_ = true;
    mutex.unlock();
}


BatchedEllipseFitter::~BatchedEllipseFitter() {
    cusolverDnDestroy(cusolver_handle_);
    cusolverDnDestroyGesvdjInfo(gesvdj_params_);
    cublasDestroy(cublas_handle_);
}


void BatchedEllipseFitter::svd_with_col_major_input(float* src, float* U,
                                                    float* S, float* V, int m,
                                                    int n, const int batch_size,
                                                    int* d_info) {
    Timer timer;
    timer.start();
    CHECK(m < 33 && n < 33);
    const int lda = m;
    const int ldu = m;
    const int ldv = n;
    const int minmn = (m < n) ? m : n;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    int lwork = 0;        /* size of workspace */
    float* d_work = NULL; /* device workspace for gesvdjBatched */

    cudaDeviceSynchronize();
    cusolverDnSgesvdjBatched_bufferSize(cusolver_handle_, jobz, m, n, src, lda,
                                        S, U, ldu, V, ldv, &lwork,
                                        gesvdj_params_, batch_size);
    cudaDeviceSynchronize();
    timer.stop();
    std::cout << "time1.6.1: " << timer.get_time_in_ms() << std::endl;
    timer.reset();
    timer.start();

    checkCUDAError("Could not SgesvdjBatched_bufferSize");
    cudaSafeCall(cudaMalloc((void**)&d_work, sizeof(float) * lwork));
    cudaDeviceSynchronize();
    cusolverDnSgesvdjBatched(cusolver_handle_, jobz, m, n, src, lda, S, U, ldu,
                             V, ldv, d_work, lwork, d_info, gesvdj_params_,
                             batch_size);
    checkCUDAError("SgesvdjBatched failed");
    cudaDeviceSynchronize();
    cudaDeviceSynchronize();
    timer.stop();
    std::cout << "time1.6.2: " << timer.get_time_in_ms() << std::endl;
    timer.reset();
    timer.start();


    if (d_work) {
        cudaSafeCall(cudaFree(d_work));
    }
}

void BatchedEllipseFitter::solve(float* d_A, float* d_b, float* d_x,
                                 int x_num, int batch_size, int sample_size) {
    Timer timer;
    timer.start();
    float *d_u, *d_v, *d_s;
    cudaSafeCall(cudaMalloc((void**)&d_u, sample_size * sample_size *
                                                  batch_size * sizeof(float)));
    cudaSafeCall(cudaMalloc((void**)&d_v,
                            x_num * x_num * batch_size * sizeof(float)));
    cudaSafeCall(cudaMalloc((void**)&d_s, x_num * batch_size * sizeof(float)));
    int* d_info = NULL;
    cudaSafeCall(cudaMalloc((void**)&d_info, batch_size * sizeof(int)));
    cudaDeviceSynchronize();
    timer.stop();
    std::cout << "time1.5.1: " << timer.get_time_in_ms() << std::endl;
    timer.reset();
    timer.start();


    svd_with_col_major_input(d_A, d_u, d_s, d_v, sample_size, x_num,
                             batch_size, d_info);
    cudaDeviceSynchronize();
    timer.stop();
    std::cout << "time1.5.2: " << timer.get_time_in_ms() << std::endl;
    timer.reset();
    timer.start();


    // 3. Calculate Result of equation X = Ut * (S^-1) * V * b
    const float alpha = 1.f;
    const float beta = 0.f;
    // Ut * b
    int m = x_num;
    int k = sample_size;
    int n = 1;
    int offset_u = 0;
    offset_u = k * k;  // When we use batched SVD API, offset_U = ldu * k = k*k,
                       // (Not m*k)
    float* d_ut_mul_b;
    cudaSafeCall(cudaMalloc((void**)&d_ut_mul_b,
                            m * n * batch_size * sizeof(float)));
    float* d_matA = d_u;
    float* d_matB = d_b;
    float* d_matC = d_ut_mul_b;

    cublasSgemmStridedBatched(cublas_handle_, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha,
                              d_matA, k, offset_u, d_matB, k, k * n, &beta,
                              d_matC, m, m * n, batch_size);
    cudaDeviceSynchronize();
    timer.stop();
    std::cout << "time1.5.3: " << timer.get_time_in_ms() << std::endl;
    timer.reset();
    timer.start();



    // (S^-1) * (Ut * b)
    kernels::element_wise_div(d_ut_mul_b, d_s, batch_size * x_num);
    cudaDeviceSynchronize();
    timer.stop();
    std::cout << "time1.5.4: " << timer.get_time_in_ms() << std::endl;
    timer.reset();
    timer.start();


    // V * (S^-1) * (Ut * b)
    m = x_num;
    k = x_num;
    n = 1;
    // d_matA = d_v;
    // d_matB = d_ut_mul_b;
    // d_matC = affine_mat;
    cublasSgemmStridedBatched(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                              &alpha, d_v, m, m * k, d_ut_mul_b, k, k * n,
                              &beta, d_x, m, m * n, batch_size);
    cudaDeviceSynchronize();
    timer.stop();
    std::cout << "time1.5.5: " << timer.get_time_in_ms() << std::endl;
    timer.reset();
    timer.start();


    if (d_u) {
        cudaSafeCall(cudaFree(d_u));
    }
    if (d_v) {
        cudaSafeCall(cudaFree(d_v));
    }
    if (d_s) {
        cudaSafeCall(cudaFree(d_s));
    }
    if (d_info) {
        cudaSafeCall(cudaFree(d_info));
    }
    if (d_ut_mul_b) {
        cudaSafeCall(cudaFree(d_ut_mul_b));
    }
}

std::vector<cv::RotatedRect> BatchedEllipseFitter::fit(
        std::vector<std::vector<cv::Point2f>> batched_points) {
    Timer timer;
    timer.start();
    CHECK(batched_points[0].size() <= 32);
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

    float* centers;
    cudaSafeCall(
            cudaMalloc((void**)(&centers), batch_size * 2 * sizeof(float)));
    cudaMemset((void*)centers, 0, batch_size * 2 * sizeof(float));

    kernels::get_centers(points, centers, batch_size, sample_size);
    // print_cuda(centers, batch_size * 2, 2);
    timer.stop();
    std::cout << "time0: " << timer.get_time_in_ms() << std::endl;
    timer.reset();
    timer.start();


    float* d_A;
    float* d_b;
    float* d_x;
    float* d_r;
    cudaSafeCall(cudaMalloc((void**)(&d_A),
                            batch_size * sample_size * 5 * sizeof(float)));
    cudaSafeCall(cudaMalloc((void**)(&d_b),
                            batch_size * sample_size * sizeof(float)));
    cudaSafeCall(cudaMalloc((void**)(&d_x), batch_size * 5 * sizeof(float)));
    cudaSafeCall(cudaMalloc((void**)(&d_r), batch_size * 2 * sizeof(float)));

    timer.stop();
    std::cout << "time1.1: " << timer.get_time_in_ms() << std::endl;
    timer.reset();
    timer.start();


    kernels::fill_param(points, centers, d_A, d_b, batch_size, sample_size);
    cudaDeviceSynchronize();
    timer.stop();
    std::cout << "time1.2.1: " << timer.get_time_in_ms() << std::endl;
    timer.reset();
    timer.start();

    solve(d_A, d_b, d_x, 5, batch_size, sample_size);
    cudaDeviceSynchronize();
    timer.stop();
    std::cout << "time1.2.2: " << timer.get_time_in_ms() << std::endl;
    timer.reset();
    timer.start();
    // print_cuda(d_x, batch_size * 5, 5);


    kernels::fill_param2(d_x, d_A, d_b, batch_size);
    cudaDeviceSynchronize();
    timer.stop();
    std::cout << "time1.3.1: " << timer.get_time_in_ms() << std::endl;
    timer.reset();
    timer.start();

    solve(d_A, d_b, d_r, 2, batch_size, 2);
    // print_cuda(d_r, batch_size * 2, 2);
    cudaDeviceSynchronize();
    timer.stop();
    std::cout << "time1.3.2: " << timer.get_time_in_ms() << std::endl;
    timer.reset();
    timer.start();


    kernels::fill_param3(points, centers, d_r, d_A, d_b, batch_size, sample_size);
    cudaDeviceSynchronize();
    timer.stop();
    std::cout << "time1.4.1: " << timer.get_time_in_ms() << std::endl;
    timer.reset();
    timer.start();

    solve(d_A, d_b, d_x, 3, batch_size, sample_size);
    // print_cuda(d_x, batch_size * 3, 3);

    cudaDeviceSynchronize();
    timer.stop();
    std::cout << "time1.4.2: " << timer.get_time_in_ms() << std::endl;
    timer.reset();
    timer.start();


    const float min_eps = 1e-8;

    std::vector<float> h_r(batch_size * 2);
    std::vector<float> h_x(batch_size * 3);
    std::vector<float> h_centers(batch_size * 2);
    cudaSafeCall(cudaMemcpy(h_r.data(), d_r, batch_size * 2 * sizeof(float),
                            cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(h_x.data(), d_x, batch_size * 3 * sizeof(float),
                            cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(h_centers.data(), centers,
                            batch_size * 2 * sizeof(float),
                            cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_r);
    cudaFree(centers);
    cudaFree(points);

    std::vector<cv::RotatedRect> boxes(0);
    for (int i = 0; i < batch_size; ++i) {
        float t,r0 = h_r[i * 2], r1 = h_r[i * 2 + 1], r2, r3, r4;
        float x0 = h_x[i * 3], x1 = h_x[i * 3 + 1], x2 = h_x[i * 3 + 2];

        r4 = -0.5 * atan2(x2, x1 - x0);
        if (fabs(x2) > min_eps )
            t = x2/sin(-2.0 * r4);
        else
            t = x1 - x0;
        r2 = fabs(x0 + x1 - t);
        if( r2 > min_eps )
            r2 = std::sqrt(2.0 / r2);
        r3 = fabs(x0 + x1 + t);
        if( r3 > min_eps )
            r3 = std::sqrt(2.0 / r3);

        cv::RotatedRect box;
        box.center.x = (float)r0 + h_centers[i * 2];
        box.center.y = (float)r1 + h_centers[i * 2 + 1];
        box.size.width = (float)(r2*2);
        box.size.height = (float)(r3*2);
        if( box.size.width > box.size.height )
        {
            float tmp;
            CV_SWAP( box.size.width, box.size.height, tmp );
            box.angle = (float)(90 + r4*180/CV_PI);
        }
        if( box.angle < -180 )
            box.angle += 360;
        if( box.angle > 360 )
            box.angle -= 360;
        // printf("box: %f %f %f %f %f\n", box.center.x, box.center.y, box.size.width, box.size.height, box.angle);
        boxes.push_back(box);
    }
    timer.stop();
    std::cout << "time2: " << timer.get_time_in_ms() << std::endl;
    timer.reset();
    timer.start();


    return boxes;
}
