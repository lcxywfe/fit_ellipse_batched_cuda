#include <cuda.h>
#include <cuda_runtime.h>


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

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

void checkCUDAErrorFn(const char* msg, const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

