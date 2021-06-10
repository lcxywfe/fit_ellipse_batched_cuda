#pragma once
#include <cuda.>
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



