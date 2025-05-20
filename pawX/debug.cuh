#pragma once

#define CHECK_CUDA(call)                                                                                       \
    do {                                                                                                       \
        cudaError_t err = call;                                                                                \
        if (err != cudaSuccess) {                                                                              \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                \
        }                                                                                                      \
    } while (0)

#define CHECK_CUDA_LAUNCH(call)                                                                                \
    do {                                                                                                       \
        call;                                                                                                  \
        cudaError_t err = cudaGetLastError();                                                                  \
        if (err != cudaSuccess) {                                                                              \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                \
        }                                                                                                      \
    } while (0)
