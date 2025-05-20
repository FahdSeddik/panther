#pragma once

#include <torch/extension.h>

#include <chrono>
#include <iostream>
#ifdef _MSC_VER
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
#else
typedef __device_builtin__ struct CUstream_st* cudaStream_t;
typedef __device_builtin__ struct CUevent_st* cudaEvent_t;
#endif

namespace detail {

struct ScopedTimer {
    bool is_cuda;
    const char* tag;

    // CPU timing
    std::chrono::high_resolution_clock::time_point cpu_start;

    // GPU timing
    cudaEvent_t gpu_start{nullptr}, gpu_end{nullptr};
    cudaStream_t stream{nullptr};

    ScopedTimer(const torch::Tensor& ref, const char* tag_);

    ~ScopedTimer();
};

}  // namespace detail