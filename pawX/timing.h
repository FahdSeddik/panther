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

/**
 * @brief A scoped timer utility for measuring execution time on CPU or GPU.
 *
 * This struct provides automatic timing for a code block, supporting both CPU and CUDA GPU timing.
 * Upon construction, it starts the timer (CPU or GPU depending on the input tensor's device).
 * Upon destruction, it stops the timer and reports the elapsed time, optionally tagged with a label.
 *
 * @note Timing is performed using std::chrono for CPU and CUDA events for GPU.
 *
 * @param is_cuda Indicates whether the timer is measuring CUDA GPU time.
 * @param tag A string label to identify the timing scope.
 * @param cpu_start The starting time point for CPU timing.
 * @param gpu_start The CUDA event marking the start of GPU timing.
 * @param gpu_end The CUDA event marking the end of GPU timing.
 * @param stream The CUDA stream used for timing.
 *
 * @constructor ScopedTimer(const torch::Tensor& ref, const char* tag_)
 *   Initializes the timer based on the device of the provided tensor.
 *
 * @destructor ~ScopedTimer()
 *   Stops the timer and reports the elapsed time.
 */
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