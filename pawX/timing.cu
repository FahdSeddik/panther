#include "timing.h"
// barrier of reordering
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace detail {

/**
 * @brief Constructs a ScopedTimer object for timing code execution on CPU or CUDA devices.
 *
 * This constructor initializes timing based on the device type of the provided tensor.
 * If the tensor is on CUDA, it sets up CUDA events and records the start event on the current stream.
 * If the tensor is on CPU, it records the start time using a high-resolution clock.
 *
 * @param ref  A reference tensor used to determine the device (CPU or CUDA) for timing.
 * @param tag_ A string tag for identifying the timer instance (useful for logging or profiling).
 */
ScopedTimer::ScopedTimer(const torch::Tensor& ref, const char* tag_)
    : is_cuda(ref.is_cuda()), tag(tag_), stream(nullptr), gpu_start(nullptr), gpu_end(nullptr) {
    if (is_cuda) {
        // Grab the current CUDA stream from PyTorch
        stream = at::cuda::getCurrentCUDAStream().stream();
        // Create events
        cudaEventCreate(&gpu_start);
        cudaEventCreate(&gpu_end);
        // Record start on the current stream
        cudaEventRecord(gpu_start, stream);
    } else {
        cpu_start = std::chrono::high_resolution_clock::now();
    }
}

/**
 * @brief Destructor for the ScopedTimer class.
 *
 * This destructor measures and reports the elapsed time for a scoped code region,
 * either on the GPU (using CUDA events) or on the CPU (using std::chrono).
 *
 * - If `is_cuda` is true, it records the end CUDA event, synchronizes, calculates
 *   the elapsed time in milliseconds, prints the result, and destroys the CUDA events.
 * - If `is_cuda` is false, it measures the elapsed time using high-resolution CPU timers,
 *   prints the result in milliseconds.
 *
 * The timing result is printed to std::cout with the associated tag and whether the
 * measurement was on the GPU or CPU.
 */
ScopedTimer::~ScopedTimer() {
    if (is_cuda) {
        // Record end and synchronize
        cudaEventRecord(gpu_end, stream);
        cudaEventSynchronize(gpu_end);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, gpu_start, gpu_end);

        std::cout << tag << " took " << ms << " ms (GPU)\n";

        // Clean up
        cudaEventDestroy(gpu_start);
        cudaEventDestroy(gpu_end);
    } else {
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        std::cout << tag << " took " << ms << " ms (CPU)\n";
    }
}
}  // namespace detail