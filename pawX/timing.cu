#include "timing.h"
// barrier of reordering
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace detail {

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