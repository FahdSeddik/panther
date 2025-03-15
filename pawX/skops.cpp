#include "skops.h"

torch::Tensor scaled_sign_sketch(int64_t m, int64_t n, c10::optional<torch::Device> device, c10::optional<torch::Dtype> dtype) {
    // Set tensor options
    auto options = torch::TensorOptions().device(device.value_or(torch::kCPU)).dtype(dtype.value_or(torch::kFloat));

    // Generate {-1, 1} directly using sign function and scale by precomputed factor
    return torch::sign(torch::rand({m, n}, options) - 0.5) * (1.0 / std::sqrt(static_cast<double>(m)));
}
