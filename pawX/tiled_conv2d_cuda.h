#pragma once

#include <torch/extension.h>

#include <vector>

torch::Tensor tiled_conv2d_cuda_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& kernel_size,
    const c10::optional<torch::Tensor>& bias_opt = c10::nullopt);
