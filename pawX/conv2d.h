#pragma once

#include <torch/extension.h>

torch::Tensor sketched_conv2d_forward(const torch::Tensor &x,
                                      const torch::Tensor &S1s,
                                      const torch::Tensor &U1s,
                                      const std::vector<int64_t> &stride,
                                      const std::vector<int64_t> &padding,
                                      const std::vector<int64_t> &kernel_size,
                                      const c10::optional<torch::Tensor> &bias_opt);

torch::autograd::tensor_list sketched_conv2d_backward(const torch::Tensor &x_windows,
                                                      const torch::Tensor &S1s,
                                                      const torch::Tensor &U1s,
                                                      const std::vector<int64_t> &stride,
                                                      const std::vector<int64_t> &padding,
                                                      const std::vector<int64_t> &kernel_size,
                                                      const std::vector<int64_t> &in_shape,
                                                      const torch::Tensor &grad_out);

torch::Tensor sketched_conv_cuda(
    const torch::Tensor& x,
    const torch::Tensor& S1s,
    const std::vector<int64_t> &stride,
    const std::vector<int64_t> &padding,
    const std::vector<int64_t> &kernel_size
    );