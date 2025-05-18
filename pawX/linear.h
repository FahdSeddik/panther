#pragma once

#include <torch/extension.h>

#define AT_DISPATCH_CASE_FLOAT_AND_HALF(...)             \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)

#define AT_DISPATCH_FLOAT_AND_HALF(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(                             \
        TYPE, NAME, AT_DISPATCH_CASE_FLOAT_AND_HALF(__VA_ARGS__))

torch::Tensor sketched_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    c10::optional<torch::Tensor> bias = c10::nullopt,
    const bool use_gpu = false);

torch::Tensor sketched_linear_forward_cpu(
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    c10::optional<torch::Tensor> bias = c10::nullopt);

torch::Tensor sketched_linear_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    c10::optional<torch::Tensor> bias = c10::nullopt);

std::vector<torch::Tensor> sketched_linear_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    const bool has_bias = false,
    const bool use_gpu = false);

std::vector<torch::Tensor> sketched_linear_backward_cpu(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    const bool has_bias = false);

std::vector<torch::Tensor> sketched_linear_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    const bool has_bias = false);