#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> randomized_svd(const torch::Tensor& A, int64_t k, double tol);