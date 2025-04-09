#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cqrrpt(
    const torch::Tensor& M, double gamma = 1.25, const std::string& F = "default");