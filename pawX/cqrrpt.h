#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cqrrpt_core(
    const torch::Tensor& M, const torch::Tensor& S);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cqrrpt(
    const torch::Tensor& M, double gamma = 1.25, const std::string& F = "default");