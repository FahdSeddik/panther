#pragma once

#include <torch/extension.h>

#include "skops.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cqrrpt(
    const torch::Tensor& M, double gamma = 1.25, const DistributionFamily& F = DistributionFamily::Gaussian);