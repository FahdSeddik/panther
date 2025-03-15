#pragma once

#include <torch/extension.h>

torch::Tensor scaled_sign_sketch(int64_t m, int64_t n,
                                 c10::optional<torch::Device> device = c10::nullopt,
                                 c10::optional<torch::Dtype> dtype = c10::nullopt);
