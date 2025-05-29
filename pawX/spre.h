#pragma once

#include <torch/extension.h>

#include <torch/torch.h>
#include <cmath>
#include <iostream>

class sinSRPEImpl : public torch::nn::Module{
public:
    // Parameters
    torch::Tensor freqs;
    torch::Tensor phases;
    torch::Tensor scales;
    torch::Tensor z;

    // Hyperparameters
    int64_t num_heads;
    int64_t perHead_in;
    int64_t sines;
    int64_t num_realizations;
    torch::Device device;
    torch::Dtype dtype;

    sinSRPEImpl(int64_t num_heads_,
            int64_t perHead_in_,
            int64_t sines_,
            int64_t num_realizations_ = 256,
            torch::Device device_ = torch::kCPU,
            torch::Dtype dtype_ = torch::kFloat);

    // Forward returns a pair of Tensors: qbar and kbar
    std::pair<torch::Tensor, torch::Tensor> forward(int64_t len);
};

TORCH_MODULE(sinSRPE);

// Usage example (in some function):
// auto model = std::make_shared<sinSRPE>(8, 64, 16, 256, torch::kCUDA, torch::kFloat);
// auto [qbar, kbar] = model->forward(128);
