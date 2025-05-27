#pragma once

#include <torch/extension.h>

torch::Tensor create_projection_matrix(
    int m, int d, int seed = 42, bool scaling = false,
    c10::optional<torch::ScalarType> dtype = c10::nullopt,
    c10::optional<torch::Device> device = c10::nullopt);

torch::Tensor causal_numerator_apply(
    const torch::Tensor& query_prime,
    const torch::Tensor& key_prime,
    const torch::Tensor& value_prime);

torch::Tensor causal_denominator_apply(
    const torch::Tensor& query_prime,
    const torch::Tensor& key_prime);

torch::Tensor rmha_forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& Wq,
    const torch::Tensor& Wk,
    const torch::Tensor& Wv,
    const torch::Tensor& W0,
    int64_t num_heads,
    int64_t embed_dim,
    const std::string& kernel_fn,
    bool causal,
    c10::optional<torch::Tensor> attention_mask = c10::nullopt,
    c10::optional<torch::Tensor> bq = c10::nullopt,
    c10::optional<torch::Tensor> bk = c10::nullopt,
    c10::optional<torch::Tensor> bv = c10::nullopt,
    c10::optional<torch::Tensor> b0 = c10::nullopt,
    c10::optional<torch::Tensor> projection_matrix = c10::nullopt);