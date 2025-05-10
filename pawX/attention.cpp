#include "attention.h"

#include <c10/util/Optional.h>

#include <cmath>
#include <vector>

// CausalNumeratorFunction
struct CausalNumeratorFunction : public torch::autograd::Function<CausalNumeratorFunction> {
    // Forward pass: computes
    //   sums = cumsum(einsum("dijk,dijl->dijkl", ks, vs), dim=0)
    //   result = einsum("dijkl,dijk->dijl", sums, qs)
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 torch::Tensor qs,
                                 torch::Tensor ks,
                                 torch::Tensor vs) {
        // Compute einsum("dijk,dijl->dijkl", ks, vs)
        auto einsumKV = torch::einsum("dijk,dijl->dijkl", {ks, vs});
        // Compute cumulative sum over dimension 0.
        auto sums = torch::cumsum(einsumKV, 0);
        // Compute einsum("dijkl,dijk->dijl", sums, qs)
        auto result = torch::einsum("dijkl,dijk->dijl", {sums, qs});

        // Save tensors for backward.
        ctx->save_for_backward({qs, ks, vs, sums});
        return result;
    }

    // Backward pass: receives gradient with respect to the output (res_grad)
    // and returns gradients for qs, ks, and vs.
    static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext* ctx,
                                               std::vector<torch::Tensor> grad_outputs) {
        auto res_grad = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto qs = saved[0];
        auto ks = saved[1];
        auto vs = saved[2];
        auto sums = saved[3];

        // Compute:
        // grads = flip(cumsum(flip(torch.matmul(qs.unsqueeze(-1), res_grad.unsqueeze(-2)), dims={0}), 0), dims={0})
        auto mat = torch::matmul(qs.unsqueeze(-1), res_grad.unsqueeze(-2));
        auto grads = torch::flip(torch::cumsum(torch::flip(mat, {0}), /*dim=*/0), {0});

        // q_grads = einsum("dijkl,dijl->dijk", sums, res_grad)
        auto q_grads = torch::einsum("dijkl,dijl->dijk", {sums, res_grad});
        // k_grads = einsum("dijkl,dijl->dijk", grads, vs)
        auto k_grads = torch::einsum("dijkl,dijl->dijk", {grads, vs});
        // v_grads = einsum("dijkl,dijk->dijl", grads, ks)
        auto v_grads = torch::einsum("dijkl,dijk->dijl", {grads, ks});

        return {q_grads, k_grads, v_grads};
    }
};

// CausalDenominatorFunction
struct CausalDenominatorFunction : public torch::autograd::Function<CausalDenominatorFunction> {
    // Forward pass: computes
    //   sums = cumsum(ks, dim=0)
    //   result = einsum("dijk,dijk->dij", qs, sums)
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                                 torch::Tensor qs,
                                 torch::Tensor ks) {
        auto sums = torch::cumsum(ks, /*dim=*/0);
        auto result = torch::einsum("dijk,dijk->dij", {qs, sums});
        ctx->save_for_backward({qs, ks, sums});
        return result;
    }

    // Backward pass: returns gradients for qs and ks.
    static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext* ctx,
                                               std::vector<torch::Tensor> grad_outputs) {
        auto res_grad = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto qs = saved[0];
        // auto ks = saved[1];  // Not needed for computing qs grad.
        auto sums = saved[2];

        // q_grads = einsum("dijk,dij->dijk", sums, res_grad)
        auto q_grads = torch::einsum("dijk,dij->dijk", {sums, res_grad});
        // partial_k = einsum("dijk,dij->dijk", qs, res_grad)
        auto partial_k = torch::einsum("dijk,dij->dijk", {qs, res_grad});
        // k_grads = flip(cumsum(flip(partial_k, dims={0}), 0), dims={0})
        auto k_grads = torch::flip(torch::cumsum(torch::flip(partial_k, {0}), /*dim=*/0), {0});

        return {q_grads, k_grads};
    }
};

// Wrapper for causal numerator.
torch::Tensor causal_numerator_apply(
    const torch::Tensor& query_prime,
    const torch::Tensor& key_prime,
    const torch::Tensor& value_prime) {
    return CausalNumeratorFunction::apply(query_prime, key_prime, value_prime);
}

// Wrapper for causal denominator.
torch::Tensor causal_denominator_apply(
    const torch::Tensor& query_prime,
    const torch::Tensor& key_prime) {
    return CausalDenominatorFunction::apply(query_prime, key_prime);
}

// Noncausal numerator: performs two permutations, matrix multiplications, then permutes back.
inline torch::Tensor noncausal_numerator(const torch::Tensor& query_prime,
                                         const torch::Tensor& key_prime,
                                         const torch::Tensor& value) {
    // Permute key and value.
    auto kvs = torch::matmul(key_prime.permute({1, 2, 3, 0}),
                             value.permute({1, 2, 0, 3}));
    // Multiply and permute back.
    return torch::matmul(query_prime.permute({1, 2, 0, 3}), kvs).permute({2, 0, 1, 3});
}

// Noncausal denominator: sums key_prime along dim=0 and then sums the product over the last dimension.
inline torch::Tensor noncausal_denominator(const torch::Tensor& query_prime,
                                           const torch::Tensor& key_prime) {
    return (query_prime * key_prime.sum(/*dim=*/0)).sum(-1);
}

// Create a projection matrix with QR-based orthogonal blocks.
// Parameters:
//   m: total number of rows in the projection matrix.
//   d: the inner dimension (each block is d x d).
//   seed: RNG seed for reproducibility.
//   scaling: if true, rows are scaled to sqrt(d), otherwise each row is scaled by its chi-distributed norm.
//   dtype: desired data type (if not provided, defaults to torch::kFloat).
//   device: desired device (if not provided, defaults to CPU).
torch::Tensor create_projection_matrix(
    int m, int d, int seed, bool scaling,
    c10::optional<torch::ScalarType> dtype,
    c10::optional<torch::Device> device) {
    // Set up tensor options from provided dtype/device or fallback to defaults.
    auto opts = torch::TensorOptions().dtype(dtype.value_or(torch::kFloat)).device(device.value_or(torch::kCPU));
    torch::manual_seed(seed);

    // Compute the number of full dxd blocks.
    int nb_full_blocks = m / d;
    std::vector<torch::Tensor> block_list;
    block_list.reserve(nb_full_blocks + 1);

    // Create complete blocks using QR decomposition.
    for (int i = 0; i < nb_full_blocks; ++i) {
        auto qr = torch::linalg_qr(torch::randn({d, d}, opts));
        // Transpose so that rows are orthonormal.
        block_list.emplace_back(std::move(std::get<0>(qr).transpose(0, 1)));
    }

    // If m is not an exact multiple of d, form the last incomplete block.
    int remaining_rows = m - nb_full_blocks * d;
    if (remaining_rows > 0) {
        auto qr = torch::linalg_qr(torch::randn({d, d}, opts));
        // Take only the first 'remaining_rows' rows (after transposition).
        block_list.emplace_back(std::move(std::get<0>(qr).transpose(0, 1).narrow(0, 0, remaining_rows)));
    }

    // Stack blocks vertically.
    auto final_matrix = torch::vstack(block_list);

    // Compute row scaling multiplier.
    torch::Tensor multiplier;
    if (scaling) {
        multiplier = torch::full({m}, std::sqrt(static_cast<double>(d)), opts);
    } else {
        multiplier = torch::norm(torch::randn({m, d}, opts), 2, /*dim=*/1);
    }
    // Multiply by diagonal multiplier.
    return torch::matmul(torch::diag(multiplier), final_matrix);
}

// Implements the relu_kernel_transform
// data: input tensor of shape (B, L, H, D) (for example)
// is_query: a flag not used in computation here, but left for API parity
// projection_matrix: optional tensor of shape (M, D)
// numerical_stabilizer: a small constant added after activation
inline torch::Tensor relu_kernel_transform(
    const torch::Tensor& data, bool is_query = true,
    c10::optional<torch::Tensor> projection_matrix = c10::nullopt,
    double numerical_stabilizer = 1e-5,
    c10::optional<torch::Device> device = c10::nullopt,
    c10::optional<torch::ScalarType> dtype = c10::nullopt) {
    // Build options from the provided device/dtype or fall back to data's options.
    auto options = data.options();
    if (dtype.has_value()) {
        options = options.dtype(dtype.value());
    }
    if (device.has_value()) {
        options = options.device(device.value());
    }

    // Case 1: No projection matrix provided.
    if (!projection_matrix.has_value()) {
        // The result will inherit data's options automatically.
        return torch::relu(data) + numerical_stabilizer;
    } else {
        // Use the provided projection matrix. Ensure that it uses the desired options.
        auto proj = projection_matrix.value();
        // Check if the projection matrix is on the same device as data.
        if (proj.device() != data.device()) {
            throw std::runtime_error("Projection matrix must be on the same device as data.");
        }

        // Get the number of rows (M) of the projection matrix.
        auto m = proj.size(0);
        // Compute ratio = 1.0 / sqrt(m) as a double scalar.
        auto ratio = 1.0 / std::sqrt(static_cast<double>(m));

        // Return the activated and stabilized result.
        return torch::relu(ratio * torch::matmul(data, proj.transpose(0, 1))) + numerical_stabilizer;
    }
}

// Implements the softmax_kernel_transform equivalent in C++ using libtorch.
// - data: input tensor.
// - is_query: flag to determine reduction along dimension -3 when false.
// - projection_matrix: required projection matrix tensor.
// - numerical_stabilizer: a small constant for stabilization.
// - device: optional device (if not provided, taken from data).
// - dtype: optional dtype (if not provided, taken from data).
inline torch::Tensor softmax_kernel_transform(
    const torch::Tensor& data,
    bool is_query = true,
    c10::optional<torch::Tensor> projection_matrix = c10::nullopt,
    double numerical_stabilizer = 1e-5,
    c10::optional<torch::Device> device = c10::nullopt,
    c10::optional<torch::ScalarType> dtype = c10::nullopt) {
    TORCH_CHECK(projection_matrix.has_value(), "projection_matrix is required for softmax kernel");

    // Get options from data, override with provided device/dtype if available.
    auto opts = data.options();
    if (dtype.has_value()) opts = opts.dtype(dtype.value());
    if (device.has_value()) opts = opts.device(device.value());

    // Ensure projection_matrix uses the desired options.
    auto proj = projection_matrix.value().to(opts);

    // Compute data_normalizer = 1 / sqrt(sqrt(last-dim of data)) and scale data.
    double normalizer = 1.0 / std::sqrt(std::sqrt(static_cast<double>(data.size(-1))));
    // Avoid extra temporaries by directly using the scaled data.
    auto scaled_data = data * normalizer;

    // Compute ratio = 1 / sqrt(number of rows of projection_matrix)
    double ratio = 1.0 / std::sqrt(static_cast<double>(proj.size(0)));

    // Compute data_dash = scaled_data @ proj.T (equivalent to einsum "blhd,md->blhm")
    auto data_dash = torch::matmul(scaled_data, proj.transpose(0, 1));

    // Compute diag_data = (||scaled_data||^2 / 2) unsqueezed to match dimensions.
    auto diag_data = torch::unsqueeze(scaled_data.pow(2).sum(-1).div(2.0), -1);

    // Compute max_val along last dimension of data_dash.
    auto max_val = std::get<0>(data_dash.max(-1, /*keepdim=*/true));
    // For key/value data, reduce max further along dimension -3.
    if (!is_query)
        max_val = std::get<0>(max_val.max(-3, /*keepdim=*/true));

    // Final result: ratio * (exp(data_dash - diag_data - max_val) + numerical_stabilizer)
    return ratio * (torch::exp(data_dash - diag_data - max_val) + numerical_stabilizer);
}

// Favor attention wrapper function.
// Parameters:
//   query, key, value: Input tensors (assumed to have compatible shapes)
//   kernel_fn: "relu" or "softmax"
//   causal: whether to use causal attention or not.
//   projection_matrix: optional projection matrix used in kernel transforms.
inline torch::Tensor favor_attention(const torch::Tensor& query,
                                     const torch::Tensor& key,
                                     const torch::Tensor& value,
                                     const std::string& kernel_fn,
                                     c10::optional<torch::Tensor> attention_mask,  // [B, L, S]
                                     bool causal,
                                     c10::optional<torch::Tensor> projection_matrix = c10::nullopt) {
    torch::Tensor query_prime, key_prime;
    if (kernel_fn == "relu") {
        query_prime = relu_kernel_transform(query, true, projection_matrix);
        key_prime = relu_kernel_transform(key, false, projection_matrix);
    } else if (kernel_fn == "softmax") {
        query_prime = softmax_kernel_transform(query, true, projection_matrix);
        key_prime = softmax_kernel_transform(key, false, projection_matrix);
    } else {
        TORCH_CHECK(false, "Unsupported kernel_fn: ", kernel_fn);
    }
    // attention mask for L
    if (attention_mask.has_value()) {
        attention_mask.value() = ~attention_mask.value();
        auto maskL = attention_mask.value().all(-1);
        auto maskS = attention_mask.value().all(-2);
        TORCH_CHECK(!maskL.all(-1).any().item<bool>(), "Attention mask cannot be all ones for any batch in dimension L.");
        query_prime.masked_fill_(maskL.unsqueeze(-1).unsqueeze(-1), 0.0);
        key_prime.masked_fill_(maskS.unsqueeze(-1).unsqueeze(-1), 0.0);
    }
    query_prime = query_prime.permute({1, 0, 2, 3});
    key_prime = key_prime.permute({1, 0, 2, 3});

    // Also permute value.
    auto value_prime = value.permute({1, 0, 2, 3});

    torch::Tensor av_attention, attention_normalizer;
    if (causal) {
        av_attention = causal_numerator_apply(query_prime, key_prime, value_prime);
        attention_normalizer = causal_denominator_apply(query_prime, key_prime);
    } else {
        av_attention = noncausal_numerator(query_prime, key_prime, value_prime);
        attention_normalizer = noncausal_denominator(query_prime, key_prime);
    }

    // For attention_normalizer: permute to (1, 0, 2) and unsqueeze at the last dimension.
    attention_normalizer.masked_fill_(attention_normalizer == 0, 1.0);
    return av_attention.permute({1, 0, 2, 3}) / attention_normalizer.permute({1, 0, 2}).unsqueeze(-1);
}

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
    c10::optional<torch::Tensor> attention_mask,  // [B, L, S]
    c10::optional<torch::Tensor> bq,
    c10::optional<torch::Tensor> bk,
    c10::optional<torch::Tensor> bv,
    c10::optional<torch::Tensor> b0,
    c10::optional<torch::Tensor> projection_matrix) {
    if (attention_mask.has_value()) {
        TORCH_CHECK(attention_mask.value().dtype() == torch::kBool,
                    "Attention mask must be of type bool.");
    }
    using namespace torch::nn::functional;

    // Linear projections
    auto q_proj = linear(query, Wq, bq);
    auto k_proj = linear(key, Wk, bk);
    auto v_proj = linear(value, Wv, bv);

    int64_t head_dim = embed_dim / num_heads;

    // Reshape for multi-head attention: (B, L, H, D/H)
    auto q = q_proj.view({query.size(0), query.size(1), num_heads, head_dim});
    auto k = k_proj.view({key.size(0), key.size(1), num_heads, head_dim});
    auto v = v_proj.view({value.size(0), value.size(1), num_heads, head_dim});

    // Apply FAVOR attention
    auto attn_out = favor_attention(q, k, v, kernel_fn, attention_mask, causal, projection_matrix);

    // Merge heads: (B, L, D)
    attn_out = attn_out.reshape({query.size(0), query.size(1), embed_dim});

    // Final output projection
    auto out = linear(attn_out, W0, b0);
    return out;
}
