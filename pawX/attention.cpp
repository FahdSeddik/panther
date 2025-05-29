#include "attention.h"
#include "spre.h"

#include <c10/util/Optional.h>

#include <cmath>
#include <vector>

// CausalNumeratorFunction
struct CausalNumeratorFunction : public torch::autograd::Function<CausalNumeratorFunction>
{
    // Forward pass: computes
    //   sums = cumsum(einsum("dijk,dijl->dijkl", ks, vs), dim=0)
    //   result = einsum("dijkl,dijk->dijl", sums, qs)
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor qs,
                                 torch::Tensor ks,
                                 torch::Tensor vs)
    {
        const auto &[result, sums] = causal_numerator_forward(qs, ks, vs);

        // Save tensors for backward.
        ctx->save_for_backward({qs, ks, vs, sums});
        return result;
    }

    // Backward pass: receives gradient with respect to the output (res_grad)
    // and returns gradients for qs, ks, and vs.
    static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext *ctx,
                                               std::vector<torch::Tensor> grad_outputs)
    {
        auto res_grad = grad_outputs[0];
        auto saved = ctx->get_saved_variables();
        auto qs = saved[0];
        auto ks = saved[1];
        auto vs = saved[2];
        auto sums = saved[3];

        return causal_numerator_backward(res_grad, sums, qs, ks, vs);
    }
};

// CausalDenominatorFunction
struct CausalDenominatorFunction : public torch::autograd::Function<CausalDenominatorFunction>
{
    // Forward pass: computes
    //   sums = cumsum(ks, dim=0)
    //   result = einsum("dijk,dijk->dij", qs, sums)
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                 torch::Tensor qs,
                                 torch::Tensor ks)
    {
        const auto &[result, sums] = causal_denominator_forward(qs, ks);
        ctx->save_for_backward({qs, sums});
        return result;
    }

    // Backward pass: returns gradients for qs and ks.
    static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext *ctx,
                                               std::vector<torch::Tensor> grad_outputs)
    {
        auto res_grad = grad_outputs[0].unsqueeze(-1);
        auto saved = ctx->get_saved_variables();
        auto qs = saved[0];
        // auto ks = saved[1];  // Not needed for computing qs grad.
        auto sums = saved[1];

        return causal_denominator_backward(res_grad, sums, qs);
    }
};

/**
 * @brief Computes the causal numerator for attention mechanisms using cumulative sums.
 *
 * This function performs a forward pass for a causal attention numerator, where for each step along the first dimension,
 * it accumulates the outer product of keys and values, and then contracts with the queries. The computation is performed
 * using Einstein summation notation for efficient tensor operations.
 *
 * @param qs Query tensor of shape [I, D, J, K].
 * @param ks Key tensor of shape [I, D, J, K].
 * @param vs Value tensor of shape [I, D, J, L].
 * @return A tuple containing:
 *   - torch::Tensor: The result tensor of shape [D, I, J, L], representing the causal numerator for each step.
 *   - torch::Tensor: The final accumulated sum tensor of shape [D, J, K, L].
 *
 * @note The function assumes that the input tensors are compatible in their dimensions and reside on the same device.
 */
std::tuple<torch::Tensor, torch::Tensor> causal_numerator_forward(
    const torch::Tensor &qs, // [I, D, J, K]
    const torch::Tensor &ks, // [I, D, J, K]
    const torch::Tensor &vs)
{ // [I, D, J, L]
    auto qs_p = qs.transpose(0, 1);
    auto ks_p = ks.transpose(0, 1);
    auto vs_p = vs.transpose(0, 1);
    std::vector<torch::Tensor> result;
    auto sums = torch::zeros({ks_p.size(1), ks_p.size(2), ks_p.size(3), vs_p.size(3)}, qs.options());

    int D = qs.size(1);
    result.reserve(D); // Preallocate space for result vector.
    for (int idx = 0; idx < D; ++idx)
    {
        // sums = sums + einsum("ijk,ijl->ijkl", ks[idx], vs[idx])
        sums = sums + torch::einsum("ijk,ijl->ijkl", {ks_p[idx], vs_p[idx]});
        // einsum("ijkl,ijk->ijl", sums, qs[idx])
        auto res = torch::einsum("ijkl,ijk->ijl", {sums, qs_p[idx]});
        result.push_back(res.unsqueeze(0)); // [1, D, J, L]
    }
    auto result_tensor = torch::cat(result, 0);   // [I, D, J, L]
    return {result_tensor.transpose(0, 1), sums}; // sums is not used in this loop, so return last sums
}

/**
 * @brief Computes the backward pass for the causal numerator in an attention mechanism.
 *
 * This function calculates the gradients with respect to the query (qs), key (ks), and value (vs)
 * tensors given the gradient of the result (res_grad), the running sums (sums), and the original
 * query, key, and value tensors. The computation is performed in a causal (autoregressive) manner,
 * iterating backwards over the sequence length.
 *
 * @param res_grad Gradient of the output tensor, shape [I, D, J, L].
 * @param sums Running sums tensor used in the forward pass, shape [I, J, K, E].
 * @param qs Query tensor, shape [I, D, J, K].
 * @param ks Key tensor, shape [I, D, J, K].
 * @param vs Value tensor, shape [I, D, J, L].
 * @return std::vector<torch::Tensor> A vector containing the gradients with respect to qs, ks, and vs,
 *         each with the same shape as their respective input tensors.
 */
std::vector<torch::Tensor> causal_numerator_backward(
    const torch::Tensor &res_grad, // [I, D, J, L]
    const torch::Tensor &sums,     // [I, J, K, E]
    const torch::Tensor &qs,       // [I, D, J, K]
    const torch::Tensor &ks,       // [I, D, J, K]
    const torch::Tensor &vs)
{ // [I, D, J, L]
    int B = qs.size(0);
    int L = qs.size(1);
    int H = qs.size(2);
    int M = qs.size(3);
    auto qs_p = qs.transpose(0, 1);
    auto ks_p = ks.transpose(0, 1);
    auto vs_p = vs.transpose(0, 1);
    auto res_grad_p = res_grad.transpose(0, 1);
    auto grads = torch::zeros_like(sums);
    auto q_grads = torch::zeros_like(qs_p);
    auto k_grads = torch::zeros_like(ks_p);
    auto v_grads = torch::zeros_like(vs_p);
    torch::Tensor sums_c = sums;
    for (int64_t index = L - 1; index >= 0; --index)
    {
        // einsum("ijkl,ijl->ijk") → sum over last dimension of gr_sums * res_grad
        q_grads[index] = torch::einsum("ijkl,ijl->ijk", {sums_c, res_grad_p[index]});     // (1, B, H, M)
        grads = grads + torch::einsum("ijk,ijl->ijkl", {qs_p[index], res_grad_p[index]}); // (B, H, M, M)
        k_grads[index] = torch::einsum("ijkl,ijl->ijk", {grads, vs_p[index]});            // (1, B, H, M)
        v_grads[index] = torch::einsum("ijkl,ijk->ijl", {grads, ks_p[index]});            // (1, B, H, M)
        sums_c = sums_c - torch::einsum("ijk,ijl->ijkl", {ks_p[index], vs_p[index]});     // (B, H, M, M)
    }

    return {q_grads.transpose(0, 1), k_grads.transpose(0, 1), v_grads.transpose(0, 1)};
}

/**
 * @brief Computes the causal denominator for attention mechanisms.
 *
 * Given query and key tensors, this function computes the cumulative sum of the keys
 * along dimension 1 and then performs an element-wise multiplication with the queries,
 * followed by a sum over the last dimension. This is typically used in causal attention
 * mechanisms where each position can only attend to previous positions.
 *
 * @param qs Query tensor of shape [I, D, J, K].
 * @param ks Key tensor of shape [I, D, J, K].
 * @return A tuple containing:
 *         - The result tensor of shape [I, D, J] after the einsum operation.
 *         - The cumulative sum tensor of the keys along dimension 1, shape [I, D, J, K].
 */
std::tuple<torch::Tensor, torch::Tensor> causal_denominator_forward(
    const torch::Tensor &qs, // [I, D, J, K]
    const torch::Tensor &ks)
{ // [I, D, J, K]
    // Compute cumulative sum over dimension 0.
    auto sums = torch::cumsum(ks, /*dim=*/1);
    // Compute einsum("dijk,dijk->dij", qs, sums)
    auto result = (qs * sums).sum(-1);
    return {result, sums};
}

/**
 * Computes the backward pass for the causal denominator in an attention mechanism.
 *
 * Given the gradient of the result (`res_grad`), the precomputed sums (`sums`), and the queries (`qs`),
 * this function calculates the gradients with respect to the queries and keys.
 *
 * The computation involves:
 * - Element-wise multiplication of `sums` and `res_grad` to obtain the query gradients (`q_grads`).
 * - Element-wise multiplication of `qs` and `res_grad` to obtain partial key gradients (`partial_k`).
 * - Cumulative summation of the reversed `partial_k` along dimension 1, followed by reversing the result,
 *   to obtain the key gradients (`k_grads`).
 *
 * @param res_grad The gradient of the result tensor, shape: [*, D, I, J] (batch and attention dimensions).
 * @param sums     The precomputed sums tensor, shape: [*, D, I, J].
 * @param qs       The queries tensor, shape: [*, D, I, J].
 * @return         A vector containing two tensors: {q_grads, k_grads}.
 */
std::vector<torch::Tensor> causal_denominator_backward(
    const torch::Tensor &res_grad,
    const torch::Tensor &sums,
    const torch::Tensor &qs)
{
    // q_grads = einsum("dijk,dij->dijk", sums, res_grad)
    auto q_grads = sums * res_grad;
    // partial_k = einsum("dijk,dij->dijk", qs, res_grad)
    auto partial_k = qs * res_grad;
    // k_grads = flip(cumsum(flip(partial_k, dims={0}), 0), dims={0})
    auto k_grads = torch::flip(torch::cumsum(torch::flip(partial_k, {1}), /*dim=*/1), {1});
    return {q_grads, k_grads};
}

// Noncausal numerator: performs two permutations, matrix multiplications, then permutes back.
/**
 * Computes the non-causal numerator for attention mechanisms using kernelized attention.
 *
 * Given the projected query, key, and value tensors, this function performs the following steps:
 * 1. Computes the matrix multiplication between the permuted key and value tensors to obtain an intermediate tensor.
 * 2. Multiplies the permuted query tensor with the intermediate tensor and permutes the result back to the original shape.
 *
 * @param query_prime The projected query tensor of shape [B, L, H, M], where
 *                    B = batch size,
 *                    L = sequence length,
 *                    H = number of heads,
 *                    M = projection dimension.
 * @param key_prime   The projected key tensor of shape [B, L, H, M].
 * @param value       The value tensor of shape [B, L, H, D], where
 *                    D = value dimension.
 * @return            The result tensor of shape [B, L, H, D], representing the non-causal numerator for attention.
 */
inline torch::Tensor noncausal_numerator(const torch::Tensor &query_prime, // [B, L, H, M]
                                         const torch::Tensor &key_prime,   // [B, L, H, M]
                                         const torch::Tensor &value)
{ // [B, L, H, D]
    // Permute key and value.
    auto kvs = torch::matmul(key_prime.permute({0, 2, 3, 1}), value.permute({0, 2, 1, 3})); // [B, H, M, D]
    // Multiply and permute back.// [B, H, L, M] x [B, H, M, D] -> [B, H, L, D]
    return torch::matmul(query_prime.permute({0, 2, 1, 3}), kvs).transpose(1, 2); // [B, L, H, D]
}

// Noncausal denominator: sums key_prime along dim=0 and then sums the product over the last dimension.
/**
 * Computes the denominator for non-causal attention mechanisms.
 *
 * This function takes the "prime" representations of queries and keys, and computes
 * a normalization denominator used in non-causal attention. The computation involves
 * summing the key_prime tensor along the sequence length dimension, multiplying it
 * element-wise with the transposed query_prime tensor, summing over the last dimension,
 * and then transposing the result back to the original layout.
 *
 * @param query_prime A tensor of shape [B, L, H, M] representing the transformed queries.
 * @param key_prime   A tensor of shape [B, L, H, M] representing the transformed keys.
 * @return            A tensor of shape [B, L, H] containing the computed denominators.
 */
inline torch::Tensor noncausal_denominator(const torch::Tensor &query_prime, // [B, L, H, M]
                                           const torch::Tensor &key_prime)
{ // [B, L, H, M]
    return (query_prime.transpose(0, 1) * key_prime.sum(/*dim=*/1)).sum(-1).transpose(0, 1);
}

// Create a projection matrix with QR-based orthogonal blocks.
// Parameters:
//   m: total number of rows in the projection matrix.
//   d: the inner dimension (each block is d x d).
//   seed: RNG seed for reproducibility.
//   scaling: if true, rows are scaled to sqrt(d), otherwise each row is scaled by its chi-distributed norm.
//   dtype: desired data type (if not provided, defaults to torch::kFloat).
//   device: desired device (if not provided, defaults to CPU).
/**
 * @brief Creates a random projection matrix with orthonormal rows, optionally scaled.
 *
 * This function generates an m x d projection matrix suitable for use in attention mechanisms
 * or other applications requiring random projections. The matrix is constructed by stacking
 * orthonormal blocks (using QR decomposition) and optionally scaling the rows.
 *
 * @param m Number of rows in the projection matrix.
 * @param d Number of columns in the projection matrix.
 * @param seed Random seed for reproducibility.
 * @param scaling If true, scales each row by sqrt(d); otherwise, scales by the L2 norm of a random vector.
 * @param dtype (Optional) Desired data type of the output tensor.
 * @param device (Optional) Desired device of the output tensor.
 * @return torch::Tensor The resulting m x d projection matrix.
 *
 * @note The function ensures that the rows are orthonormal within each block of size d,
 *       and handles cases where m is not a multiple of d by creating a final incomplete block.
 */
torch::Tensor create_projection_matrix(
    int m, int d, int seed, bool scaling,
    c10::optional<torch::ScalarType> dtype,
    c10::optional<torch::Device> device)
{
    // Set up tensor options from provided dtype/device or fallback to defaults.
    auto opts = torch::TensorOptions().dtype(dtype.value_or(torch::kFloat)).device(device.value_or(torch::kCPU));
    torch::manual_seed(seed);

    // Compute the number of full dxd blocks.
    int nb_full_blocks = m / d;
    std::vector<torch::Tensor> block_list;
    block_list.reserve(nb_full_blocks + 1);

    // Create complete blocks using QR decomposition.
    for (int i = 0; i < nb_full_blocks; ++i)
    {
        auto qr = torch::linalg_qr(torch::randn({d, d}, opts));
        // Transpose so that rows are orthonormal.
        block_list.emplace_back(std::move(std::get<0>(qr).transpose(0, 1)));
    }

    // If m is not an exact multiple of d, form the last incomplete block.
    int remaining_rows = m - nb_full_blocks * d;
    if (remaining_rows > 0)
    {
        auto qr = torch::linalg_qr(torch::randn({d, d}, opts));
        // Take only the first 'remaining_rows' rows (after transposition).
        block_list.emplace_back(std::move(std::get<0>(qr).transpose(0, 1).narrow(0, 0, remaining_rows)));
    }

    // Stack blocks vertically.
    auto final_matrix = torch::vstack(block_list);

    // Compute row scaling multiplier.
    torch::Tensor multiplier;
    if (scaling)
    {
        multiplier = torch::full({m}, std::sqrt(static_cast<double>(d)), opts);
    }
    else
    {
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
/**
 * @brief Applies a ReLU kernel transformation to the input tensor, optionally projecting it using a given matrix.
 *
 * This function applies a ReLU activation to the input tensor `data`, with optional projection using
 * a provided `projection_matrix`. If a projection matrix is given, the function computes the matrix
 * product of `data` and the transpose of `projection_matrix`, scales the result by 1/sqrt(M) where M
 * is the number of rows in the projection matrix, and then applies the ReLU activation. A small
 * `numerical_stabilizer` is added to the result to improve numerical stability.
 *
 * @param data Input tensor to be transformed.
 * @param is_query Indicates if the input is a query tensor (default: true). (Currently unused.)
 * @param projection_matrix Optional projection matrix. If not provided, no projection is applied.
 * @param numerical_stabilizer Small value added to the output for numerical stability (default: 1e-5).
 * @param device Optional device to perform computation on. If not provided, uses the device of `data`.
 * @param dtype Optional data type for computation. If not provided, uses the dtype of `data`.
 * @return torch::Tensor The transformed tensor after applying the ReLU kernel (and optional projection).
 * @throws std::runtime_error If the projection matrix is provided but is not on the same device as `data`.
 */
inline torch::Tensor relu_kernel_transform(
    const torch::Tensor &data, bool is_query = true,
    c10::optional<torch::Tensor> projection_matrix = c10::nullopt,
    double numerical_stabilizer = 1e-5,
    c10::optional<torch::Device> device = c10::nullopt,
    c10::optional<torch::ScalarType> dtype = c10::nullopt)
{
    // Build options from the provided device/dtype or fall back to data's options.
    auto options = data.options();
    if (dtype.has_value())
    {
        options = options.dtype(dtype.value());
    }
    if (device.has_value())
    {
        options = options.device(device.value());
    }

    // Case 1: No projection matrix provided.
    if (!projection_matrix.has_value())
    {
        // The result will inherit data's options automatically.
        return torch::relu(data) + numerical_stabilizer;
    }
    else
    {
        // Use the provided projection matrix. Ensure that it uses the desired options.
        auto proj = projection_matrix.value();
        // Check if the projection matrix is on the same device as data.
        if (proj.device() != data.device())
        {
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
/**
 * @brief Applies a softmax kernel transformation to the input tensor for efficient attention mechanisms.
 *
 * This function projects the input data using a provided projection matrix, applies normalization and
 * numerical stabilization, and computes a softmax-like transformation suitable for kernelized attention.
 *
 * @param data Input tensor of shape (..., D), where D is the feature dimension.
 * @param is_query If true, treats the input as a query; if false, as a key/value (affects max reduction).
 * @param projection_matrix Optional projection matrix tensor of shape (M, D), required for the transformation.
 * @param numerical_stabilizer Small constant added for numerical stability (default: 1e-5).
 * @param device Optional device to perform computation on (overrides data's device if provided).
 * @param dtype Optional data type for computation (overrides data's dtype if provided).
 * @return torch::Tensor Transformed tensor after applying the softmax kernel.
 *
 * @throws c10::Error if projection_matrix is not provided.
 *
 * @note The transformation is designed for use in kernelized attention mechanisms (e.g., Performer).
 */
inline torch::Tensor softmax_kernel_transform(
    const torch::Tensor &data,
    bool is_query = true,
    c10::optional<torch::Tensor> projection_matrix = c10::nullopt,
    double numerical_stabilizer = 1e-5,
    c10::optional<torch::Device> device = c10::nullopt,
    c10::optional<torch::ScalarType> dtype = c10::nullopt)
{
    TORCH_CHECK(projection_matrix.has_value(), "projection_matrix is required for softmax kernel");

    // Get options from data, override with provided device/dtype if available.
    auto opts = data.options();
    if (dtype.has_value())
        opts = opts.dtype(dtype.value());
    if (device.has_value())
        opts = opts.device(device.value());

    // Ensure projection_matrix uses the desired options.
    auto proj = projection_matrix.value().to(opts);

    // Compute data_normalizer = 1 / sqrt(d) where d is the embedding perHead dimension.
    double normalizer = 1.0 / std::sqrt(std::sqrt(static_cast<double>(data.size(-1))));
    auto scaled_data = data * normalizer;

    // Compute ratio = 1 / sqrt(m) m is number of positive random features (needed for kernel approx)
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

    // Final result: ratio * (exp(data_dash - diag_data - max_val) + numerical_stabilizer) h
    return ratio * (torch::exp(data_dash - diag_data - max_val) + numerical_stabilizer);
}

// Favor attention wrapper function.
// Parameters:
//   query, key, value: Input tensors (assumed to have compatible shapes)
//   kernel_fn: "relu" or "softmax"
//   causal: whether to use causal attention or not.
//   projection_matrix: optional projection matrix used in kernel transforms.
/**
 * @brief Computes FAVOR (Fast Attention Via positive Orthogonal Random features) attention mechanism.
 *
 * This function applies a kernel transformation (either "relu" or "softmax") to the query and key tensors,
 * then computes the attention output using either causal or non-causal attention, optionally applying an attention mask.
 *
 * @param query Input query tensor of shape [B, L, H, D], where B is batch size, L is sequence length, H is number of heads, D is head dimension.
 * @param key Input key tensor of shape [B, S, H, D], where S is source sequence length.
 * @param value Input value tensor of shape [B, S, H, D].
 * @param kernel_fn The kernel function to use for feature transformation ("relu" or "softmax").
 * @param attention_mask Optional attention mask tensor of shape [B, L, S]. True values indicate masked positions.
 * @param causal If true, applies causal attention (prevents attending to future positions).
 * @param projection_matrix Optional projection matrix for kernel transformation.
 * @return torch::Tensor The attention output tensor of shape [B, L, H, D].
 *
 * @throws c10::Error if an unsupported kernel function is provided or if the attention mask is invalid.
 */
inline torch::Tensor favor_attention(const torch::Tensor &query,
                                     const torch::Tensor &key,
                                     const torch::Tensor &value,
                                     const std::string &kernel_fn,
                                     c10::optional<torch::Tensor> attention_mask, // [B, L, S]
                                     bool causal,
                                     c10::optional<torch::Tensor> projection_matrix = c10::nullopt)
{
    torch::Tensor query_prime, key_prime;
    if (kernel_fn == "relu")
    {
        query_prime = relu_kernel_transform(query, true, projection_matrix);
        key_prime = relu_kernel_transform(key, false, projection_matrix);
    }
    else if (kernel_fn == "softmax")
    {
        query_prime = softmax_kernel_transform(query, true, projection_matrix);
        key_prime = softmax_kernel_transform(key, false, projection_matrix);
    }
    else
    {
        TORCH_CHECK(false, "Unsupported kernel_fn: ", kernel_fn);
    }
    // attention mask for L
    if (attention_mask.has_value())
    {
        attention_mask.value() = ~attention_mask.value();
        auto maskL = attention_mask.value().all(-1);
        auto maskS = attention_mask.value().all(-2);
        TORCH_CHECK(!maskL.all(-1).any().item<bool>(), "Attention mask cannot be all ones for any batch in dimension L.");
        query_prime.masked_fill_(maskL.unsqueeze(-1).unsqueeze(-1), 0.0);
        key_prime.masked_fill_(maskS.unsqueeze(-1).unsqueeze(-1), 0.0);
    }

    torch::Tensor av_attention, attention_normalizer;
    if (causal)
    {
        av_attention = CausalNumeratorFunction::apply(query_prime, key_prime, value);
        attention_normalizer = CausalDenominatorFunction::apply(query_prime, key_prime);
    }
    else
    {
        av_attention = noncausal_numerator(query_prime, key_prime, value);    // [B, L, H, D]
        attention_normalizer = noncausal_denominator(query_prime, key_prime); // [B, L, H]
    }

    // For attention_normalizer: permute to (1, 0, 2) and unsqueeze at the last dimension.
    if (attention_mask.has_value())
        attention_normalizer.masked_fill_(attention_normalizer == 0, 1.0);
    return av_attention / attention_normalizer.unsqueeze(-1);
}

/**
 * @brief Computes the forward pass of a Reparameterized Multi-Head Attention (RMHA) layer with optional FAVOR attention.
 *
 * This function performs the following steps:
 * 1. Applies linear projections to the input query, key, and value tensors.
 * 2. Reshapes the projections for multi-head attention.
 * 3. Computes attention using the specified kernel function (e.g., FAVOR).
 * 4. Merges the attention heads.
 * 5. Applies a final linear output projection.
 *
 * @param query Input query tensor of shape [B, L, D], where B is batch size, L is sequence length, D is embedding dimension.
 * @param key Input key tensor of shape [B, S, D], where S is source sequence length.
 * @param value Input value tensor of shape [B, S, D].
 * @param Wq Weight tensor for query projection of shape [D, D].
 * @param Wk Weight tensor for key projection of shape [D, D].
 * @param Wv Weight tensor for value projection of shape [D, D].
 * @param W0 Weight tensor for output projection of shape [D, D].
 * @param num_heads Number of attention heads.
 * @param embed_dim Embedding dimension (must be divisible by num_heads).
 * @param kernel_fn Name of the kernel function to use for attention (e.g., "relu", "softmax", etc.).
 * @param causal If true, applies causal masking for autoregressive attention.
 * @param attention_mask Optional boolean mask tensor of shape [B, L, S] indicating positions to mask.
 * @param bq Optional bias tensor for query projection.
 * @param bk Optional bias tensor for key projection.
 * @param bv Optional bias tensor for value projection.
 * @param b0 Optional bias tensor for output projection.
 * @param projection_matrix Optional projection matrix for kernel-based attention.
 * @return torch::Tensor Output tensor of shape [B, L, D] after attention and output projection.
 *
 * @throws c10::Error if attention_mask is provided and is not of type bool.
 */
torch::Tensor rmha_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &value,
    const torch::Tensor &Wq,
    const torch::Tensor &Wk,
    const torch::Tensor &Wv,
    const torch::Tensor &W0,
    int64_t num_heads,
    int64_t embed_dim,
    const std::string &kernel_fn,
    bool causal,
    c10::optional<torch::Tensor> attention_mask, // [B, L, S]
    c10::optional<torch::Tensor> bq,
    c10::optional<torch::Tensor> bk,
    c10::optional<torch::Tensor> bv,
    c10::optional<torch::Tensor> b0,
    c10::optional<torch::Tensor> projection_matrix,
    std::shared_ptr<sinSRPEImpl> spre_model)
{
    if (attention_mask.has_value())
    {
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

    if (spre_model)
    {
        TORCH_CHECK(q.size(1) == k.size(1),
                    "Query and Key must be same length (i.e. self-attention) to use SPRE");
        auto [qbar, kbar] = spre_model->forward(q.size(1));
        q = (qbar * q.unsqueeze(-1)).sum(-2);
        k = (kbar * k.unsqueeze(-1)).sum(-2);
    }

    // Apply FAVOR attention
    auto attn_out = favor_attention(q, k, v, kernel_fn, attention_mask, causal, projection_matrix);

    // Merge heads: (B, L, D)
    attn_out = attn_out.reshape({query.size(0), query.size(1), embed_dim});

    // Final output projection
    auto out = linear(attn_out, W0, b0);
    return out;
}
