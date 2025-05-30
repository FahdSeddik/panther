#include "linear.h"

/**
 * @brief Performs a sketched linear transformation on the input tensor.
 *
 * This function applies a linear transformation to the input tensor using
 * sketching matrices (S1s, S2s) and orthogonal matrices (U1s, U2s). The computation
 * is dispatched to either a CUDA or CPU implementation based on the input tensor's
 * device and the use_gpu flag.
 *
 * @param input The input tensor to be transformed.
 * @param S1s The first sketching matrix tensor.
 * @param S2s The second sketching matrix tensor.
 * @param U1s The first orthogonal matrix tensor.
 * @param U2s The second orthogonal matrix tensor.
 * @param bias Optional bias tensor to be added after the transformation.
 * @param use_gpu Boolean flag indicating whether to use GPU computation if available.
 * @return torch::Tensor The result of the sketched linear transformation.
 */
torch::Tensor sketched_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    c10::optional<torch::Tensor> bias,
    const bool use_gpu) {
    if (input.device().is_cuda() && use_gpu) {
        return sketched_linear_forward_cuda(input, S1s, S2s, U1s, U2s, bias);
    } else {
        return sketched_linear_forward_cpu(input, S1s, S2s, U1s, U2s, bias);
    }
}

/**
 * @brief Performs a sketched linear forward operation on the CPU.
 *
 * This function computes a linear transformation of the input tensor using two sets of
 * sketching matrices (S1s, S2s) and two sets of transformation matrices (U1s, U2s).
 * The computation involves expanding the input tensor, performing batched matrix multiplications,
 * averaging the results, and optionally adding a bias term.
 *
 * @param input The input tensor of shape (batch_size, input_dim).
 * @param S1s Sketching matrices for the first term, of shape (num_terms, input_dim, sketched_dim1).
 * @param S2s Sketching matrices for the second term, of shape (num_terms, sketched_dim2, output_dim).
 * @param U1s Transformation matrices for the first term, of shape (num_terms, sketched_dim1, output_dim).
 * @param U2s Transformation matrices for the second term, of shape (num_terms, input_dim, sketched_dim2).
 * @param bias Optional bias tensor of shape (output_dim).
 * @return torch::Tensor The result tensor of shape (batch_size, output_dim).
 */
torch::Tensor sketched_linear_forward_cpu(
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    c10::optional<torch::Tensor> bias) {
    int64_t num_terms = S2s.size(0);
    auto input_expanded = input.unsqueeze(0).expand({num_terms, input.size(0), input.size(1)});

    torch::Tensor term1 = (input_expanded.bmm(S1s)).bmm(U1s);
    torch::Tensor term2 = (input_expanded.bmm(U2s)).bmm(S2s);

    auto result = (term1.mean(0) + term2.mean(0)) * 0.5;
    if (bias.has_value()) {
        result.add_(bias.value());
    }
    return result;
}

/**
 * @brief Computes the backward pass for a sketched linear layer.
 *
 * This function calculates the gradients with respect to the inputs, weights, and bias (if present)
 * for a linear layer that uses sketching matrices for dimensionality reduction or approximation.
 * It dispatches the computation to either a CUDA or CPU implementation based on the input tensor's device
 * and the use_gpu flag.
 *
 * @param grad_output The gradient of the loss with respect to the output of the linear layer.
 * @param input The input tensor to the linear layer.
 * @param S1s The first set of sketching matrices.
 * @param S2s The second set of sketching matrices.
 * @param U1s The first set of orthogonal matrices (possibly from SVD or similar decomposition).
 * @param U2s The second set of orthogonal matrices.
 * @param has_bias Boolean flag indicating whether the linear layer has a bias term.
 * @param use_gpu Boolean flag indicating whether to use the GPU implementation if available.
 * @return std::vector<torch::Tensor> A vector containing the gradients with respect to the input, weights, and bias (if present).
 */
std::vector<torch::Tensor> sketched_linear_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    const bool has_bias,
    bool use_gpu) {
    if (input.device().is_cuda() && use_gpu) {
        return sketched_linear_backward_cuda(grad_output, input, S1s, S2s, U1s, U2s, has_bias);
    } else {
        return sketched_linear_backward_cpu(grad_output, input, S1s, S2s, U1s, U2s, has_bias);
    }
}

/**
 * @brief Computes the backward pass for a sketched linear layer on CPU.
 *
 * This function calculates the gradients with respect to the input, S1s, S2s, and optionally the bias,
 * for a linear layer that uses sketching matrices. The computation is performed for each term in the
 * sketch, and the results are summed appropriately.
 *
 * @param grad_output Gradient of the loss with respect to the output of the layer. Shape: [B, O]
 * @param input Input tensor to the layer. Shape: [B, I]
 * @param S1s Sketching matrix S1s. Shape: [T, I, R]
 * @param S2s Sketching matrix S2s. Shape: [T, R, O]
 * @param U1s Sketching matrix U1s. Shape: [T, R, O]
 * @param U2s Sketching matrix U2s. Shape: [T, I, R]
 * @param has_bias Boolean flag indicating whether the layer has a bias term.
 * @return std::vector<torch::Tensor> A vector containing:
 *         - grad_input: Gradient with respect to the input. Shape: [B, I]
 *         - grad_S1s: Gradient with respect to S1s. Shape: [T, I, R]
 *         - grad_S2s: Gradient with respect to S2s. Shape: [T, R, O]
 *         - grad_bias (optional): Gradient with respect to the bias. Shape: [O]
 */
std::vector<torch::Tensor> sketched_linear_backward_cpu(
    const torch::Tensor& grad_output,  // [B, O]
    const torch::Tensor& input,        // [B, I]
    const torch::Tensor& S1s,          // [T, I, R]
    const torch::Tensor& S2s,          // [T, R, O]
    const torch::Tensor& U1s,          // [T, R, O]
    const torch::Tensor& U2s,
    const bool has_bias) {  // [T, I, R]
    int64_t num_terms = S2s.size(0);
    torch::Tensor g = grad_output / (2 * num_terms);
    g = g.unsqueeze(0).expand({num_terms, g.size(0), g.size(1)});

    auto input_t = input.unsqueeze(0).expand({num_terms, input.size(0), input.size(1)}).transpose(1, 2);
    auto U1s_t = U1s.transpose(1, 2);
    auto S1s_t = S1s.transpose(1, 2);
    auto U2s_t = U2s.transpose(1, 2);
    auto S2s_t = S2s.transpose(1, 2);

    torch::Tensor t1 = g.bmm(U1s_t);
    torch::Tensor grad_input = t1.bmm(S1s_t).sum(0) + g.bmm(S2s_t).bmm(U2s_t).sum(0);

    torch::Tensor grad_S2s = (U2s_t.bmm(input_t)).bmm(g);
    torch::Tensor grad_S1s = input_t.bmm(g.bmm(U1s_t));

    if (has_bias) {
        return {grad_input, grad_S1s, grad_S2s, grad_output.sum(0)};
    }
    return {grad_input, grad_S1s, grad_S2s};
}