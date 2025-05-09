#include "linear.h"

torch::Tensor sketched_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    const torch::Tensor& bias,
    const bool use_tensor_core) {
    if (input.device().is_cuda() && use_tensor_core) {
        return sketched_linear_forward_cuda(input, S1s, S2s, U1s, U2s, bias);
    } else {
        return sketched_linear_forward_cpu(input, S1s, S2s, U1s, U2s, bias);
    }
}

torch::Tensor sketched_linear_forward_cpu(
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    const torch::Tensor& bias) {
    int64_t num_terms = S2s.size(0);
    auto input_expanded = input.unsqueeze(0).expand({num_terms, input.size(0), input.size(1)});

    torch::Tensor term1 = (input_expanded.bmm(S1s)).bmm(U1s);
    torch::Tensor term2 = (input_expanded.bmm(U2s)).bmm(S2s);

    return (term1 + term2).mean(0).div(2.0f) + bias;
}

std::vector<torch::Tensor> sketched_linear_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    bool use_tensor_core) {
    if (input.device().is_cuda() && use_tensor_core) {
        return sketched_linear_backward_cuda(grad_output, input, S1s, S2s, U1s, U2s);
    } else {
        return sketched_linear_backward_cpu(grad_output, input, S1s, S2s, U1s, U2s);
    }
}

std::vector<torch::Tensor> sketched_linear_backward_cpu(
    const torch::Tensor& grad_output,  // [B, O]
    const torch::Tensor& input,        // [B, I]
    const torch::Tensor& S1s,          // [T, I, R]
    const torch::Tensor& S2s,          // [T, R, O]
    const torch::Tensor& U1s,          // [T, R, O]
    const torch::Tensor& U2s) {        // [T, I, R]
    int64_t num_terms = S2s.size(0);
    torch::Tensor g = grad_output.div(2 * num_terms).unsqueeze(0).expand({num_terms, grad_output.size(0), grad_output.size(1)});

    auto input_t = input.unsqueeze(0).expand({num_terms, input.size(0), input.size(1)}).transpose(1, 2);
    auto U1s_t = U1s.transpose(1, 2);
    auto S1s_t = S1s.transpose(1, 2);
    auto U2s_t = U2s.transpose(1, 2);
    auto S2s_t = S2s.transpose(1, 2);

    torch::Tensor t1 = g.bmm(U1s_t);
    torch::Tensor grad_input = (t1.bmm(S1s_t) + g.bmm(S2s_t).bmm(U2s_t)).sum(0);

    torch::Tensor grad_S2s = (U2s_t.bmm(input_t)).bmm(g);
    torch::Tensor grad_S1s = input_t.bmm(t1);

    return {grad_input, grad_S1s, grad_S2s, grad_output.sum(0)};
}
