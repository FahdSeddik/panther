#include "conv2d.h"

#include <torch/extension.h>

#include <vector>

// Forward function for sketched convolution
// returns output, x_windows
torch::Tensor sketched_conv2d_forward(const torch::Tensor &x,
                                      const torch::Tensor &S1s,
                                      const torch::Tensor &U1s,
                                      const std::vector<int64_t> &stride,
                                      const std::vector<int64_t> &padding,
                                      const std::vector<int64_t> &kernel_size,
                                      const c10::optional<torch::Tensor> &bias_opt)
{
    int64_t B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int64_t L = U1s.size(0), K = U1s.size(1), D1 = U1s.size(2);
    int64_t Kh = kernel_size[0], Kw = kernel_size[1];
    int64_t H_out = (H - Kh + 2 * padding[0]) / stride[0] + 1;
    int64_t W_out = (W - Kw + 2 * padding[1]) / stride[1] + 1;

    auto temp = torch::nn::functional::conv2d(x, S1s, torch::nn::functional::Conv2dFuncOptions().stride(stride).padding(padding)).view({B, L, K, H_out, W_out});

    auto temp_reshaped = temp.view({B, L * K, H_out * W_out});

    // U1s: [L, K, D1] → [LK, D1] → transpose to [D1, LK] for matmul
    auto U1s_T = U1s.view({L * K, D1}).transpose(0, 1); // [D1, LK]

    // Do batched matrix multiplication
    // We want: [B, D1, H*W] = [B, D1, LK] x [B, LK, H*W]
    auto out = torch::bmm(U1s_T.expand({B, D1, L * K}), temp_reshaped); // [B, D1, H*W]

    // Reshape to [B, D1, H, W]
    out = out.view({B, D1, H_out, W_out}) / L;

    // Add bias directly (already in target layout)
    if (bias_opt.has_value())
    {
        auto bias = bias_opt.value();
        out.add_(bias.view({1, D1, 1, 1}));
    }

    return out.contiguous();
}

// Backward function for sketched convolution
torch::autograd::tensor_list sketched_conv2d_backward(const torch::Tensor &input,
                                                      const torch::Tensor &S1s,
                                                      const torch::Tensor &S2s,
                                                      const torch::Tensor &U1s,
                                                      const torch::Tensor &U2s,
                                                      const std::vector<int64_t> &stride,
                                                      const std::vector<int64_t> &padding,
                                                      const std::vector<int64_t> &kernel_size,
                                                      const std::vector<int64_t> &in_shape,
                                                      const torch::Tensor &grad_out)
{
    int64_t num_terms = S2s.size(0);
    int64_t hout = grad_out.size(2), wout = grad_out.size(3);

    auto g_bias = grad_out.sum({0, 2, 3});
    auto g_out = grad_out.reshape({grad_out.size(0), hout * wout, grad_out.size(1)}).div(2.0 * num_terms);
    auto g_S1s = torch::einsum("nab,nbc,lcd->lad", {input.transpose(1, 2), g_out, U1s.transpose(1, 2)});
    auto g_S2s = torch::einsum("lab,nbc,ncd->lad", {U2s, input.transpose(1, 2), g_out});
    auto gout = torch::einsum("nab,lbc,lcd->nad", {g_out, U1s.transpose(1, 2), S1s.transpose(1, 2)});
    gout += torch::einsum("nab,lbc,lcd->nad", {g_out, S2s.transpose(1, 2), U2s});
    auto fold = torch::nn::Fold(torch::nn::FoldOptions(in_shape, kernel_size).stride(stride).padding(padding));
    gout = fold(gout.transpose(1, 2));
    return {gout, g_S1s, g_S2s, g_bias};
}