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
                                      const torch::Tensor &bias) {
    int64_t B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int64_t L = U1s.size(0), K = U1s.size(1), D1 = U1s.size(2);
    int64_t Kh = kernel_size[0], Kw = kernel_size[1];
    int64_t H_out = (H - Kh + 2 * padding[0]) / stride[0] + 1;
    int64_t W_out = (W - Kw + 2 * padding[1]) / stride[1] + 1;

    auto temp = torch::nn::functional::conv2d(x, S1s, torch::nn::functional::Conv2dFuncOptions().stride(stride).padding(padding)).view({B, L, K, H_out, W_out});
    auto out = torch::einsum("blkhw,lko->blhwo", {temp, U1s}).mean(1) + bias;
    return out.view({B, H_out, W_out, D1}).permute({0, 3, 1, 2});
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
                                                      const torch::Tensor &grad_out) {
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