#include "conv2d.h"

#include <torch/extension.h>

#include <vector>

// Forward function for sketched convolution
// returns output, x_windows
std::vector<torch::Tensor> sketched_conv2d_forward(const torch::Tensor &x,
                                                   const torch::Tensor &S1s,
                                                   const torch::Tensor &S2s,
                                                   const torch::Tensor &U1s,
                                                   const torch::Tensor &U2s,
                                                   const std::vector<int64_t> &stride,
                                                   const std::vector<int64_t> &padding,
                                                   const std::vector<int64_t> &kernel_size,
                                                   const torch::Tensor &bias) {
    // Save shapes
    int64_t B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int64_t L = U1s.size(0), out_channels = U1s.size(2);

    // Optional padding
    torch::Tensor x_pad = x;
    if (padding[0] > 0 || padding[1] > 0) {
        x_pad = torch::constant_pad_nd(x, {padding[1], padding[1], padding[0], padding[0]});
    }

    // Compute output spatial dims
    int64_t H_out = (x_pad.size(2) - kernel_size[0]) / stride[0] + 1;
    int64_t W_out = (x_pad.size(3) - kernel_size[1]) / stride[1] + 1;

    // Extract sliding windows: [B, H_out*W_out, C*Kh*Kw]
    auto x_windows = x_pad.as_strided(
                              {x_pad.size(0), x_pad.size(1), H_out, W_out, kernel_size[0], kernel_size[1]},
                              {x_pad.stride(0), x_pad.stride(1), x_pad.stride(2) * stride[0], x_pad.stride(3) * stride[1], x_pad.stride(2), x_pad.stride(3)})
                         .permute({0, 2, 3, 1, 4, 5})
                         .reshape({B, -1, kernel_size[0] * kernel_size[1] * C});

    // Compute sketched conv terms
    auto out1 = torch::einsum("bnd,tdr,tro->bno", {x_windows, S1s, U1s});
    auto out2 = torch::einsum("bnd,tdr,tro->bno", {x_windows, U2s.transpose(1, 2), S2s});
    auto out = (out1 + out2).div(2.0f * L) + bias;

    // Reshape to NCHW
    return {out.view({B, H_out, W_out, out_channels}).permute({0, 3, 1, 2}), x_windows};
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