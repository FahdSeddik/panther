#include "conv2d.h"

#include <torch/extension.h>

#include <vector>

// Forward function for sketched convolution
// returns output, x_windows
/**
 * @brief Performs a sketched 2D convolution operation with low-rank approximation.
 *
 * This function applies a 2D convolution to the input tensor `x` using the sketching weights `S1s`,
 * followed by a projection with the low-rank basis `U1s`. The result is optionally biased and returned.
 *
 * @param x         Input tensor of shape [B, C, H, W], where B is batch size, C is input channels, H and W are spatial dimensions.
 * @param S1s       Sketching weight tensor for the convolution, typically of shape [L, C, Kh, Kw], where L is the sketch dimension.
 * @param U1s       Low-rank basis tensor of shape [L, K, D1], where K is the number of output channels per sketch and D1 is the output dimension.
 * @param stride    Stride for the convolution, as a vector of two integers [stride_h, stride_w].
 * @param padding   Padding for the convolution, as a vector of two integers [pad_h, pad_w].
 * @param kernel_size Kernel size for the convolution, as a vector of two integers [Kh, Kw].
 * @param bias_opt  Optional bias tensor of shape [D1].
 * @return torch::Tensor Output tensor of shape [B, D1, H_out, W_out], where H_out and W_out are the output spatial dimensions.
 *
 * The function first applies a convolution with S1s, reshapes the result, and projects it using U1s.
 * The output is normalized by the sketch dimension L and optionally biased.
 */
torch::Tensor sketched_conv2d_forward(const torch::Tensor &x,
                                      const torch::Tensor &S1s,
                                      const torch::Tensor &U1s,
                                      const std::vector<int64_t> &stride,
                                      const std::vector<int64_t> &padding,
                                      const std::vector<int64_t> &kernel_size,
                                      const c10::optional<torch::Tensor> &bias_opt) {
    int64_t B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int64_t L = U1s.size(0), K = U1s.size(1), D1 = U1s.size(2);
    int64_t Kh = kernel_size[0], Kw = kernel_size[1];
    int64_t H_out = (H - Kh + 2 * padding[0]) / stride[0] + 1;
    int64_t W_out = (W - Kw + 2 * padding[1]) / stride[1] + 1;

    auto temp = torch::nn::functional::conv2d(x, S1s, torch::nn::functional::Conv2dFuncOptions().stride(stride).padding(padding)).view({B, L, K, H_out, W_out});

    auto temp_reshaped = temp.view({B, L * K, H_out * W_out});

    // U1s: [L, K, D1] → [LK, D1] → transpose to [D1, LK] for matmul
    auto U1s_T = U1s.view({L * K, D1}).transpose(0, 1);  // [D1, LK]

    // Do batched matrix multiplication
    // We want: [B, D1, H*W] = [B, D1, LK] x [B, LK, H*W]
    auto out = torch::bmm(U1s_T.expand({B, D1, L * K}), temp_reshaped);  // [B, D1, H*W]

    // Reshape to [B, D1, H, W]
    out = out.view({B, D1, H_out, W_out}) / L;

    // Add bias directly (already in target layout)
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value();
        out.add_(bias.view({1, D1, 1, 1}));
    }

    return out.contiguous();
}

// Backward function for sketched convolution
/**
 * Computes the backward pass for a sketched 2D convolution operation.
 *
 * @param input      The input tensor of shape (batch_size, in_channels, height, width).
 * @param S1s        The first set of sketch matrices (tensor).
 * @param S2s        The second set of sketch matrices (tensor).
 * @param U1s        The first set of orthogonal matrices (tensor).
 * @param U2s        The second set of orthogonal matrices (tensor).
 * @param stride     The stride of the convolution (vector of int64_t).
 * @param padding    The padding added to both sides of the input (vector of int64_t).
 * @param kernel_size The size of the convolution kernel (vector of int64_t).
 * @param in_shape   The shape of the input tensor before unfolding (vector of int64_t).
 * @param grad_out   The gradient of the loss with respect to the output of the convolution (tensor).
 * @return           A list of tensors containing gradients with respect to:
 *                   - input (gout)
 *                   - S1s (g_S1s)
 *                   - S2s (g_S2s)
 *                   - bias (g_bias)
 */
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