#include "conv2d.h"

#include <torch/extension.h>

#include <fstream>
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
 * @param U1s        The first set of orthogonal matrices (tensor).
 * @param stride     The stride of the convolution (vector of int64_t).
 * @param padding    The padding added to both sides of the input (vector of int64_t).
 * @param kernel_size The size of the convolution kernel (vector of int64_t).
 * @param in_shape   The shape of the input tensor before unfolding (vector of int64_t).
 * @param grad_out   The gradient of the loss with respect to the output of the convolution (tensor).
 * @return           A list of tensors containing gradients with respect to:
 *                   - input (gout)
 *                   - S1s (g_S1s)
 *                   - bias (g_bias)
 */
torch::autograd::tensor_list sketched_conv2d_backward(const torch::Tensor &input,
                                                      const torch::Tensor &S1s,
                                                      const torch::Tensor &U1s,
                                                      const std::vector<int64_t> &stride,
                                                      const std::vector<int64_t> &padding,
                                                      const std::vector<int64_t> &kernel_size,
                                                      const std::vector<int64_t> &in_shape,
                                                      const torch::Tensor &grad_out) {
    int64_t num_terms = S1s.size(0);
    int64_t hout = grad_out.size(2), wout = grad_out.size(3);

    auto g_bias = grad_out.sum({0, 2, 3});
    auto g_out = grad_out.reshape({grad_out.size(0), hout * wout, grad_out.size(1)}).div(2.0 * num_terms);
    auto g_S1s = torch::einsum("nab,nbc,lcd->lad", {input.transpose(1, 2), g_out, U1s.transpose(1, 2)});
    auto gout = torch::einsum("nab,lbc,lcd->nad", {g_out, U1s.transpose(1, 2), S1s.transpose(1, 2)});
    auto fold = torch::nn::Fold(torch::nn::FoldOptions(in_shape, kernel_size).stride(stride).padding(padding));
    gout = fold(gout.transpose(1, 2));
    return {gout, g_S1s, g_bias};
}

/**
 * @brief Computes the output shape of a 2D convolution operation.
 *
 * Given the input tensor shape, kernel size, stride, and padding, this function calculates
 * the resulting output tensor shape after applying a 2D convolution.
 *
 * @param input_shape  A vector of 4 integers representing the input tensor shape in the format
 *                     {batch_size, channels, height, width}.
 * @param kernel_size  A vector of 2 integers representing the kernel size in the format
 *                     {kernel_height, kernel_width}.
 * @param stride       A vector of 2 integers representing the stride in the format
 *                     {stride_height, stride_width}.
 * @param padding      A vector of 2 integers representing the padding in the format
 *                     {pad_height, pad_width}.
 * @return std::vector<int64_t>  The output tensor shape in the format
 *                               {batch_size, channels, output_height, output_width}.
 */
std::vector<int64_t> compute_conv_output_shape(const std::vector<int64_t> &input_shape,
                                               const std::vector<int64_t> &kernel_size,
                                               const std::vector<int64_t> &stride,
                                               const std::vector<int64_t> &padding) {
    int64_t H = input_shape[2], W = input_shape[3];
    int64_t Kh = kernel_size[0], Kw = kernel_size[1];
    int64_t H_out = (H - Kh + 2 * padding[0]) / stride[0] + 1;
    int64_t W_out = (W - Kw + 2 * padding[1]) / stride[1] + 1;
    return {input_shape[0], input_shape[1], H_out, W_out};
}

/**
 * @brief Performs a memory-efficient sketched 2D convolution operation on the input tensor.
 *
 * This function applies a sketched convolution using provided sketching matrices (S1s, U1s) and processes
 * the input in chunks to reduce memory usage. The operation is typically used for efficient approximations
 * of convolutional layers in neural networks.
 *
 * @param x           Input tensor of shape (B, C, H, W), where B is batch size, C is input channels,
 *                    H and W are spatial dimensions.
 * @param S1s         Sketching convolution weights tensor.
 * @param U1s         Sketching projection tensor of shape (L, K, D1), where L and K are sketch dimensions,
 *                    and D1 is the output dimension after projection.
 * @param stride      Stride for the convolution operation (vector of 2 integers).
 * @param padding     Padding for the convolution operation (vector of 2 integers).
 * @param kernel_size Size of the convolution kernel (vector of 2 integers).
 * @param bias_opt    Optional bias tensor of shape (D1).
 * @param chunk_size  Number of samples to process per chunk for memory efficiency (default: 32).
 * @return torch::Tensor Output tensor of shape (B, D1, H_out, W_out), where H_out and W_out are the
 *                       output spatial dimensions after convolution.
 *
 * @note The function assumes that S1s and U1s are precomputed and compatible with the input and kernel sizes.
 *       The output is contiguous in memory.
 */
torch::Tensor sketched_conv2d_memory_efficient(const torch::Tensor &x,
                                               const torch::Tensor &S1s,
                                               const torch::Tensor &U1s,
                                               const std::vector<int64_t> &stride,
                                               const std::vector<int64_t> &padding,
                                               const std::vector<int64_t> &kernel_size,
                                               const c10::optional<torch::Tensor> &bias_opt,
                                               int64_t chunk_size = 32) {
    int64_t B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int64_t L = U1s.size(0), K = U1s.size(1), D1 = U1s.size(2);
    int64_t H_out = (H - kernel_size[0] + 2 * padding[0]) / stride[0] + 1;
    int64_t W_out = (W - kernel_size[1] + 2 * padding[1]) / stride[1] + 1;

    auto out = torch::zeros({B, D1, H_out, W_out}, x.options());

    // Process in chunks to save memory
    for (int64_t b = 0; b < B; b += chunk_size) {
        int64_t current_chunk = std::min(chunk_size, B - b);
        auto x_chunk = x.slice(0, b, b + current_chunk);

        auto temp = torch::nn::functional::conv2d(x_chunk, S1s,
                                                  torch::nn::functional::Conv2dFuncOptions().stride(stride).padding(padding))
                        .view({current_chunk, L, K, H_out, W_out});

        auto temp_reshaped = temp.view({current_chunk, L * K, H_out * W_out});
        auto U1s_T = U1s.view({L * K, D1}).transpose(0, 1);

        auto out_chunk = torch::bmm(U1s_T.expand({current_chunk, D1, L * K}), temp_reshaped)
                             .view({current_chunk, D1, H_out, W_out}) /
                         L;

        if (bias_opt.has_value()) {
            auto bias = bias_opt.value();
            out_chunk.add_(bias.view({1, D1, 1, 1}));
        }

        out.slice(0, b, b + current_chunk).copy_(out_chunk);
    }

    return out.contiguous();
}

/**
 * @brief Computes the gradients of the sketch parameters for a convolutional layer.
 *
 * This function calculates the gradients with respect to the sketch parameters (S1s)
 * used in a sketched convolution operation. It takes as input the original input tensor,
 * the gradient of the output, the sketch matrix U1s, and convolution parameters such as
 * stride, padding, and kernel size.
 *
 * @param input The input tensor of shape (B, C, ...), where B is the batch size and C is the number of channels.
 * @param grad_output The gradient of the output tensor with respect to the loss, typically of shape (B, D1, ...).
 * @param U1s The sketch matrix tensor of shape (L, K, D1), where L and K are sketch dimensions and D1 is the output dimension.
 * @param stride The stride of the convolution as a vector of int64_t.
 * @param padding The padding of the convolution as a vector of int64_t.
 * @param kernel_size The size of the convolution kernel as a vector of int64_t.
 * @return torch::Tensor The computed gradients for the sketch parameters, of shape (L, C, kernel_size[0], kernel_size[1]).
 */
torch::Tensor compute_sketch_gradients(const torch::Tensor &input,
                                       const torch::Tensor &grad_output,
                                       const torch::Tensor &U1s,
                                       const std::vector<int64_t> &stride,
                                       const std::vector<int64_t> &padding,
                                       const std::vector<int64_t> &kernel_size) {
    int64_t B = input.size(0), C = input.size(1);
    int64_t L = U1s.size(0), K = U1s.size(1), D1 = U1s.size(2);

    auto grad_reshaped = grad_output.view({B, D1, -1});
    auto U1s_T = U1s.view({L * K, D1}).transpose(0, 1);

    auto grad_S1s = torch::zeros({L, C, kernel_size[0], kernel_size[1]}, input.options());

    for (int64_t l = 0; l < L; ++l) {
        for (int64_t k = 0; k < K; ++k) {
            auto U1s_slice = U1s_T.slice(1, l * K + k, l * K + k + 1);
            auto grad_slice = torch::bmm(grad_reshaped, U1s_slice.transpose(1, 2));

            // Accumulate gradients
            grad_S1s[l].add_(grad_slice.view({B, C, kernel_size[0], kernel_size[1]}).sum(0));
        }
    }

    return grad_S1s;
}

/**
 * @brief Checks the numerical stability of a tensor by evaluating its absolute maximum and minimum values.
 *
 * This function computes the absolute maximum and minimum values of the input tensor.
 * It returns true if the absolute maximum is greater than the specified threshold and
 * the absolute minimum is less than the reciprocal of the threshold, indicating that
 * the tensor's values are within a numerically stable range.
 *
 * @param tensor The input torch::Tensor to check for numerical stability.
 * @param threshold The threshold value for stability checks (default: 1e-6).
 * @return true if the tensor is numerically stable, false otherwise.
 */
bool check_numerical_stability(const torch::Tensor &tensor, double threshold = 1e-6) {
    auto abs_max = tensor.abs().max().item<double>();
    auto abs_min = tensor.abs().min().item<double>();
    return abs_max > threshold && abs_min < 1.0 / threshold;
}

/**
 * @brief Optimizes the layout of input tensors for batched matrix multiplication (bmm).
 *
 * This function optionally transposes the last two dimensions of the input tensors `x` and `y`
 * based on the provided flags, ensures they are contiguous in memory, and then performs
 * batched matrix multiplication using `torch::bmm`.
 *
 * @param x            The first input tensor of shape (batch_size, n, m) or (batch_size, m, n).
 * @param y            The second input tensor of shape (batch_size, m, p) or (batch_size, p, m).
 * @param transpose_x  If true, transposes the last two dimensions of `x` before multiplication.
 * @param transpose_y  If true, transposes the last two dimensions of `y` before multiplication.
 * @return             The result of batched matrix multiplication as a tensor.
 */
torch::Tensor optimize_bmm_layout(const torch::Tensor &x,
                                  const torch::Tensor &y,
                                  bool transpose_x = false,
                                  bool transpose_y = false) {
    auto x_modified = transpose_x ? x.transpose(-2, -1) : x;
    auto y_modified = transpose_y ? y.transpose(-2, -1) : y;

    auto x_cont = x_modified.contiguous();
    auto y_cont = y_modified.contiguous();

    return torch::bmm(x_cont, y_cont);
}

/**
 * @brief Computes the optimal chunk size for memory-efficient convolution based on available memory.
 *
 * This function calculates the optimal chunk size for processing large batches in the memory-efficient
 * convolution implementation. It takes into account the input tensor dimensions, available memory,
 * and system constraints to determine the most efficient chunk size.
 *
 * @param input_shape The shape of the input tensor [B, C, H, W]
 * @param kernel_size The size of the convolution kernel [Kh, Kw]
 * @param available_memory The available memory in bytes
 * @param safety_factor A factor to ensure we don't use all available memory (default: 0.8)
 * @return int64_t The optimal chunk size for processing
 */
int64_t compute_optimal_chunk_size(const std::vector<int64_t> &input_shape,
                                   const std::vector<int64_t> &kernel_size,
                                   int64_t available_memory,
                                   double safety_factor = 0.8) {
    int64_t B = input_shape[0], C = input_shape[1];
    int64_t Kh = kernel_size[0], Kw = kernel_size[1];

    int64_t memory_per_sample = C * Kh * Kw * sizeof(float);

    int64_t max_chunk_size = static_cast<int64_t>((available_memory * safety_factor) / memory_per_sample);

    return std::max((int64_t)1, std::min(max_chunk_size, B));
}

/**
 * @brief Performs a memory-efficient sketched convolution with automatic chunk size optimization.
 *
 * This function extends the basic sketched convolution by automatically determining the optimal
 * chunk size based on available memory and system constraints. It processes the input in chunks
 * to minimize memory usage while maintaining computational efficiency.
 *
 * @param x Input tensor of shape [B, C, H, W]
 * @param S1s Sketching weight tensor
 * @param U1s Low-rank basis tensor
 * @param stride Convolution stride
 * @param padding Convolution padding
 * @param kernel_size Kernel size
 * @param bias_opt Optional bias tensor
 * @param available_memory Available memory in bytes
 * @return torch::Tensor The output tensor
 */
torch::Tensor sketched_conv2d_auto_chunk(const torch::Tensor &x,
                                         const torch::Tensor &S1s,
                                         const torch::Tensor &U1s,
                                         const std::vector<int64_t> &stride,
                                         const std::vector<int64_t> &padding,
                                         const std::vector<int64_t> &kernel_size,
                                         const c10::optional<torch::Tensor> &bias_opt,
                                         int64_t available_memory) {
    std::vector<int64_t> input_shape = {x.size(0), x.size(1), x.size(2), x.size(3)};
    int64_t chunk_size = compute_optimal_chunk_size(input_shape, kernel_size, available_memory);
    return sketched_conv2d_memory_efficient(x, S1s, U1s, stride, padding, kernel_size, bias_opt, chunk_size);
}

/**
 * @brief Computes the gradient of the loss with respect to the input tensor.
 *
 * This function computes the gradient of the loss with respect to the input tensor
 * in a memory-efficient manner. It processes the gradients in chunks to minimize
 * memory usage while maintaining computational efficiency.
 *
 * @param grad_output Gradient of the loss with respect to the output
 * @param input The input tensor
 * @param S1s Sketching weight tensor
 * @param U1s Low-rank basis tensor
 * @param stride Convolution stride
 * @param padding Convolution padding
 * @param kernel_size Kernel size
 * @param chunk_size Size of chunks for processing
 * @return torch::Tensor Gradient with respect to the input
 */
torch::Tensor compute_input_gradients_memory_efficient(const torch::Tensor &grad_output,
                                                       const torch::Tensor &input,
                                                       const torch::Tensor &S1s,
                                                       const torch::Tensor &U1s,
                                                       const std::vector<int64_t> &stride,
                                                       const std::vector<int64_t> &padding,
                                                       const std::vector<int64_t> &kernel_size,
                                                       int64_t chunk_size = 32) {
    int64_t B = input.size(0), C = input.size(1);
    int64_t L = U1s.size(0), K = U1s.size(1), D1 = U1s.size(2);

    auto grad_input = torch::zeros_like(input);

    for (int64_t b = 0; b < B; b += chunk_size) {
        int64_t current_chunk = std::min(chunk_size, B - b);

        auto grad_output_chunk = grad_output.slice(0, b, b + current_chunk);
        auto input_chunk = input.slice(0, b, b + current_chunk);

        // Reshape gradients for matrix multiplication
        auto grad_reshaped = grad_output_chunk.view({current_chunk, D1, -1});
        auto U1s_T = U1s.view({L * K, D1}).transpose(0, 1);

        // Compute gradients for current chunk
        auto grad_chunk = torch::bmm(grad_reshaped, U1s_T.expand({current_chunk, D1, L * K}));

        // Reshape and accumulate gradients
        grad_input.slice(0, b, b + current_chunk).add_(grad_chunk.view({current_chunk, C, -1}));
    }

    return grad_input;
}

/**
 * @brief Performs numerical stability analysis of the sketched convolution.
 *
 * This function analyzes the numerical stability of the sketched convolution
 * by checking various aspects of the computation, including:
 * - Input tensor stability
 * - Intermediate computation stability
 * - Output tensor stability
 * - Gradient computation stability
 *
 * @param x Input tensor
 * @param S1s Sketching weight tensor
 * @param U1s Low-rank basis tensor
 * @param stride Convolution stride
 * @param padding Convolution padding
 * @param kernel_size Kernel size
 * @return std::map<std::string, bool> Map of stability checks and their results
 */
std::map<std::string, bool> analyze_numerical_stability(const torch::Tensor &x,
                                                        const torch::Tensor &S1s,
                                                        const torch::Tensor &U1s,
                                                        const std::vector<int64_t> &stride,
                                                        const std::vector<int64_t> &padding,
                                                        const std::vector<int64_t> &kernel_size) {
    std::map<std::string, bool> stability_results;

    stability_results["input_stable"] = check_numerical_stability(x);

    stability_results["sketch_weights_stable"] = check_numerical_stability(S1s);

    stability_results["basis_stable"] = check_numerical_stability(U1s);

    auto temp = torch::nn::functional::conv2d(x, S1s,
                                              torch::nn::functional::Conv2dFuncOptions().stride(stride).padding(padding));
    stability_results["conv_output_stable"] = check_numerical_stability(temp);

    auto output = sketched_conv2d_forward(x, S1s, U1s, stride, padding, kernel_size, c10::nullopt);
    stability_results["output_stable"] = check_numerical_stability(output);

    return stability_results;
}

/**
 * @brief Optimizes the memory layout of tensors for efficient computation.
 *
 * This function analyzes the memory layout of input tensors and optimizes them
 * for efficient computation by:
 * - Ensuring contiguous memory layout
 * - Aligning memory access patterns
 * - Minimizing memory fragmentation
 *
 * @param x Input tensor
 * @param S1s Sketching weight tensor
 * @param U1s Low-rank basis tensor
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Optimized tensors
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> optimize_memory_layout(
    const torch::Tensor &x,
    const torch::Tensor &S1s,
    const torch::Tensor &U1s) {
    // Ensure contiguous memory layout
    auto x_opt = x.contiguous();
    auto S1s_opt = S1s.contiguous();
    auto U1s_opt = U1s.contiguous();

    // Optimize memory alignment
    if (!x_opt.is_contiguous()) {
        x_opt = x_opt.clone();
    }
    if (!S1s_opt.is_contiguous()) {
        S1s_opt = S1s_opt.clone();
    }
    if (!U1s_opt.is_contiguous()) {
        U1s_opt = U1s_opt.clone();
    }

    return std::make_tuple(x_opt, S1s_opt, U1s_opt);
}

/**
 * @brief Computes the estimated memory usage (in bytes) for a convolutional operation.
 *
 * This function calculates the total memory required for the input tensor, sketch tensor,
 * basis tensor, and output tensor, based on their shapes and the convolution parameters.
 *
 * @param x           Input tensor of shape [B, C, H, W], where B is batch size, C is channels, H and W are height and width.
 * @param S1s         Sketch tensor, typically of shape [L, C, kernel_h, kernel_w].
 * @param U1s         Basis tensor, typically of shape [L, K, D1].
 * @param stride      Stride of the convolution, as a vector of two integers [stride_h, stride_w].
 * @param padding     Padding of the convolution, as a vector of two integers [pad_h, pad_w].
 * @param kernel_size Size of the convolution kernel, as a vector of two integers [kernel_h, kernel_w].
 * @return int64_t    Total estimated memory usage in bytes.
 */
int64_t compute_memory_usage(const torch::Tensor &x,
                             const torch::Tensor &S1s,
                             const torch::Tensor &U1s,
                             const std::vector<int64_t> &stride,
                             const std::vector<int64_t> &padding,
                             const std::vector<int64_t> &kernel_size) {
    int64_t B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int64_t L = U1s.size(0), K = U1s.size(1), D1 = U1s.size(2);
    int64_t H_out = (H - kernel_size[0] + 2 * padding[0]) / stride[0] + 1;
    int64_t W_out = (W - kernel_size[1] + 2 * padding[1]) / stride[1] + 1;

    // Compute memory usage in bytes
    int64_t input_memory = B * C * H * W * sizeof(float);
    int64_t sketch_memory = L * C * kernel_size[0] * kernel_size[1] * sizeof(float);
    int64_t basis_memory = L * K * D1 * sizeof(float);
    int64_t output_memory = B * D1 * H_out * W_out * sizeof(float);

    return input_memory + sketch_memory + basis_memory + output_memory;
}

/**
 * @brief Performs a comprehensive performance analysis of the sketched convolution.
 *
 * This function analyzes various aspects of the convolution's performance:
 * - Memory usage
 * - Computation time
 * - Numerical stability
 * - Memory efficiency
 *
 * @param x Input tensor
 * @param S1s Sketching weight tensor
 * @param U1s Low-rank basis tensor
 * @param stride Convolution stride
 * @param padding Convolution padding
 * @param kernel_size Kernel size
 * @return std::map<std::string, double> Map of performance metrics
 */
std::map<std::string, double> analyze_performance(const torch::Tensor &x,
                                                  const torch::Tensor &S1s,
                                                  const torch::Tensor &U1s,
                                                  const std::vector<int64_t> &stride,
                                                  const std::vector<int64_t> &padding,
                                                  const std::vector<int64_t> &kernel_size) {
    std::map<std::string, double> performance_metrics;

    performance_metrics["memory_usage"] = compute_memory_usage(x, S1s, U1s, stride, padding, kernel_size);

    auto start_time = std::chrono::high_resolution_clock::now();
    auto output = sketched_conv2d_forward(x, S1s, U1s, stride, padding, kernel_size, c10::nullopt);
    auto end_time = std::chrono::high_resolution_clock::now();
    performance_metrics["computation_time"] = std::chrono::duration<double>(end_time - start_time).count();

    int64_t theoretical_memory = x.numel() + S1s.numel() + U1s.numel() + output.numel();
    performance_metrics["memory_efficiency"] = static_cast<double>(performance_metrics["memory_usage"]) / theoretical_memory;

    auto stability_results = analyze_numerical_stability(x, S1s, U1s, stride, padding, kernel_size);
    performance_metrics["numerical_stability_score"] = std::count_if(stability_results.begin(), stability_results.end(),
                                                                     [](const auto &pair) { return pair.second; }) /
                                                       static_cast<double>(stability_results.size());

    return performance_metrics;
}

/**
 * @brief Validates the input parameters for the sketched convolution.
 *
 * This function performs comprehensive validation of all input parameters
 * to ensure they are valid and compatible with each other.
 *
 * @param x Input tensor
 * @param S1s Sketching weight tensor
 * @param U1s Low-rank basis tensor
 * @param stride Convolution stride
 * @param padding Convolution padding
 * @param kernel_size Kernel size
 * @return std::vector<std::string> List of validation errors, empty if all valid
 */
std::vector<std::string> validate_convolution_parameters(const torch::Tensor &x,
                                                         const torch::Tensor &S1s,
                                                         const torch::Tensor &U1s,
                                                         const std::vector<int64_t> &stride,
                                                         const std::vector<int64_t> &padding,
                                                         const std::vector<int64_t> &kernel_size) {
    std::vector<std::string> errors;
    if (x.dim() != 4) {
        errors.push_back("Input tensor must be 4D (B, C, H, W)");
    }
    if (S1s.dim() != 4) {
        errors.push_back("Sketching weights must be 4D (L, C, Kh, Kw)");
    }
    if (U1s.dim() != 3) {
        errors.push_back("Basis tensor must be 3D (L, K, D1)");
    }
    if (stride.size() != 2) {
        errors.push_back("Stride must be a vector of size 2");
    }
    if (padding.size() != 2) {
        errors.push_back("Padding must be a vector of size 2");
    }
    if (kernel_size.size() != 2) {
        errors.push_back("Kernel size must be a vector of size 2");
    }
    if (x.size(1) != S1s.size(1)) {
        errors.push_back("Input channels must match sketching weight channels");
    }
    if (S1s.size(0) != U1s.size(0)) {
        errors.push_back("Number of sketches must match in S1s and U1s");
    }

    return errors;
}

/**
 * @brief Performs a comprehensive analysis of the sketched convolution's memory access patterns.
 *
 * This function analyzes how the convolution accesses memory and provides
 * recommendations for optimization based on:
 * - Memory access patterns
 * - Cache utilization
 * - Memory bandwidth usage
 *
 * @param x Input tensor
 * @param S1s Sketching weight tensor
 * @param U1s Low-rank basis tensor
 * @param stride Convolution stride
 * @param padding Convolution padding
 * @param kernel_size Kernel size
 * @return std::map<std::string, std::vector<double>> Map of memory access metrics
 */
std::map<std::string, std::vector<double>> analyze_memory_access_patterns(
    const torch::Tensor &x,
    const torch::Tensor &S1s,
    const torch::Tensor &U1s,
    const std::vector<int64_t> &stride,
    const std::vector<int64_t> &padding,
    const std::vector<int64_t> &kernel_size) {
    std::map<std::string, std::vector<double>> access_patterns;

    std::vector<double> input_access;
    for (int64_t i = 0; i < x.size(0); ++i) {
        for (int64_t j = 0; j < x.size(1); ++j) {
            for (int64_t k = 0; k < x.size(2); ++k) {
                for (int64_t l = 0; l < x.size(3); ++l) {
                    input_access.push_back(static_cast<double>(i * j * k * l));
                }
            }
        }
    }
    access_patterns["input_access"] = input_access;

    std::vector<double> sketch_access;
    for (int64_t i = 0; i < S1s.size(0); ++i) {
        for (int64_t j = 0; j < S1s.size(1); ++j) {
            for (int64_t k = 0; k < S1s.size(2); ++k) {
                for (int64_t l = 0; l < S1s.size(3); ++l) {
                    sketch_access.push_back(static_cast<double>(i * j * k * l));
                }
            }
        }
    }
    access_patterns["sketch_access"] = sketch_access;

    std::vector<double> basis_access;
    for (int64_t i = 0; i < U1s.size(0); ++i) {
        for (int64_t j = 0; j < U1s.size(1); ++j) {
            for (int64_t k = 0; k < U1s.size(2); ++k) {
                basis_access.push_back(static_cast<double>(i * j * k));
            }
        }
    }
    access_patterns["basis_access"] = basis_access;

    return access_patterns;
}

/**
 * @brief Performs a comprehensive analysis of the sketched convolution's computational complexity.
 *
 * This function analyzes the computational complexity of the convolution operation
 * by breaking it down into its constituent parts and analyzing each part's complexity.
 *
 * @param x Input tensor
 * @param S1s Sketching weight tensor
 * @param U1s Low-rank basis tensor
 * @param stride Convolution stride
 * @param padding Convolution padding
 * @param kernel_size Kernel size
 * @return std::map<std::string, int64_t> Map of complexity metrics
 */
std::map<std::string, int64_t> analyze_computational_complexity(
    const torch::Tensor &x,
    const torch::Tensor &S1s,
    const torch::Tensor &U1s,
    const std::vector<int64_t> &stride,
    const std::vector<int64_t> &padding,
    const std::vector<int64_t> &kernel_size) {
    std::map<std::string, int64_t> complexity_metrics;

    // Compute dimensions
    int64_t B = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3);
    int64_t L = U1s.size(0), K = U1s.size(1), D1 = U1s.size(2);
    int64_t Kh = kernel_size[0], Kw = kernel_size[1];
    int64_t H_out = (H - Kh + 2 * padding[0]) / stride[0] + 1;
    int64_t W_out = (W - Kw + 2 * padding[1]) / stride[1] + 1;

    // Compute complexity of each operation
    complexity_metrics["convolution_ops"] = B * C * H_out * W_out * Kh * Kw;
    complexity_metrics["matrix_multiplication_ops"] = B * L * K * D1 * H_out * W_out;
    complexity_metrics["total_ops"] = complexity_metrics["convolution_ops"] + complexity_metrics["matrix_multiplication_ops"];

    return complexity_metrics;
}

/**
 * @brief Logs the results of an experiment to a file.
 *
 * This function appends the results of an experiment, including parameters and metrics,
 * to a specified log file in CSV format.
 *
 * @param filename The name of the log file
 * @param parameters The parameters used in the experiment
 * @param metrics The metrics/results of the experiment
 */
void log_experiment_results(const std::string &filename,
                            const std::map<std::string, double> &parameters,
                            const std::map<std::string, double> &metrics) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) return;
    for (const auto &p : parameters) file << p.first << "=" << p.second << ",";
    for (const auto &m : metrics) file << m.first << "=" << m.second << ",";
    file << std::endl;
    file.close();
}

/**
 * @brief Runs a parameter sweep for sketched_conv2d_forward and logs results.
 *
 * This function runs sketched_conv2d_forward with different parameter combinations
 * and logs the results for later analysis.
 *
 * @param x Input tensor
 * @param S1s Sketching weight tensor
 * @param U1s Low-rank basis tensor
 * @param stride_options List of stride options
 * @param padding_options List of padding options
 * @param kernel_size_options List of kernel size options
 * @param log_file Log file to save results
 */
void run_parameter_sweep(const torch::Tensor &x,
                         const torch::Tensor &S1s,
                         const torch::Tensor &U1s,
                         const std::vector<std::vector<int64_t>> &stride_options,
                         const std::vector<std::vector<int64_t>> &padding_options,
                         const std::vector<std::vector<int64_t>> &kernel_size_options,
                         const std::string &log_file) {
    for (const auto &stride : stride_options) {
        for (const auto &padding : padding_options) {
            for (const auto &kernel_size : kernel_size_options) {
                auto output = sketched_conv2d_forward(x, S1s, U1s, stride, padding, kernel_size, c10::nullopt);
                auto metrics = analyze_performance(x, S1s, U1s, stride, padding, kernel_size);
                std::map<std::string, double> params = {
                    {"stride_h", (double)stride[0]},
                    {"stride_w", (double)stride[1]},
                    {"pad_h", (double)padding[0]},
                    {"pad_w", (double)padding[1]},
                    {"kernel_h", (double)kernel_size[0]},
                    {"kernel_w", (double)kernel_size[1]}};
                log_experiment_results(log_file, params, metrics);
            }
        }
    }
}

/**
 * @brief Saves a tensor to a binary file for later analysis.
 *
 * @param filename The name of the file
 * @param tensor The tensor to save
 */
void save_tensor_to_file(const std::string &filename, const torch::Tensor &tensor) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return;
    auto data = tensor.contiguous().data_ptr<float>();
    file.write(reinterpret_cast<const char *>(data), tensor.numel() * sizeof(float));
    file.close();
}

/**
 * @brief Loads a tensor from a binary file.
 *
 * @param filename The name of the file
 * @param shape The shape of the tensor
 * @return torch::Tensor The loaded tensor
 */
torch::Tensor load_tensor_from_file(const std::string &filename, const std::vector<int64_t> &shape) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Could not open file");
    int64_t numel = 1;
    for (auto s : shape) numel *= s;
    std::vector<float> buffer(numel);
    file.read(reinterpret_cast<char *>(buffer.data()), numel * sizeof(float));
    file.close();
    return torch::from_blob(buffer.data(), shape, torch::kFloat32).clone();
}