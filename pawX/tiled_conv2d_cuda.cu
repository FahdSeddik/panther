#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "tiled_conv2d_cuda.h"

// Constants for tile size optimization
constexpr int TILE_SIZE = 16;    // 16x16 tile size
constexpr int BLOCK_SIZE = 256;  // Threads per block

/**
 * @brief CUDA kernel for performing 2D convolution using tiling and shared memory.
 *
 * This kernel implements a tiled 2D convolution for batched, multi-channel input tensors.
 * It loads tiles of the input and kernel weights into shared memory to improve memory access efficiency.
 * The kernel supports configurable stride and padding.
 *
 * @tparam scalar_t Data type of input, weight, and output tensors (e.g., float, double).
 *
 * @param input    Pointer to input tensor of shape [B, C, H, W].
 * @param weight   Pointer to weight tensor of shape [K, C, Kh, Kw].
 * @param output   Pointer to output tensor of shape [B, K, H_out, W_out].
 * @param B        Batch size.
 * @param C        Number of input channels.
 * @param H        Input height.
 * @param W        Input width.
 * @param K        Number of output channels.
 * @param Kh       Kernel height.
 * @param Kw       Kernel width.
 * @param H_out    Output height.
 * @param W_out    Output width.
 * @param stride_h Stride along the height dimension.
 * @param stride_w Stride along the width dimension.
 * @param pad_h    Padding along the height dimension.
 * @param pad_w    Padding along the width dimension.
 *
 * @details
 * - Uses shared memory to cache input tiles and kernel weights for each block.
 * - Each thread computes one output element for a given batch and output channel.
 * - Handles boundary conditions for input padding.
 * - Assumes TILE_SIZE is defined elsewhere and determines the tile/block size.
 * - Synchronization is used to ensure all threads have loaded shared memory before computation.
 */
// CUDA kernel for tiled 2D convolution
template <typename scalar_t>
__global__ void tiled_conv2d_kernel(
    const scalar_t* __restrict__ input,   // [B, C, H, W]
    const scalar_t* __restrict__ weight,  // [K, C, Kh, Kw]
    scalar_t* __restrict__ output,        // [B, K, H_out, W_out]
    const int B,
    const int C,
    const int H,
    const int W,
    const int K,
    const int Kh,
    const int Kw,
    const int H_out,
    const int W_out,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w) {
    // Shared memory for input tile and weight
    __shared__ scalar_t input_tile[TILE_SIZE + 2][TILE_SIZE + 2];
    __shared__ scalar_t weight_tile[TILE_SIZE][TILE_SIZE];

    const int out_h = blockIdx.y * TILE_SIZE + threadIdx.y;  // output height index
    const int out_w = blockIdx.x * TILE_SIZE + threadIdx.x;  // output width index
    const int b = blockIdx.z;

    scalar_t sum = 0.0f;

    if (out_h < H_out && out_w < W_out) {
        for (int c = 0; c < C; c++) {  // loop over input channels
            for (int i = threadIdx.y; i < TILE_SIZE + 2; i += blockDim.y) {
                for (int j = threadIdx.x; j < TILE_SIZE + 2; j += blockDim.x) {
                    const int in_h = out_h * stride_h - pad_h + i;  // input height index
                    const int in_w = out_w * stride_w - pad_w + j;  // input width index

                    if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                        input_tile[i][j] = input[b * C * H * W + c * H * W + in_h * W + in_w];
                    } else {
                        input_tile[i][j] = 0.0f;
                    }
                }
            }

            for (int k = 0; k < K; k++) {  // loop over output channels
                for (int i = threadIdx.y; i < Kh; i += blockDim.y) {
                    for (int j = threadIdx.x; j < Kw; j += blockDim.x) {
                        weight_tile[i][j] = weight[k * C * Kh * Kw + c * Kh * Kw + i * Kw + j];
                    }
                }

                __syncthreads();

                for (int i = 0; i < Kh; i++) {
                    for (int j = 0; j < Kw; j++) {
                        sum += input_tile[threadIdx.y + i][threadIdx.x + j] * weight_tile[i][j];
                    }
                }

                __syncthreads();
            }
        }

        if (out_h < H_out && out_w < W_out) {
            output[b * K * H_out * W_out + blockIdx.z * H_out * W_out + out_h * W_out + out_w] = sum;
        }
    }
}

/**
 * @brief Performs a 2D convolution operation on the input tensor using tiled CUDA kernels.
 *
 * This function computes the forward pass of a 2D convolution with optional bias addition,
 * using a tiled CUDA kernel for improved memory access and performance. The function supports
 * various floating-point types, including float, double, half, and bfloat16.
 *
 * @param input      Input tensor of shape (B, C, H, W), where
 *                   B: batch size,
 *                   C: number of input channels,
 *                   H: input height,
 *                   W: input width.
 * @param weight     Convolution kernel tensor of shape (K, C, Kh, Kw), where
 *                   K: number of output channels,
 *                   Kh: kernel height,
 *                   Kw: kernel width.
 * @param stride     Vector of 2 integers specifying the stride (stride_h, stride_w).
 * @param padding    Vector of 2 integers specifying the padding (pad_h, pad_w).
 * @param kernel_size Vector of 2 integers specifying the kernel size (Kh, Kw).
 * @param bias_opt   Optional bias tensor of shape (K).
 *
 * @return torch::Tensor
 *         Output tensor of shape (B, K, H_out, W_out), where
 *         H_out = (H - Kh + 2 * pad_h) / stride_h + 1,
 *         W_out = (W - Kw + 2 * pad_w) / stride_w + 1.
 *
 * @note The function launches a CUDA kernel with a grid and block configuration determined
 *       by the output dimensions and a predefined TILE_SIZE. The bias, if provided, is
 *       broadcast and added to the output tensor after the convolution.
 */
torch::Tensor tiled_conv2d_cuda_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& kernel_size,
    const c10::optional<torch::Tensor>& bias_opt) {
    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int K = weight.size(0);
    const int Kh = kernel_size[0];
    const int Kw = kernel_size[1];

    const int H_out = (H - Kh + 2 * padding[0]) / stride[0] + 1;
    const int W_out = (W - Kw + 2 * padding[1]) / stride[1] + 1;

    auto output = torch::zeros({B, K, H_out, W_out}, input.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (W_out + TILE_SIZE - 1) / TILE_SIZE,
        (H_out + TILE_SIZE - 1) / TILE_SIZE,
        B);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "tiled_conv2d_cuda_forward",
        [&]() {
            tiled_conv2d_kernel<scalar_t><<<grid, block>>>(
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                B, C, H, W,
                K, Kh, Kw,
                H_out, W_out,
                stride[0], stride[1],
                padding[0], padding[1]);
        });

    if (bias_opt.has_value()) {
        auto bias = bias_opt.value();
        output.add_(bias.view({1, bias.size(0), 1, 1}));
    }

    return output;
}
