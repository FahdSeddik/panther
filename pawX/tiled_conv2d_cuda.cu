#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "tiled_conv2d_cuda.h"

// Constants for tile size optimization
constexpr int TILE_SIZE = 16;    // 16x16 tile size
constexpr int BLOCK_SIZE = 256;  // Threads per block

// CUDA kernel for tiled 2D convolution
template <typename scalar_t>
__global__ void tiled_conv2d_kernel(
    const scalar_t* __restrict__ input,   // [B, C, H, W]
    const scalar_t* __restrict__ weight,  // [K, C, Kh, Kw]
    scalar_t* __restrict__ output,        // [B, K, H_out, W_out]
    const int B,                          // batch size
    const int C,                          // input channels
    const int H,                          // input height
    const int W,                          // input width
    const int K,                          // output channels
    const int Kh,                         // kernel height
    const int Kw,                         // kernel width
    const int H_out,                      // output height
    const int W_out,                      // output width
    const int stride_h,                   // vertical stride
    const int stride_w,                   // horizontal stride
    const int pad_h,                      // vertical padding
    const int pad_w) {                    // horizontal padding
    // Shared memory for input tile and weight
    __shared__ scalar_t input_tile[TILE_SIZE + 2][TILE_SIZE + 2];  // [TILE_SIZE + 2, TILE_SIZE + 2]
    __shared__ scalar_t weight_tile[TILE_SIZE][TILE_SIZE];         // [TILE_SIZE, TILE_SIZE]

    // Calculate output position
    const int out_h = blockIdx.y * TILE_SIZE + threadIdx.y;  // output height index
    const int out_w = blockIdx.x * TILE_SIZE + threadIdx.x;  // output width index
    const int b = blockIdx.z;                                // batch index

    // Initialize accumulator
    scalar_t sum = 0.0f;

    // Load input tile into shared memory
    if (out_h < H_out && out_w < W_out) {
        for (int c = 0; c < C; c++) {  // loop over input channels
            // Load input tile
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

            // Load weight tile
            for (int k = 0; k < K; k++) {  // loop over output channels
                for (int i = threadIdx.y; i < Kh; i += blockDim.y) {
                    for (int j = threadIdx.x; j < Kw; j += blockDim.x) {
                        weight_tile[i][j] = weight[k * C * Kh * Kw + c * Kh * Kw + i * Kw + j];
                    }
                }

                __syncthreads();

                // Compute convolution
                for (int i = 0; i < Kh; i++) {
                    for (int j = 0; j < Kw; j++) {
                        sum += input_tile[threadIdx.y + i][threadIdx.x + j] * weight_tile[i][j];
                    }
                }

                __syncthreads();
            }
        }

        // Write output
        if (out_h < H_out && out_w < W_out) {
            output[b * K * H_out * W_out + blockIdx.z * H_out * W_out + out_h * W_out + out_w] = sum;
        }
    }
}

// Wrapper function for the CUDA kernel
torch::Tensor tiled_conv2d_cuda_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& kernel_size,
    const c10::optional<torch::Tensor>& bias_opt) {
    // Get dimensions
    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int K = weight.size(0);
    const int Kh = kernel_size[0];
    const int Kw = kernel_size[1];

    // Calculate output dimensions
    const int H_out = (H - Kh + 2 * padding[0]) / stride[0] + 1;
    const int W_out = (W - Kw + 2 * padding[1]) / stride[1] + 1;

    // Create output tensor
    auto output = torch::zeros({B, K, H_out, W_out}, input.options());

    // Calculate grid and block dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (W_out + TILE_SIZE - 1) / TILE_SIZE,
        (H_out + TILE_SIZE - 1) / TILE_SIZE,
        B);

    // Dispatch based on input type
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

    // Add bias if present
    if (bias_opt.has_value()) {
        auto bias = bias_opt.value();
        output.add_(bias.view({1, bias.size(0), 1, 1}));
    }

    return output;
}
