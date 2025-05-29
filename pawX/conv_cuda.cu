#include <torch/extension.h>
#include "cuda_tensor_accessor.cuh"
#include "debug.cuh"
#include "kernel_launch.cuh"
#include "conv2d.h"

#define TILE_DIM 16 // Tile dimension for thread block

// CUDA kernel for sketched convolution
template <typename scalar_t>
__global__ void sketched_conv_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> S1s,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,
    int H_in, int W_in, int H_out, int W_out, int C_in, int LC,
    int K_h, int K_w, int stride_h, int stride_w, int padding_h, int padding_w) {

    // Shared memory tile dimensions using (TILE_DIM-1)*S + K
    int sh_tile_h = (TILE_DIM - 1) * stride_h + K_h;
    int sh_tile_w = (TILE_DIM - 1) * stride_w + K_w;
    
    extern __shared__ char shmem_char[];
    scalar_t* input_tile = reinterpret_cast<scalar_t*>(shmem_char);
    scalar_t* S1s_tile = input_tile + sh_tile_h * sh_tile_w;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int b_idx = blockIdx.z; // Batch index

    // Output pixel coordinates for this thread
    int out_x = blockIdx.x * TILE_DIM + tx;
    int out_y = blockIdx.y * TILE_DIM + ty;

    // Base global input coordinates for this thread's first shared memory element
    int global_in_y_base = blockIdx.y * TILE_DIM * stride_h - padding_h + ty;
    int global_in_x_base = blockIdx.x * TILE_DIM * stride_w - padding_w + tx;

    for (int term_idx = 0; term_idx < LC; ++term_idx) {
        scalar_t acc_sum = static_cast<scalar_t>(0.0);
        for (int c_in_idx = 0; c_in_idx < C_in; ++c_in_idx) {
            // Load input tile into shared memory
            for (int sh_y = ty, g_in_y = global_in_y_base; sh_y < sh_tile_h; sh_y += TILE_DIM, g_in_y += TILE_DIM) {
                for (int sh_x = tx, g_in_x = global_in_x_base; sh_x < sh_tile_w; sh_x += TILE_DIM, g_in_x += TILE_DIM) {
                    if (g_in_y >= 0 && g_in_y < H_in && g_in_x >= 0 && g_in_x < W_in) {
                        input_tile[sh_y * sh_tile_w + sh_x] = input[b_idx][c_in_idx][g_in_y][g_in_x];
                    } else {
                        input_tile[sh_y * sh_tile_w + sh_x] = static_cast<scalar_t>(0.0); // Pad
                    }
                }
            }
            // Load S1s tile into shared memory
            for (int kh = ty; kh < K_h; kh += TILE_DIM) {
                for (int kw = tx; kw < K_w; kw += TILE_DIM) {
                    S1s_tile[kh * K_w + kw] = S1s[term_idx][c_in_idx][kh][kw];
                }
            }
            __syncthreads();

            // Perform convolution from shared memory if thread is within output bounds
            if (out_y < H_out && out_x < W_out) {
                for (int kh = 0; kh < K_h; ++kh) {
                    for (int kw = 0; kw < K_w; ++kw) {
                        int sh_in_y = ty * stride_h + kh;
                        int sh_in_x = tx * stride_w + kw;

                        acc_sum += input_tile[sh_in_y * sh_tile_w + sh_in_x] * S1s_tile[kh * K_w + kw];
                    }
                }
            }
            __syncthreads();
        }

        if (out_y < H_out && out_x < W_out) {
            output[b_idx][term_idx][out_y][out_x] = acc_sum;
        }
    }
}

// Wrapper function
torch::Tensor sketched_conv_cuda(
    const torch::Tensor& x, const torch::Tensor& S1s,
    const std::vector<int64_t>& stride_vec, const std::vector<int64_t>& padding_vec,
    const std::vector<int64_t>& kernel_size_vec) {

    TORCH_CHECK(x.is_cuda() && S1s.is_cuda(), "Input tensors must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == S1s.scalar_type(), "Input and filter tensor types must match");

    const int64_t B = x.size(0), C_in = x.size(1), H_in = x.size(2), W_in = x.size(3);
    const int64_t LC = S1s.size(0);
    const int64_t K_h = kernel_size_vec[0], K_w = kernel_size_vec[1];
    const int64_t S_h = stride_vec[0], S_w = stride_vec[1];
    const int64_t P_h = padding_vec[0], P_w = padding_vec[1];

    const int64_t H_out = (H_in - K_h + 2 * P_h) / S_h + 1;
    const int64_t W_out = (W_in - K_w + 2 * P_w) / S_w + 1;
    TORCH_CHECK(H_out > 0 && W_out > 0, "Output dimensions must be positive");

    torch::Tensor output = torch::zeros({B, LC, H_out, W_out}, x.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "sketched_conv_cuda_kernel", ([&] {
        dim3 block_dim(TILE_DIM, TILE_DIM);
        dim3 grid_dim((W_out + TILE_DIM - 1) / TILE_DIM, (H_out + TILE_DIM - 1) / TILE_DIM, B);

        // Shared memory tile dimensions using (TILE_DIM-1)*S + K
        size_t sh_tile_h = (TILE_DIM - 1) * S_h + K_h;
        size_t sh_tile_w = (TILE_DIM - 1) * S_w + K_w;
        size_t shmem_size = sh_tile_h * sh_tile_w * sizeof(scalar_t) + K_h * K_w * sizeof(scalar_t);
        
        // Handle case where calculated tile size is zero (e.g. K_h/K_w is 0)
        if (sh_tile_h == 0 || sh_tile_w == 0) {
             if (K_h > 0 && K_w > 0) { // Only an issue if kernel should be valid
                fprintf(stderr, "Warning: Calculated shared memory tile dimension is zero (h:%zu, w:%zu) but kernel size is non-zero. Check TILE_DIM, stride, kernel_size.\n", sh_tile_h, sh_tile_w);
             }
             // If K_h or K_w is 0, then output might be weird but shmem_size might be 0.
             // Kernel launch with 0 shared memory is valid if kernel doesn't try to use it or if TILE_DIM=1 and S=0 (not practical).
             // For safety, if either dimension is 0, shmem_size will be 0.
        }

        int dev_id;
        cudaGetDevice(&dev_id);
        int max_shmem_per_block;
        cudaDeviceGetAttribute(&max_shmem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, dev_id);
        if (shmem_size > max_shmem_per_block) {
            throw std::runtime_error("Requested shared memory size exceeds device limit.");
        }

        sketched_conv_kernel<scalar_t><<<grid_dim, block_dim, shmem_size>>>(
            x.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            S1s.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            H_in, W_in, H_out, W_out, C_in, LC, K_h, K_w, S_h, S_w, P_h, P_w);
    }));
    return output;
}