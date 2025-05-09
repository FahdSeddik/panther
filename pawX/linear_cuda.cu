#include <torch/extension.h>

#include "linear.h"

#define TILE_DIM 16

template <typename scalar_t>
__global__ void sklinear_forward_intermediate(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,  // gets loaded term number of times
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> S1s,    // gets loaded batch / TILE times
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> U2s,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,  // 2 * num * batch * low_rank
    int batch_size, int input_dim, int low_rank_dim) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int termIdx = blockIdx.z;
    int batchIdx = blockIdx.y * TILE_DIM + ty;    // row
    int lowRankIdx = blockIdx.x * TILE_DIM + tx;  // col
    extern __shared__ float shData[];
    scalar_t* sharedS1 = (scalar_t*)shData;
    scalar_t* sharedU2 = sharedS1 + TILE_DIM * TILE_DIM;
    scalar_t* sharedInput = sharedU2 + TILE_DIM * TILE_DIM;

    scalar_t sum1 = 0, sum2 = 0;
    for (int stride = 0; stride < input_dim; stride += TILE_DIM) {
        if (lowRankIdx < low_rank_dim && stride + ty < input_dim) {
            sharedS1[ty * TILE_DIM + tx] = S1s[termIdx][stride + ty][lowRankIdx];
            sharedU2[ty * TILE_DIM + tx] = U2s[termIdx][stride + ty][lowRankIdx];
        } else {
            sharedS1[ty * TILE_DIM + tx] = 0;
            sharedU2[ty * TILE_DIM + tx] = 0;
        }
        // Load intermediate results into shared memory.
        if (batchIdx < batch_size && stride + tx < input_dim) {
            sharedInput[ty * TILE_DIM + tx] = input[batchIdx][stride + tx];
        } else {
            sharedInput[ty * TILE_DIM + tx] = 0;
        }
        __syncthreads();
        if (batchIdx < batch_size && lowRankIdx < low_rank_dim) {
            for (int i = 0; i < TILE_DIM; i++) {
                auto s1_val = sharedS1[i * TILE_DIM + tx];
                auto u2_val = sharedU2[i * TILE_DIM + tx];
                auto input_val = sharedInput[ty * TILE_DIM + i];
                sum1 += s1_val * input_val;
                sum2 += u2_val * input_val;
            }
        }
        __syncthreads();
    }

    if (batchIdx < batch_size && lowRankIdx < low_rank_dim) {
        // Write the result to the output tensor.
        output[0][termIdx][batchIdx][lowRankIdx] = sum1;
        output[1][termIdx][batchIdx][lowRankIdx] = sum2;
    }
}

template <typename scalar_t>
__global__ void sklinear_forward_output(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> intermediate,  // 2 x num_terms x batch_size x low_rank_dim
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> S2s,           // term x low_rank_dim x output_dim
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> U1s,           // term x low_rank_dim x output_dim
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> bias,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,  // batch x output_dim
    int batch_size, int input_dim, int num_terms, int low_rank_dim, int output_dim) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int batchIdx = blockIdx.y * TILE_DIM + ty;   // row
    int outputIdx = blockIdx.x * TILE_DIM + tx;  // col
    extern __shared__ float shData[];
    scalar_t* sharedS2 = (scalar_t*)shData;
    scalar_t* sharedU1 = sharedS2 + TILE_DIM * TILE_DIM;
    scalar_t* sharedInter1 = sharedU1 + TILE_DIM * TILE_DIM;
    scalar_t* sharedInter2 = sharedInter1 + TILE_DIM * TILE_DIM;
    scalar_t* sharedBias = sharedInter2 + TILE_DIM * TILE_DIM;
    scalar_t sum = 0;

    if (outputIdx < output_dim && ty == 0) {
        sharedBias[tx] = bias[outputIdx];
    }

    for (int term = 0; term < num_terms; term++) {
        for (int stride = 0; stride < low_rank_dim; stride += TILE_DIM) {
            // Load S2s and U1s into shared memory.
            if (outputIdx < output_dim && stride + ty < low_rank_dim) {
                sharedS2[ty * TILE_DIM + tx] = S2s[term][stride + ty][outputIdx];
                sharedU1[ty * TILE_DIM + tx] = U1s[term][stride + ty][outputIdx];
            } else {
                sharedS2[ty * TILE_DIM + tx] = 0;
                sharedU1[ty * TILE_DIM + tx] = 0;
            }
            // Load intermediate results into shared memory.
            if (batchIdx < batch_size && stride + tx < low_rank_dim) {
                sharedInter1[ty * TILE_DIM + tx] = intermediate[0][term][batchIdx][stride + tx];
                sharedInter2[ty * TILE_DIM + tx] = intermediate[1][term][batchIdx][stride + tx];
            } else {
                sharedInter1[ty * TILE_DIM + tx] = 0;
                sharedInter2[ty * TILE_DIM + tx] = 0;
            }
            __syncthreads();
            if (batchIdx < batch_size && outputIdx < output_dim) {
                for (int i = 0; i < TILE_DIM; i++) {
                    auto inter1_val = sharedInter1[ty * TILE_DIM + i];
                    auto inter2_val = sharedInter2[ty * TILE_DIM + i];
                    auto s2_val = sharedS2[i * TILE_DIM + tx];
                    auto u1_val = sharedU1[i * TILE_DIM + tx];
                    sum += inter1_val * u1_val + inter2_val * s2_val;
                }
            }
            __syncthreads();
        }
    }

    if (batchIdx < batch_size && outputIdx < output_dim) {
        // Write the result to the output tensor.
        output[batchIdx][outputIdx] = sum / (static_cast<scalar_t>(2 * num_terms)) + sharedBias[tx];
    }
}

// Wrapper function that sets up grid dimensions, shared memory, and kernel launch.
torch::Tensor sketched_linear_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    const torch::Tensor& bias) {
    // Get dimensions.
    int batch_size = input.size(0);
    int input_dim = input.size(1);
    int num_terms = S1s.size(0);
    int low_rank_dim = S1s.size(2);  // also U1s.dim(2) and S2s.dim(1)
    int output_dim = S2s.size(2);

    // Create output tensor.
    auto output = torch::zeros({batch_size, output_dim}, input.options());

    dim3 grid((low_rank_dim + TILE_DIM - 1) / TILE_DIM, (batch_size + TILE_DIM - 1) / TILE_DIM, num_terms);
    dim3 block(TILE_DIM, TILE_DIM);
    int shared_mem_size = 3 * TILE_DIM * TILE_DIM * input.element_size();

    // allocate intermediate output tensor contigously
    auto output_intermediate = torch::zeros({2, num_terms, batch_size, low_rank_dim}, input.options().memory_format(torch::MemoryFormat::Contiguous));
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "sklinear_forward_intermediate",
        ([&] {
            sklinear_forward_intermediate<scalar_t><<<grid, block, shared_mem_size>>>(
                input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                S1s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                U2s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                output_intermediate.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                batch_size, input_dim, low_rank_dim);
        }));

    dim3 grid2((output_dim + TILE_DIM - 1) / TILE_DIM, (batch_size + TILE_DIM - 1) / TILE_DIM);
    dim3 block2(TILE_DIM, TILE_DIM);
    int shared_mem_size2 = (4 * TILE_DIM * TILE_DIM + TILE_DIM) * input.element_size();

    // Launch the kernel.
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "sklinear_forward_output",
        ([&] {
            sklinear_forward_output<scalar_t><<<grid2, block2, shared_mem_size2>>>(
                output_intermediate.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                S2s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                U1s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                batch_size, input_dim, num_terms, low_rank_dim, output_dim);
        }));

    return output;
}

std::vector<torch::Tensor> sketched_linear_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s) {
    // raise not implemented error
    throw std::runtime_error("sketched_linear_backward_cuda not implemented yet.");
    return std::vector<torch::Tensor>{};
}