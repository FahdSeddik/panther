#include <torch/extension.h>

#define TILE_DIM 16

template <typename scalar_t>
__global__ void sklinear_forward_intermediate(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,  // gets loaded once
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> S1s,    // gets loaded batch / TILE times
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> U2s,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,  // 2 * num * batch * low_rank
    int batch_size, int input_dim, int low_rank_dim) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int term = blockIdx.y;
    int batch = ty + blockIdx.x * TILE_DIM;  // Batch index.
    // if (batch >= batch_size) return;
    extern __shared__ float shData[];  // 2 * TILE_DIM * low_rank_dim
    scalar_t* sharedIntermediate = (scalar_t*)shData;

    // set all intermediate to 0
    for (int i = tx + ty * TILE_DIM; i < TILE_DIM * low_rank_dim * 2; i += TILE_DIM * TILE_DIM) {
        sharedIntermediate[i] = 0;
    }
    __syncthreads();
    for (int i = tx; i < input_dim; i += TILE_DIM) {
        scalar_t input_val = batch < batch_size ? input[batch][i] : 0;
        for (int j = 0; j < low_rank_dim; j++) {
            auto s1_val = S1s[term][i][j], u2_val = U2s[term][i][j];
            auto intermediate1 = input_val * s1_val, intermediate2 = input_val * u2_val;
            for (int offset = 8; offset > 0; offset >>= 1) {
                intermediate1 += __shfl_down_sync(0xFFFFFFFF, intermediate1, offset);
                intermediate2 += __shfl_down_sync(0xFFFFFFFF, intermediate2, offset);
            }
            if (tx == 0) {  // first and 16 thread in each warp
                sharedIntermediate[ty * low_rank_dim + j] += intermediate1;
                sharedIntermediate[TILE_DIM * low_rank_dim + ty * low_rank_dim + j] += intermediate2;
            }
        }
    }
    __syncthreads();
    // write output
    for (int i = tx + ty * TILE_DIM; i < TILE_DIM * low_rank_dim; i += TILE_DIM * TILE_DIM) {
        int localBatch = i / low_rank_dim;
        int localDim = i % low_rank_dim;
        int batchIdx = localBatch + blockIdx.x * TILE_DIM;
        if (batchIdx < batch_size) {
            output[0][term][batchIdx][localDim] = sharedIntermediate[localBatch * low_rank_dim + localDim];
            output[1][term][batchIdx][localDim] = sharedIntermediate[TILE_DIM * low_rank_dim + localBatch * low_rank_dim + localDim];
        }
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

    // Define grid dimensions: one block per (batch, output) pair.
    dim3 grid((batch_size + TILE_DIM - 1) / TILE_DIM, num_terms);
    dim3 block(TILE_DIM, TILE_DIM);
    int shared_mem_size = 2 * TILE_DIM * low_rank_dim * input.element_size();

    // allocate intermediate output tensor contigously
    auto output_intermediate = torch::zeros({2, num_terms, batch_size, low_rank_dim}, input.options()).contiguous();
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