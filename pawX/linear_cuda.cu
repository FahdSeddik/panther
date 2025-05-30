#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "cuda_tensor_accessor.cuh"
#include "debug.cuh"
#include "kernel_launch.cuh"
#include "linear.h"
#define TILE_DIM 16

/**
 * @brief CUDA kernel for computing intermediate results in a linear transformation with low-rank factorization.
 *
 * This kernel processes a batch of input data using two sets of low-rank factors (S1s and U2s) and stores
 * the intermediate results in a 5D output tensor. Shared memory is used to cache tiles of the input and factor
 * matrices for efficient access by threads within a block.
 *
 * @tparam scalar_t The data type of the input and output tensors (e.g., float, double).
 * @param input 2D flexible tensor accessor for the input data of shape [batch_size, input_dim].
 * @param S1s 3D tensor accessor for the first set of low-rank factors of shape [num_terms, input_dim, low_rank_dim].
 * @param U2s 3D tensor accessor for the second set of low-rank factors of shape [num_terms, input_dim, low_rank_dim].
 * @param output 5D tensor accessor for the output, storing intermediate results of shape [numTilesI, 2, num_terms, batch_size, low_rank_dim].
 * @param batch_size Number of batches in the input.
 * @param input_dim Dimensionality of the input features.
 * @param low_rank_dim Dimensionality of the low-rank factors.
 * @param num_terms Number of terms (e.g., for multi-term expansions).
 *
 * Each thread block processes a TILE_DIM x TILE_DIM tile of the input and factor matrices.
 * Shared memory is used to store tiles of S1s, U2s, and input for reuse within the block.
 * For each term, the kernel computes two sums (sum1 and sum2) per batch and low-rank index,
 * corresponding to the products with S1s and U2s, and writes them to the output tensor.
 */
template <typename scalar_t>
__global__ void sklinear_forward_intermediate(
    const FlexibleTensorAccessor<scalar_t, 2> input,                                 // gets loaded term number of times
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> S1s,  // gets loaded batch / TILE times
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> U2s,
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> output,  // numTilesI, 2 * num * batch * low_rank
    int batch_size, int input_dim, int low_rank_dim, int num_terms) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int batchIdx = blockIdx.y * TILE_DIM + ty;    // row
    int lowRankIdx = blockIdx.x * TILE_DIM + tx;  // col
    int stride = blockIdx.z * TILE_DIM;
    extern __shared__ float shData[];
    scalar_t* sharedS1 = (scalar_t*)shData;
    scalar_t* sharedU2 = sharedS1 + TILE_DIM * TILE_DIM;
    scalar_t* sharedInput = sharedU2 + TILE_DIM * TILE_DIM;

    scalar_t sum1 = 0, sum2 = 0;
    for (int term = 0; term < num_terms; term++) {
        sum1 = 0, sum2 = 0;
        if (lowRankIdx < low_rank_dim && stride + ty < input_dim) {
            sharedS1[ty * TILE_DIM + tx] = S1s[term][stride + ty][lowRankIdx];
            sharedU2[ty * TILE_DIM + tx] = U2s[term][stride + ty][lowRankIdx];
        } else {
            sharedS1[ty * TILE_DIM + tx] = 0;
            sharedU2[ty * TILE_DIM + tx] = 0;
        }
        // Load intermediate results into shared memory.
        if (term == 0) {
            if (batchIdx < batch_size && stride + tx < input_dim) {
                sharedInput[ty * TILE_DIM + tx] = input(batchIdx, stride + tx);
            } else {
                sharedInput[ty * TILE_DIM + tx] = 0;
            }
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

        if (batchIdx < batch_size && lowRankIdx < low_rank_dim) {
            // Write the result to the output tensor.
            output[blockIdx.z][0][term][batchIdx][lowRankIdx] = sum1;
            output[blockIdx.z][1][term][batchIdx][lowRankIdx] = sum2;
        }
    }
}

/**
 * @brief CUDA kernel for computing the forward pass of a low-rank linear layer with optional bias.
 *
 * This kernel computes the output of a linear transformation using low-rank factorization,
 * processing the input in tiles for efficient memory access and parallelization. It supports
 * optional bias addition and leverages shared memory to cache intermediate values and weights.
 *
 * @tparam scalar_t Data type of the computation (e.g., float, double).
 * @param intermediate [in] 4D tensor accessor (2 x num_terms x batch_size x low_rank_dim) containing intermediate results.
 * @param S2s [in] 3D tensor accessor (term x low_rank_dim x output_dim) for S2 weight matrices.
 * @param U1s [in] 3D tensor accessor (term x low_rank_dim x output_dim) for U1 weight matrices.
 * @param bias [in] Pointer to bias vector (length: output_dim), can be nullptr if hasBias is false.
 * @param hasBias [in] Boolean flag indicating whether to add bias to the output.
 * @param output [out] 2D tensor accessor (batch_size x output_dim) to store the computed output.
 * @param batch_size [in] Number of input samples in the batch.
 * @param input_dim [in] Input feature dimension (not directly used in this kernel).
 * @param num_terms [in] Number of low-rank terms in the decomposition.
 * @param low_rank_dim [in] Rank of the low-rank approximation.
 * @param output_dim [in] Output feature dimension.
 *
 * The kernel uses shared memory to load tiles of S2s, U1s, and intermediate results for each term,
 * and accumulates the result for each (batch, output) pair. If bias is enabled, it is added to the
 * final output. The result is normalized by (2 * num_terms).
 *
 * Grid and block dimensions should be set such that each thread computes one output element
 * (batchIdx, outputIdx) per tile.
 */
template <typename scalar_t>
__global__ void sklinear_forward_output(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> intermediate,  // 2 x num_terms x batch_size x low_rank_dim
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> S2s,           // term x low_rank_dim x output_dim
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> U1s,           // term x low_rank_dim x output_dim
    const scalar_t* bias,
    bool hasBias,
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
    scalar_t* sharedBias = nullptr;
    if (hasBias) {
        sharedBias = sharedInter2 + TILE_DIM * TILE_DIM;
    }
    scalar_t sum = 0;

    if (hasBias) {
        if (outputIdx < output_dim && ty == 0) {
            sharedBias[tx] = bias[outputIdx];
        }
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
        if (hasBias) {
            output[batchIdx][outputIdx] = sum / (static_cast<scalar_t>(2 * num_terms)) + sharedBias[tx];
        } else {
            output[batchIdx][outputIdx] = sum / (static_cast<scalar_t>(2 * num_terms));
        }
    }
}

// Wrapper function that sets up grid dimensions, shared memory, and kernel launch.
/**
 * @brief Performs the forward pass of a sketched linear layer using CUDA.
 *
 * This function computes a linear transformation of the input tensor using a low-rank sketching technique.
 * It supports both single-term and multi-term sketching, and optionally adds a bias.
 * The computation is dispatched to custom CUDA kernels for efficiency, depending on the rank dimension.
 *
 * @param input      Input tensor of shape (batch_size, input_dim).
 * @param S1s        Sketch matrix tensor of shape (num_terms, input_dim, low_rank_dim).
 * @param S2s        Sketch matrix tensor of shape (num_terms, low_rank_dim, output_dim).
 * @param U1s        Sketch matrix tensor of shape (num_terms, low_rank_dim, output_dim).
 * @param U2s        Sketch matrix tensor of shape (num_terms, input_dim, low_rank_dim).
 * @param bias       Optional bias tensor of shape (output_dim).
 * @return           Output tensor of shape (batch_size, output_dim).
 *
 * The function chooses between two computation strategies:
 *   - For low_rank_dim <= 96, it launches a custom CUDA kernel for the output computation.
 *   - For low_rank_dim > 96, it computes the result using batched matrix multiplications and averages the results.
 *
 * Intermediate results are stored in a contiguous tensor for efficient computation.
 * The function supports both float and half precision types.
 */
torch::Tensor sketched_linear_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    c10::optional<torch::Tensor> bias) {
    // Get dimensions.
    int batch_size = input.size(0);
    int input_dim = input.size(1);
    int num_terms = S1s.size(0);
    int low_rank_dim = S1s.size(2);  // also U1s.dim(2) and S2s.dim(1)
    int output_dim = S2s.size(2);
    dim3 block(TILE_DIM, TILE_DIM);

    // Create output tensor.
    torch::Tensor output_intermediate;
    if (num_terms > 1) {
        int numTilesI = (input_dim + TILE_DIM - 1) / TILE_DIM;
        dim3 grid((low_rank_dim + TILE_DIM - 1) / TILE_DIM, (batch_size + TILE_DIM - 1) / TILE_DIM, numTilesI);

        // allocate intermediate output tensor contigously
        output_intermediate = torch::zeros({numTilesI, 2, num_terms, batch_size, low_rank_dim}, S1s.options());
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(),
            "sklinear_forward_intermediate",
            ([&] {
                sklinear_forward_intermediate<scalar_t><<<grid, block, 3 * TILE_DIM * TILE_DIM * sizeof(scalar_t)>>>(
                    tensor_utils::buildAccessor<scalar_t, 2>(input),
                    S1s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    U2s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    output_intermediate.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                    batch_size, input_dim, low_rank_dim, num_terms);
            }));
        output_intermediate = output_intermediate.sum(0).contiguous();
    } else {
        auto input_expanded = input.unsqueeze(0);
        output_intermediate = torch::stack({input_expanded.bmm(S1s), input_expanded.bmm(U2s)}, 0);
    }

    if (low_rank_dim <= 96) {
        dim3 grid2((output_dim + TILE_DIM - 1) / TILE_DIM, (batch_size + TILE_DIM - 1) / TILE_DIM);
        auto output = torch::zeros({batch_size, output_dim}, S1s.options());
        // Launch the kernel.
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.scalar_type(),
            "sklinear_forward_output",
            ([&] {
                sklinear_forward_output<scalar_t><<<grid2, block, (4 * TILE_DIM * TILE_DIM + ((bias.has_value()) ? TILE_DIM : 0)) * sizeof(scalar_t)>>>(
                    output_intermediate.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                    S2s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    U1s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                    bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr,
                    bias.has_value(),
                    output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    batch_size, input_dim, num_terms, low_rank_dim, output_dim);
            }));

        return output;
    } else {
        auto result = (output_intermediate[0].bmm(U1s) + output_intermediate[1].bmm(S2s)).mean(0).div(2.0f);
        if (bias.has_value()) {
            result.add_(bias.value());
        }
        return result;
    }
}

/**
 * @brief CUDA kernel for computing intermediate gradients for a custom linear layer with low-rank structure.
 *
 * This kernel computes two sets of intermediate gradients for each term in a sum, using shared memory tiling for efficiency.
 * It processes the backward pass for a low-rank linear operation, where the output is decomposed into terms involving
 * matrices U1s and S2s. The results are stored in grad_intermediate for further processing.
 *
 * @tparam scalar_t Data type of the tensors (e.g., float, double).
 * @param grad_output [B, O] Gradient of the output with respect to the loss.
 * @param U1s [T, R, O] Low-rank factor matrix U1 for each term.
 * @param S2s [T, R, O] Low-rank factor matrix S2 for each term.
 * @param grad_intermediate [numTilesO, 2, T, B, R] Output tensor to store intermediate gradients for each tile, term, batch, and rank.
 * @param B Batch size.
 * @param O Output dimension.
 * @param T Number of terms in the sum (e.g., for low-rank decomposition).
 * @param R Low-rank dimension.
 *
 * @details
 * The kernel uses shared memory to tile portions of the input and factor matrices for efficient memory access.
 * For each term, it loads the relevant slices of U1s and S2s into shared memory, as well as the corresponding
 * slice of grad_output (only for the first term). Each thread computes the sum of products for its assigned
 * batch and rank indices, accumulating two separate sums (sum1 and sum2) corresponding to U1s and S2s.
 * The results are written to the grad_intermediate tensor for use in subsequent computations.
 *
 * Shared memory layout:
 *   - subTileA: Holds a tile of grad_output.
 *   - subTileB1: Holds a tile of U1s for the current term.
 *   - subTileB2: Holds a tile of S2s for the current term.
 *
 * Grid/block configuration:
 *   - blockIdx.x: Tiles over the low-rank dimension (R).
 *   - blockIdx.y: Tiles over the batch dimension (B).
 *   - blockIdx.z: Tiles over the output dimension (O).
 *   - threadIdx.x, threadIdx.y: Indices within a tile.
 *
 * Synchronization:
 *   - __syncthreads() is used to ensure all threads have loaded shared memory before computation.
 */
template <typename scalar_t>
__global__ void sklinear_backward_intermediate(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_output,  // [B, O]
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> U1s,          // [T, R, O]
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> S2s,          // [T, R, O]
    torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> grad_intermediate,  // [numTilesO, 2, T, B, R]
    int B, int O, int T, int R) {
    extern __shared__ float shData[];
    scalar_t* sharedData = (scalar_t*)shData;
    scalar_t* subTileA = sharedData;
    scalar_t* subTileB1 = sharedData + TILE_DIM * TILE_DIM;
    scalar_t* subTileB2 = subTileB1 + TILE_DIM * TILE_DIM;

    int tx = threadIdx.x, ty = threadIdx.y;
    int batchIdx = blockIdx.y * TILE_DIM + ty;    // row
    int lowRankIdx = blockIdx.x * TILE_DIM + tx;  // col
    int stride = blockIdx.z * TILE_DIM;

    scalar_t sum1, sum2;
    for (int term = 0; term < T; term++) {
        sum1 = 0, sum2 = 0;
        if (lowRankIdx < R && stride + ty < O) {
            subTileB1[ty * TILE_DIM + tx] = U1s[term][lowRankIdx][stride + ty];
            subTileB2[ty * TILE_DIM + tx] = S2s[term][lowRankIdx][stride + ty];
        } else {
            subTileB1[ty * TILE_DIM + tx] = 0;
            subTileB2[ty * TILE_DIM + tx] = 0;
        }
        if (term == 0) {
            if (batchIdx < B && stride + tx < O) {
                subTileA[ty * TILE_DIM + tx] = grad_output[batchIdx][stride + tx];
            } else {
                subTileA[ty * TILE_DIM + tx] = 0;
            }
        }
        __syncthreads();
        if (batchIdx < B && lowRankIdx < R) {
            for (int i = 0; i < TILE_DIM; i++) {
                auto u1_val = subTileB1[i * TILE_DIM + tx];
                auto s2_val = subTileB2[i * TILE_DIM + tx];
                auto grad_val = subTileA[ty * TILE_DIM + i];
                sum1 += u1_val * grad_val;
                sum2 += s2_val * grad_val;
            }
        }
        __syncthreads();

        if (batchIdx < B && lowRankIdx < R) {
            // Write the result to the output tensor.
            grad_intermediate[blockIdx.z][0][term][batchIdx][lowRankIdx] = sum1;
            grad_intermediate[blockIdx.z][1][term][batchIdx][lowRankIdx] = sum2;
        }
    }
}

/**
 * @brief CUDA kernel to compute intermediate gradients for the backward pass of a linear operation using shared memory tiling.
 *
 * This kernel multiplies the U2s tensor with the input tensor in a tiled fashion to compute the intermediate gradients
 * required for the backward pass of a linear layer. Shared memory is used to optimize memory access patterns.
 *
 * @tparam scalar_t The data type of the tensors (e.g., float, double).
 * @param input [B, I] - FlexibleTensorAccessor for the input tensor, where B is the batch size and I is the input dimension.
 * @param U2s [T, I, R] - PackedTensorAccessor for the U2s tensor, where T is the number of terms, I is the input dimension, and R is the rank.
 * @param interm_gradS2s [numTilesK, T, R, B] - PackedTensorAccessor for the output tensor to store intermediate gradients.
 * @param B The batch size.
 * @param I The input dimension.
 * @param R The rank dimension.
 * @param T The number of terms.
 *
 * @details
 * - The kernel uses shared memory to load tiles of U2s and input for efficient matrix multiplication.
 * - Each thread block processes a TILE_DIM x TILE_DIM tile.
 * - The computation is performed for each term in T, accumulating the result in 'sum'.
 * - Results are written to the interm_gradS2s tensor.
 * - Synchronization is used to ensure correct shared memory usage.
 */
template <typename scalar_t>
__global__ void sklinear_backward_grad_S2_interm(
    const FlexibleTensorAccessor<scalar_t, 2> input,                                      // [B,I]
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> U2s,       // [T,I,R]
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> interm_gradS2s,  // [numTilesK, T, R, B]
    int B, int I, int R, int T) {
    // multiply U2s x input
    extern __shared__ float shData[];
    scalar_t* sharedData = (scalar_t*)shData;
    scalar_t* subTileA = sharedData;
    scalar_t* subTileB = subTileA + TILE_DIM * TILE_DIM;

    int tx = threadIdx.x, ty = threadIdx.y;
    int lowRankIdx = blockIdx.y * TILE_DIM + ty;  // row
    int batchIdx = blockIdx.x * TILE_DIM + tx;    // col
    int stride = blockIdx.z * TILE_DIM;

    scalar_t sum;
    for (int term = 0; term < T; term++) {
        sum = 0;
        if (lowRankIdx < R && stride + tx < I) {
            subTileA[ty * TILE_DIM + tx] = U2s[term][stride + tx][lowRankIdx];
        } else {
            subTileA[ty * TILE_DIM + tx] = 0;
        }
        if (term == 0) {
            if (batchIdx < B && stride + ty < I) {
                subTileB[ty * TILE_DIM + tx] = input(batchIdx, stride + ty);
            } else {
                subTileB[ty * TILE_DIM + tx] = 0;
            }
        }
        __syncthreads();
        if (batchIdx < B && lowRankIdx < R) {
            for (int i = 0; i < TILE_DIM; i++) {
                auto u2_val = subTileB[i * TILE_DIM + tx];
                auto input_val = subTileA[ty * TILE_DIM + i];
                sum += u2_val * input_val;
            }
        }
        __syncthreads();

        if (batchIdx < B && lowRankIdx < R) {
            // Write the result to the output tensor.
            interm_gradS2s[blockIdx.z][term][lowRankIdx][batchIdx] = sum;
        }
    }
}

/**
 * @brief CUDA kernel to compute the backward gradient for S2 output in a low-rank linear layer.
 *
 * This kernel performs a batched matrix multiplication between the intermediate gradient tensor
 * (interm_gradS2s) and the output gradient tensor (grad_output) to produce the gradient with respect
 * to S2 (grad_S2s). The computation is tiled and uses shared memory for improved performance.
 *
 * @tparam scalar_t Data type of the tensors (e.g., float, double).
 * @param[in] interm_gradS2s Intermediate gradients with shape [T, R, B], where:
 *                           T - number of terms,
 *                           R - low-rank dimension,
 *                           B - batch size.
 * @param[in] grad_output Gradient of the output with shape [B, O], where:
 *                        O - output dimension.
 * @param[out] grad_S2s Output tensor for the gradients with respect to S2, shape [T, R, O].
 * @param[in] B Batch size.
 * @param[in] R Low-rank dimension.
 * @param[in] T Number of terms.
 * @param[in] O Output dimension.
 *
 * Each thread block computes a TILE_DIM x TILE_DIM tile of the output matrix for a given term.
 * Shared memory is used to load tiles of interm_gradS2s and grad_output for efficient access.
 * The kernel accumulates the result in the 'sum' variable and writes it to grad_S2s.
 */
template <typename scalar_t>
__global__ void sklinear_backward_grad_S2_output(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> interm_gradS2s,  // [T, R, B]
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_output,     // [B, O]
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_S2s,              // [T, R, O]
    int B, int R, int T, int O) {
    // interm_gradS2s x grad_output
    extern __shared__ float shData[];
    scalar_t* sharedData = (scalar_t*)shData;
    scalar_t* subTileA = sharedData;
    scalar_t* subTileB = subTileA + TILE_DIM * TILE_DIM;

    int tx = threadIdx.x, ty = threadIdx.y;
    int lowRankIdx = blockIdx.y * TILE_DIM + ty;  // row
    int outputIdx = blockIdx.x * TILE_DIM + tx;   // col
    int term = blockIdx.z;

    scalar_t sum = 0;
    for (int stride = 0; stride < B; stride += TILE_DIM) {
        if (lowRankIdx < R && stride + tx < B) {
            subTileA[ty * TILE_DIM + tx] = interm_gradS2s[term][lowRankIdx][stride + tx];
        } else {
            subTileA[ty * TILE_DIM + tx] = 0;
        }
        if (lowRankIdx < R && stride + ty < B) {
            subTileB[ty * TILE_DIM + tx] = grad_output[stride + ty][outputIdx];
        } else {
            subTileB[ty * TILE_DIM + tx] = 0;
        }
        __syncthreads();
        if (lowRankIdx < R && outputIdx < O) {
            for (int i = 0; i < TILE_DIM; i++) {
                auto interm_val = subTileA[ty * TILE_DIM + i];
                auto grad_val = subTileB[i * TILE_DIM + tx];
                sum += interm_val * grad_val;
            }
        }
        __syncthreads();
    }

    if (lowRankIdx < R && outputIdx < O) {
        // Write the result to the output tensor.
        grad_S2s[term][lowRankIdx][outputIdx] = sum;
    }
}

/**
 * @brief CUDA kernel to compute the gradient with respect to the input for a custom linear operation.
 *
 * This kernel computes grad_input for a batched linear operation with intermediate gradients and two sets of weights.
 * The computation performed is:
 *     grad_input[batchIdx][inputIdx] = sum_{term=0}^{T-1} sum_{stride=0}^{R-1} (
 *         grad_interm[0][term][batchIdx][stride] * S1s[term][inputIdx][stride] +
 *         grad_interm[1][term][batchIdx][stride] * U2s[term][inputIdx][stride]
 *     )
 *
 * Shared memory tiling is used for efficient memory access.
 *
 * @tparam scalar_t Data type of the tensors (e.g., float, double).
 * @param grad_interm [2, T, B, R] - Intermediate gradients for two terms, over time, batch, and rank dimensions.
 * @param S1s        [T, I, R]     - First set of weights, over time, input, and rank dimensions.
 * @param U2s        [T, I, R]     - Second set of weights, over time, input, and rank dimensions.
 * @param grad_input [B, I]        - Output tensor for the computed input gradients (to be written by the kernel).
 * @param B          Batch size.
 * @param I          Input size.
 * @param R          Rank dimension.
 * @param T          Time or sequence length.
 *
 * Each thread block computes a TILE_DIM x TILE_DIM tile of the output grad_input matrix.
 * Shared memory is used to load tiles of grad_interm and S1s/U2s for efficient computation.
 */
template <typename scalar_t>
__global__ void sklinear_backward_grad_input(
    const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grad_interm,  // [2,T,B,R]
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> S1s,          // [T,I,R]
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> U2s,          // [T,I,R]
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_input,         // [B,I]
    int B, int I, int R, int T) {
    // interm0 * S1s + interm1 * U2s
    extern __shared__ float shData[];
    scalar_t* sharedData = (scalar_t*)shData;
    scalar_t* subTileA1 = sharedData;
    scalar_t* subTileA2 = subTileA1 + TILE_DIM * TILE_DIM;
    scalar_t* subTileB1 = subTileA2 + TILE_DIM * TILE_DIM;
    scalar_t* subTileB2 = subTileB1 + TILE_DIM * TILE_DIM;

    int tx = threadIdx.x, ty = threadIdx.y;
    int batchIdx = blockIdx.y * TILE_DIM + ty;  // row
    int inputIdx = blockIdx.x * TILE_DIM + tx;  // col

    scalar_t sum = 0;

    for (int term = 0; term < T; term++) {
        for (int stride = 0; stride < R; stride += TILE_DIM) {
            if (batchIdx < B && stride + tx < R) {
                subTileA1[ty * TILE_DIM + tx] = grad_interm[0][term][batchIdx][stride + tx];
                subTileA2[ty * TILE_DIM + tx] = grad_interm[1][term][batchIdx][stride + tx];
            } else {
                subTileA1[ty * TILE_DIM + tx] = 0;
                subTileA2[ty * TILE_DIM + tx] = 0;
            }
            if (inputIdx < I && stride + ty < R) {
                subTileB1[ty * TILE_DIM + tx] = S1s[term][inputIdx][stride + ty];
                subTileB2[ty * TILE_DIM + tx] = U2s[term][inputIdx][stride + ty];
            } else {
                subTileB1[ty * TILE_DIM + tx] = 0;
                subTileB2[ty * TILE_DIM + tx] = 0;
            }
            __syncthreads();
            if (batchIdx < B && inputIdx < I) {
                for (int i = 0; i < TILE_DIM; i++) {
                    auto interm1_val = subTileA1[ty * TILE_DIM + i];
                    auto interm2_val = subTileA2[ty * TILE_DIM + i];
                    auto s1_val = subTileB1[i * TILE_DIM + tx];
                    auto u2_val = subTileB2[i * TILE_DIM + tx];
                    sum += interm1_val * s1_val + interm2_val * u2_val;
                }
            }
            __syncthreads();
        }
    }

    if (batchIdx < B && inputIdx < I) {
        grad_input[batchIdx][inputIdx] = sum;
    }
}

/**
 * @brief CUDA kernel to compute the backward gradient for S1 in a linear layer with shared memory tiling.
 *
 * This kernel computes the gradient of S1 by performing a tiled matrix multiplication between the input tensor
 * and an intermediate tensor (interm0), storing the result in grad_S1s. Shared memory is used to optimize memory access.
 *
 * @tparam scalar_t The data type of the tensors (e.g., float, double).
 * @param input [B, I] - 2D tensor accessor for the input matrix.
 * @param interm0 [T, B, R] - 3D tensor accessor for the intermediate tensor.
 * @param grad_S1s [T, I, R] - 3D tensor accessor for the output gradient tensor to be written.
 * @param B Batch size (number of rows in input and interm0).
 * @param I Input feature dimension (number of columns in input, number of rows in grad_S1s).
 * @param R Low-rank dimension (number of columns in interm0 and grad_S1s).
 * @param T Number of terms (outer dimension for interm0 and grad_S1s).
 *
 * Each thread block computes a TILE_DIM x TILE_DIM submatrix of the output for a given term (T).
 * Shared memory is used to load tiles of the input and interm0 matrices for efficient computation.
 * The kernel accumulates the sum of products for each output element and writes the result to grad_S1s.
 */
template <typename scalar_t>
__global__ void sklinear_backward_grad_S1(
    const FlexibleTensorAccessor<scalar_t, 2> input,                                     // [B,I]
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> interm0,  // [T,B,R]
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_S1s,       // [T,I,R]
    int B, int I, int R, int T) {
    // input * interm0
    extern __shared__ float shData[];
    scalar_t* subTileA = (scalar_t*)shData;
    scalar_t* subTileB = subTileA + TILE_DIM * TILE_DIM;

    int tx = threadIdx.x, ty = threadIdx.y;
    int inputIdx = blockIdx.y * TILE_DIM + ty;    // row
    int lowRankIdx = blockIdx.x * TILE_DIM + tx;  // col
    int term = blockIdx.z;

    scalar_t sum = 0;
    for (int stride = 0; stride < B; stride += TILE_DIM) {
        if (inputIdx < I && stride + tx < B) {
            subTileA[ty * TILE_DIM + tx] = input(stride + tx, inputIdx);
        } else {
            subTileA[ty * TILE_DIM + tx] = 0;
        }
        if (lowRankIdx < R && stride + ty < B) {
            subTileB[ty * TILE_DIM + tx] = interm0[term][stride + ty][lowRankIdx];
        } else {
            subTileB[ty * TILE_DIM + tx] = 0;
        }
        __syncthreads();
        if (inputIdx < I && lowRankIdx < R) {
            for (int i = 0; i < TILE_DIM; i++) {
                auto input_val = subTileA[ty * TILE_DIM + i];
                auto interm_val = subTileB[i * TILE_DIM + tx];
                sum += input_val * interm_val;
            }
        }
        __syncthreads();
    }

    if (inputIdx < I && lowRankIdx < R) {
        grad_S1s[term][inputIdx][lowRankIdx] = sum;
    }
}

/**
 * @brief Computes the backward pass for a sketched linear CUDA layer.
 *
 * This function calculates the gradients with respect to the input, S1s, S2s, and optionally the bias
 * for a custom sketched linear layer using CUDA kernels and multiple CUDA streams for parallelism.
 * It assumes all input tensors are contiguous and on the same CUDA device.
 *
 * The computation involves several intermediate CUDA kernel launches to compute:
 *   - Intermediate gradients for input and S1s/S2s using U1s, U2s, and the sketched matrices.
 *   - Gradients for S2s and S1s via reduction and further kernel launches.
 *   - Optionally, the gradient with respect to the bias if present.
 *
 * Synchronization between streams and events is used to ensure correct ordering of operations.
 *
 * @param grad_output The gradient of the loss with respect to the output of the layer. Shape: [B, O]
 * @param input       The input tensor to the layer. Shape: [B, I]
 * @param S1s         The first sketch matrix. Shape: [T, I, R]
 * @param S2s         The second sketch matrix. Shape: [T, R, O]
 * @param U1s         The first orthogonal matrix. Shape: [T, O, R]
 * @param U2s         The second orthogonal matrix. Shape: [T, R, I]
 * @param has_bias    Boolean flag indicating if the layer has a bias term.
 * @return std::vector<torch::Tensor> A vector containing:
 *   - grad_input: Gradient with respect to the input. Shape: [B, I]
 *   - grad_S1s:   Gradient with respect to S1s. Shape: [T, I, R]
 *   - grad_S2s:   Gradient with respect to S2s. Shape: [T, R, O]
 *   - grad_o:     (Optional) Gradient with respect to the bias. Shape: [O]
 *
 * @note All returned tensors are allocated on the same device as the input.
 * @note This function assumes the existence of CUDA kernels:
 *   - sklinear_backward_intermediate
 *   - sklinear_backward_grad_S2_interm
 *   - sklinear_backward_grad_S2_output
 *   - sklinear_backward_grad_input
 *   - sklinear_backward_grad_S1
 * and utility functions for tensor access.
 */
std::vector<torch::Tensor> sketched_linear_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    const bool has_bias) {
    TORCH_CHECK(S1s.is_contiguous(), "S1s tensor must be contiguous in memory.");
    TORCH_CHECK(S2s.is_contiguous(), "S2s tensor must be contiguous in memory.");
    TORCH_CHECK(U1s.is_contiguous(), "U1s tensor must be contiguous in memory.");
    TORCH_CHECK(U2s.is_contiguous(), "U2s tensor must be contiguous in memory.");
    // g = grad_output.div(2 * num_terms)
    // t1 = g * U1s.T -> interm[0]
    // grad_input ->  interm[0]  * S1s.T +  interm[1]  * U2s.T
    // grad_input = ([g * U1s.T] * S1s.T + [g * S2s.T] * U2s.T).sum(0)
    // grad_S2s = U2s.T * input.T * g
    // grad_S1s = input.t * interm[0]
    auto device_id = input.get_device();

    int64_t I = input.size(1), O = grad_output.size(1);
    int64_t T = S1s.size(0), R = S1s.size(2), B = grad_output.size(0);
    int64_t numTilesO = (O + TILE_DIM - 1) / TILE_DIM;
    torch::cuda::synchronize(device_id);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid1((R + TILE_DIM - 1) / TILE_DIM,
               (B + TILE_DIM - 1) / TILE_DIM,
               numTilesO);
    at::cuda::CUDAStream torch_stream1 = at::cuda::getStreamFromPool(false, device_id);
    cudaStream_t stream1 = torch_stream1.stream();
    at::cuda::setCurrentCUDAStream(torch_stream1);
    auto g = grad_output.div(2.0f * T).contiguous();
    cudaEvent_t afterGcompute;
    cudaEventCreate(&afterGcompute);
    cudaEventRecord(afterGcompute, stream1);

    auto grad_intermediate = torch::zeros({numTilesO, 2, T, B, R}, S1s.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "sklinear_backward_intermediate",
        ([&] {
            sklinear_backward_intermediate<scalar_t><<<grid1, block, 3 * TILE_DIM * TILE_DIM * sizeof(scalar_t), stream1>>>(
                g.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                U1s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                S2s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                grad_intermediate.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                B, O, T, R);
        }));

    int64_t numTilesI = (I + TILE_DIM - 1) / TILE_DIM;
    at::cuda::CUDAStream torch_stream2 = at::cuda::getStreamFromPool(false, device_id);
    cudaStream_t stream2 = torch_stream2.stream();
    at::cuda::setCurrentCUDAStream(torch_stream2);
    auto interm_gradS2s = torch::zeros({numTilesI, T, R, B}, S1s.options());

    dim3 grid2((B + TILE_DIM - 1) / TILE_DIM,
               (R + TILE_DIM - 1) / TILE_DIM,
               numTilesI);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "sklinear_backward_grad_S2_interm",
        ([&] {
            sklinear_backward_grad_S2_interm<scalar_t><<<grid2, block, 2 * TILE_DIM * TILE_DIM * sizeof(scalar_t), stream2>>>(
                tensor_utils::buildAccessor<scalar_t, 2>(input),
                U2s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                interm_gradS2s.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                B, I, R, T);
        }));
    at::cuda::setCurrentCUDAStream(torch_stream2);
    auto i_gradS2s = interm_gradS2s.sum(0);
    auto grad_S2s = torch::zeros({T, R, O}, S1s.options());
    cudaStreamWaitEvent(stream2, afterGcompute);
    dim3 grid4((O + TILE_DIM - 1) / TILE_DIM,
               (R + TILE_DIM - 1) / TILE_DIM,
               T);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "sklinear_backward_grad_S2_output",
        ([&] {
            sklinear_backward_grad_S2_output<scalar_t><<<grid4, block, 2 * TILE_DIM * TILE_DIM * sizeof(scalar_t), stream2>>>(
                i_gradS2s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                g.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                grad_S2s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                B, R, T, O);
        }));

    at::cuda::setCurrentCUDAStream(torch_stream1);
    auto interm = grad_intermediate.sum(0).contiguous();

    cudaEvent_t afterIntermSum;
    cudaEventCreate(&afterIntermSum);
    cudaEventRecord(afterIntermSum, stream1);
    cudaStreamWaitEvent(stream1, afterIntermSum);

    auto grad_input = torch::zeros({B, I}, S1s.options());
    dim3 grid3((I + TILE_DIM - 1) / TILE_DIM,
               (B + TILE_DIM - 1) / TILE_DIM);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "sklinear_backward_grad_input",
        ([&] {
            sklinear_backward_grad_input<scalar_t><<<grid3, block, 4 * TILE_DIM * TILE_DIM * sizeof(scalar_t), stream1>>>(
                interm.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
                S1s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                U2s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                grad_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                B, I, R, T);
        }));

    at::cuda::CUDAStream torch_stream3 = at::cuda::getStreamFromPool(false, device_id);
    cudaStream_t stream3 = torch_stream3.stream();
    at::cuda::setCurrentCUDAStream(torch_stream3);
    auto grad_S1s = torch::zeros({T, I, R}, S1s.options());

    cudaStreamWaitEvent(stream3, afterIntermSum);

    auto interm0 = interm[0].contiguous();

    dim3 grid5((R + TILE_DIM - 1) / TILE_DIM,
               (I + TILE_DIM - 1) / TILE_DIM,
               T);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(),
        "sklinear_backward_grad_S1",
        ([&] {
            sklinear_backward_grad_S1<scalar_t><<<grid5, block, 2 * TILE_DIM * TILE_DIM * sizeof(scalar_t), stream3>>>(
                tensor_utils::buildAccessor<scalar_t, 2>(input),
                interm0.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                grad_S1s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                B, I, R, T);
        }));

    torch::Tensor grad_o;
    if (has_bias) {
        at::cuda::CUDAStream torch_stream4 = at::cuda::getStreamFromPool(false, device_id);
        cudaStream_t stream4 = torch_stream4.stream();
        at::cuda::setCurrentCUDAStream(torch_stream4);
        grad_o = grad_output.sum(0);
        at::cuda::stream_synchronize(stream4);
    }

    at::cuda::stream_synchronize(stream1);
    at::cuda::stream_synchronize(stream2);
    at::cuda::stream_synchronize(stream3);
    torch::cuda::synchronize(device_id);
    at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream(device_id));
    cudaEventDestroy(afterIntermSum);
    cudaEventDestroy(afterGcompute);

    if (has_bias) {
        return {grad_input, grad_S1s, grad_S2s, grad_o};
    }
    return {grad_input, grad_S1s, grad_S2s};
}