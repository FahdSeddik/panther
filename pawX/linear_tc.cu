#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

#include "cuda_tensor_accessor.cuh"
#include "debug.cuh"
#include "linear.h"

using half_t = __half;  // WMMA inputs must be half precision
using float_t = float;  // Accumulate in FP32
namespace wmma = nvcuda::wmma;

// Tile sizes for WMMA fragments
static const int M = 16;
static const int N = 16;
static const int K = 16;

// Block tiling: 4×4 warps → 16 warps → 512 threads
static const int BLOCK_ROW_WARPS = 4;
static const int BLOCK_COL_WARPS = 4;
static const int WARPS_PER_BLOCK = BLOCK_ROW_WARPS * BLOCK_COL_WARPS;
static const int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

// Each block covers a 64×64 chunk of output
static const int TILE_WIDTH_M = M * BLOCK_ROW_WARPS;
static const int TILE_WIDTH_N = N * BLOCK_COL_WARPS;
static const int TILE_WIDTH_K = TILE_WIDTH_M;

template <typename scalar_t>
__global__ void sklinear_forward_intermediate_wmma(
    const FlexibleTensorAccessor<scalar_t, 2> input,  // [B,I]
    const FlexibleTensorAccessor<scalar_t, 3> S1s,    // [T,I,R]
    const FlexibleTensorAccessor<scalar_t, 3> U2s,    // [T,I,R]
    FlexibleTensorAccessor<scalar_t, 5> partial,      // [numTilesK,2,T,B,R]
    int B, int I, int R, int T, const float_t DIVISOR) {
    // Dynamic shared memory buffer:
    __shared__ half subTileA[TILE_WIDTH_K][TILE_WIDTH_M];
    __shared__ half subTileB1[TILE_WIDTH_N][TILE_WIDTH_K];
    __shared__ half subTileB2[TILE_WIDTH_N][TILE_WIDTH_K];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int k = blockIdx.z * TILE_WIDTH_K;
    int aRow = blockIdx.x * TILE_WIDTH_M;  // aRow
    int bCol = blockIdx.y * TILE_WIDTH_N;  // bCol
    auto zero = cuda_type<scalar_t>::get_zero();

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, M, N, K, half_t, wmma::col_major> fA;
    wmma::fragment<wmma::matrix_b, M, N, K, half_t, wmma::col_major> fB1, fB2;
    wmma::fragment<wmma::accumulator, M, N, K, cuda_type_t<scalar_t>> acc1, acc2;

    for (int term = 0; term < T; term++) {
        wmma::fill_fragment(acc1, zero);
        wmma::fill_fragment(acc2, zero);
#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
            int idx = tid + i;
            int aX = idx % TILE_WIDTH_M;
            int aY = idx / TILE_WIDTH_M;
            int bX = idx % TILE_WIDTH_K;
            int bY = idx / TILE_WIDTH_K;
            if (term == 0) subTileA[aY][aX] = (((aRow + aX) < B) && ((k + aY) < I)) ? __float2half(input.template get<float>(aRow + aX, k + aY) * DIVISOR) : __float2half(0.0f);
            subTileB1[bY][bX] = (((bCol + bY) < R) && ((k + bX) < I)) ? S1s.template get<half_t>(term, k + bX, bCol + bY) : __float2half(0.0f);
            subTileB2[bY][bX] = (((bCol + bY) < R) && ((k + bX) < I)) ? U2s.template get<half_t>(term, k + bX, bCol + bY) : __float2half(0.0f);
        }
        __syncthreads();
#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_K; i += K) {
            int subtileARow = M * (tx / warpSize);
            int subtileACol = i;
            int subtileBRow = i;
            int subtileBCol = N * ty;

            wmma::load_matrix_sync(fA, (half_t*)subTileA + subtileARow + subtileACol * TILE_WIDTH_M, TILE_WIDTH_M);
            wmma::load_matrix_sync(fB1, (half_t*)subTileB1 + subtileBRow + subtileBCol * TILE_WIDTH_K, TILE_WIDTH_K);
            wmma::load_matrix_sync(fB2, (half_t*)subTileB2 + subtileBRow + subtileBCol * TILE_WIDTH_K, TILE_WIDTH_K);
            wmma::mma_sync(acc1, fA, fB1, acc1);
            wmma::mma_sync(acc2, fA, fB2, acc2);
        }

        int warpM = (blockIdx.x * blockDim.x + tx) / warpSize;
        int warpN = blockIdx.y * blockDim.y + ty;
        int cRow = warpM * M;
        int cCol = warpN * N;

        if (cRow < B && cCol < R) {
            wmma::store_matrix_sync(&partial(blockIdx.z, 0, term, cRow, cCol), acc1, R, wmma::mem_row_major);
            wmma::store_matrix_sync(&partial(blockIdx.z, 1, term, cRow, cCol), acc2, R, wmma::mem_row_major);
        }
    }
}

template <typename scalar_t>
__global__ void sklinear_forward_output_wmma(
    const FlexibleTensorAccessor<scalar_t, 4> inter,  // [2,T,B,R]
    const FlexibleTensorAccessor<scalar_t, 3> U1s,    // [T,R,O]
    const FlexibleTensorAccessor<scalar_t, 3> S2s,    // [T,R,O]
    const FlexibleTensorAccessor<scalar_t, 1> bias,   // [O]
    bool hasBias,
    FlexibleTensorAccessor<scalar_t, 2> out,  // [B,O]
    int B, int R, int T, int O) {
    // __shared__ half subTileA1[TILE_WIDTH_K][TILE_WIDTH_M];
    // __shared__ half subTileA2[TILE_WIDTH_K][TILE_WIDTH_M];
    // __shared__ half subTileB1[TILE_WIDTH_N][TILE_WIDTH_K];
    // __shared__ half subTileB2[TILE_WIDTH_N][TILE_WIDTH_K];
    // __shared__ float_t shBias[TILE_WIDTH_N];
    extern __shared__ float_t shData[];
    half_t* subTileA1 = (half_t*)shData;
    half_t* subTileA2 = (half_t*)&subTileA1[TILE_WIDTH_K * TILE_WIDTH_M];
    half_t* subTileB1 = (half_t*)&subTileA2[TILE_WIDTH_K * TILE_WIDTH_M];
    half_t* subTileB2 = (half_t*)&subTileB1[TILE_WIDTH_N * TILE_WIDTH_K];
    cuda_type_t<scalar_t>* shBias = nullptr;
    if (hasBias) {
        shBias = (cuda_type_t<scalar_t>*)&subTileB2[TILE_WIDTH_N * TILE_WIDTH_K];
    }
    auto zero = cuda_type<scalar_t>::get_zero();

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + threadIdx.x;

    int base_b = blockIdx.x * TILE_WIDTH_M;  // aRow
    int base_o = blockIdx.y * TILE_WIDTH_N;  // bCol

    wmma::fragment<wmma::matrix_a, M, N, K, half_t, wmma::col_major> fA1, fA2;
    wmma::fragment<wmma::matrix_b, M, N, K, half_t, wmma::col_major> fB1, fB2;
    wmma::fragment<wmma::accumulator, M, N, K, cuda_type_t<scalar_t>> acc;
    wmma::fragment<wmma::accumulator, M, N, K, cuda_type_t<scalar_t>> c_frag;
    wmma::fill_fragment(acc, zero);

    if (hasBias) {
        // load bias into memory N
        for (int i = tx + ty * blockDim.x; i < N; i += blockDim.x * blockDim.y) {
            shBias[i] = (base_o + i) < O ? bias(base_o + i) : zero;
        }
    }

    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < R; k += TILE_WIDTH_K) {
#pragma unroll 1
            for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
                int idx = tid + i;
                int aX = idx % TILE_WIDTH_M;
                int aY = idx / TILE_WIDTH_M;
                int bX = idx % TILE_WIDTH_K;
                int bY = idx / TILE_WIDTH_K;
                subTileA1[aY * TILE_WIDTH_M + aX] = (((base_b + aX) < B) && ((k + aY) < R)) ? inter.template get<half_t>(0, t, base_b + aX, k + aY) : __float2half(0.0f);
                subTileA2[aY * TILE_WIDTH_M + aX] = (((base_b + aX) < B) && ((k + aY) < R)) ? inter.template get<half_t>(1, t, base_b + aX, k + aY) : __float2half(0.0f);
                subTileB1[bY * TILE_WIDTH_K + bX] = (((base_o + bY) < O) && ((k + bX) < R)) ? U1s.template get<half_t>(t, k + bX, base_o + bY) : __float2half(0.0f);
                subTileB2[bY * TILE_WIDTH_K + bX] = (((base_o + bY) < O) && ((k + bX) < R)) ? S2s.template get<half_t>(t, k + bX, base_o + bY) : __float2half(0.0f);
            }
            __syncthreads();
#pragma unroll 1
            for (int i = 0; i < TILE_WIDTH_K; i += K) {
                int subtileARow = M * (tx / warpSize);
                int subtileACol = i;
                int subtileBRow = i;
                int subtileBCol = N * ty;

                wmma::load_matrix_sync(fA1, subTileA1 + subtileARow + subtileACol * TILE_WIDTH_M, TILE_WIDTH_M);
                wmma::load_matrix_sync(fA2, subTileA2 + subtileARow + subtileACol * TILE_WIDTH_M, TILE_WIDTH_M);
                wmma::load_matrix_sync(fB1, subTileB1 + subtileBRow + subtileBCol * TILE_WIDTH_K, TILE_WIDTH_K);
                wmma::load_matrix_sync(fB2, subTileB2 + subtileBRow + subtileBCol * TILE_WIDTH_K, TILE_WIDTH_K);
                wmma::mma_sync(acc, fA1, fB1, acc);
                wmma::mma_sync(acc, fA2, fB2, acc);
            }
        }
    }

    int warpM = (blockIdx.x * blockDim.x + tx) / warpSize;
    int warpN = blockIdx.y * blockDim.y + ty;
    int cRow = warpM * M;
    int cCol = warpN * N;

    if (cRow < B && cCol < O) {
        if (hasBias) {
            wmma::load_matrix_sync(c_frag, shBias, 0, wmma::mem_row_major);
            for (int i = 0; i < c_frag.num_elements; i++) {
                acc.x[i] = cuda_type<scalar_t>::add(acc.x[i], c_frag.x[i]);
            }
        }
        wmma::store_matrix_sync(&out(cRow, cCol), acc, O, wmma::mem_row_major);
    }
}

// Host launcher: computes and passes shared_bytes for both kernels
torch::Tensor sketched_linear_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    c10::optional<torch::Tensor> bias) {
    TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf, "Input tensor must be float or half precision.");
    TORCH_CHECK(S1s.is_contiguous(), "S1s tensor must be contiguous.");
    TORCH_CHECK(U2s.is_contiguous(), "U2s tensor must be contiguous.");
    TORCH_CHECK(S2s.is_contiguous(), "S2s tensor must be contiguous.");
    TORCH_CHECK(U1s.is_contiguous(), "U1s tensor must be contiguous.");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_contiguous(), "Bias tensor must be contiguous.");
    }

    int B = input.size(0), I = input.size(1);
    int T = S1s.size(0), R = S1s.size(2), O = S2s.size(2);
    dim3 block(BLOCK_ROW_WARPS * 32, BLOCK_COL_WARPS);

    torch::Tensor interm;
    if (T > 1) {
        int64_t numTilesI = (I + TILE_WIDTH_K - 1) / TILE_WIDTH_K;
        auto partial = torch::zeros({numTilesI, 2, T, B, R}, S1s.options());
        dim3 grid1((B + TILE_WIDTH_M - 1) / TILE_WIDTH_M,
                   (R + TILE_WIDTH_N - 1) / TILE_WIDTH_N,
                   numTilesI);

        AT_DISPATCH_FLOAT_AND_HALF(
            input.scalar_type(),
            "sklinear_forward_intermediate_wmma",
            [&] {
                sklinear_forward_intermediate_wmma<scalar_t><<<grid1, block>>>(
                    tensor_utils::buildAccessor<scalar_t, 2>(input),
                    tensor_utils::buildAccessor<scalar_t, 3>(S1s),
                    tensor_utils::buildAccessor<scalar_t, 3>(U2s),
                    tensor_utils::buildAccessor<scalar_t, 5>(partial),
                    B, I, R, T, (1.0f / (2.0f * T)));
            });

        interm = partial.sum(0);
    } else {
        auto input_expanded = input.unsqueeze(0);
        interm = torch::stack({input_expanded.bmm(S1s), input_expanded.bmm(U2s)}, 0);
    }

    if (R <= 96) {
        auto out = torch::zeros({B, O}, S1s.options());
        dim3 grid2((B + TILE_WIDTH_M - 1) / TILE_WIDTH_M,
                   (O + TILE_WIDTH_N - 1) / TILE_WIDTH_N);
        AT_DISPATCH_FLOAT_AND_HALF(
            interm.scalar_type(),
            "sklinear_forward_output_wmma",
            [&] {
                int shared_bytes = (TILE_WIDTH_K * TILE_WIDTH_M * 2 + 2 * TILE_WIDTH_N * TILE_WIDTH_K) * sizeof(half_t) + (bias.has_value() ? TILE_WIDTH_N : 0) * sizeof(scalar_t);
                sklinear_forward_output_wmma<scalar_t><<<grid2, block, shared_bytes>>>(
                    tensor_utils::buildAccessor<scalar_t, 4>(interm),
                    tensor_utils::buildAccessor<scalar_t, 3>(U1s),
                    tensor_utils::buildAccessor<scalar_t, 3>(S2s),
                    bias.has_value() ? tensor_utils::buildAccessor<scalar_t, 1>(bias.value()) : FlexibleTensorAccessor<scalar_t, 1>(),
                    bias.has_value(),
                    tensor_utils::buildAccessor<scalar_t, 2>(out),
                    B, R, T, O);
            });
        return out;
    } else {
        auto result = (interm[0].bmm(U1s) + interm[1].bmm(S2s)).mean(0).div(2.0f);
        if (bias.has_value()) {
            result.add_(bias.value());
        }
        return result;
    }
}

template <typename scalar_t>
__global__ void sklinear_backward_intermediate_wmma(
    const FlexibleTensorAccessor<scalar_t, 2> grad_output,  // [B,O]
    const FlexibleTensorAccessor<scalar_t, 3> U1s,          // [T,R,O]
    const FlexibleTensorAccessor<scalar_t, 3> S2s,          // [T,R,O]
    FlexibleTensorAccessor<scalar_t, 5> grad_intermediate,  // [numTilesK,2,T,B,R]
    int B, int O, int T, int R) {
    __shared__ half subTileA[TILE_WIDTH_K][TILE_WIDTH_M];
    __shared__ half subTileB1[TILE_WIDTH_N][TILE_WIDTH_K];
    __shared__ half subTileB2[TILE_WIDTH_N][TILE_WIDTH_K];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int aRow = blockIdx.x * TILE_WIDTH_M;  // aRow
    int bCol = blockIdx.y * TILE_WIDTH_N;  // bCol
    auto zero = cuda_type<scalar_t>::get_zero();

    int k = blockIdx.z * TILE_WIDTH_K;
    wmma::fragment<wmma::matrix_a, M, N, K, half_t, wmma::col_major> fA;
    wmma::fragment<wmma::matrix_b, M, N, K, half_t, wmma::col_major> fB1, fB2;
    wmma::fragment<wmma::accumulator, M, N, K, cuda_type_t<scalar_t>> acc1, acc2;
    for (int term = 0; term < T; term++) {
        wmma::fill_fragment(acc1, zero);
        wmma::fill_fragment(acc2, zero);
#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
            int idx = tid + i;
            int aX = idx % TILE_WIDTH_M;
            int aY = idx / TILE_WIDTH_M;
            int bX = idx % TILE_WIDTH_K;
            int bY = idx / TILE_WIDTH_K;
            if (term == 0) subTileA[aY][aX] = (((aRow + aX) < B) && ((k + aY) < O)) ? grad_output.template get<half_t>(aRow + aX, k + aY) : __float2half(0.0f);
            subTileB1[bY][bX] = (((bCol + bY) < R) && ((k + bX) < O)) ? U1s.template get<half_t>(term, bCol + bY, k + bX) : __float2half(0.0f);
            subTileB2[bY][bX] = (((bCol + bY) < R) && ((k + bX) < O)) ? S2s.template get<half_t>(term, bCol + bY, k + bX) : __float2half(0.0f);
        }
        __syncthreads();

#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_K; i += K) {
            int subtileARow = M * (tx / warpSize);
            int subtileACol = i;
            int subtileBRow = i;
            int subtileBCol = N * ty;

            wmma::load_matrix_sync(fA, (half_t*)subTileA + subtileARow + subtileACol * TILE_WIDTH_M, TILE_WIDTH_M);
            wmma::load_matrix_sync(fB1, (half_t*)subTileB1 + subtileBRow + subtileBCol * TILE_WIDTH_K, TILE_WIDTH_K);
            wmma::load_matrix_sync(fB2, (half_t*)subTileB2 + subtileBRow + subtileBCol * TILE_WIDTH_K, TILE_WIDTH_K);
            wmma::mma_sync(acc1, fA, fB1, acc1);
            wmma::mma_sync(acc2, fA, fB2, acc2);
        }

        int warpM = (blockIdx.x * blockDim.x + tx) / warpSize;
        int warpN = blockIdx.y * blockDim.y + ty;
        int cRow = warpM * M;
        int cCol = warpN * N;

        if (cRow < B && cCol < R) {
            wmma::store_matrix_sync(&grad_intermediate(blockIdx.z, 0, term, cRow, cCol), acc1, R, wmma::mem_row_major);
            wmma::store_matrix_sync(&grad_intermediate(blockIdx.z, 1, term, cRow, cCol), acc2, R, wmma::mem_row_major);
        }
    }
}

template <typename scalar_t>
__global__ void sklinear_backward_grad_S2_interm_wmma(
    const FlexibleTensorAccessor<scalar_t, 2> input,     // [B,I]
    const FlexibleTensorAccessor<scalar_t, 3> U2s,       // [T,I,R]
    FlexibleTensorAccessor<scalar_t, 4> interm_gradS2s,  // [numTilesK, T, R, B]
    int B, int I, int R, int T) {
    __shared__ half subTileA[TILE_WIDTH_K][TILE_WIDTH_M];  // U2s
    __shared__ half subTileB[TILE_WIDTH_N][TILE_WIDTH_K];  // input

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int aRow = blockIdx.x * TILE_WIDTH_M;  // aRow -> R
    int bCol = blockIdx.y * TILE_WIDTH_N;  // bCol -> B
    auto zero = cuda_type<scalar_t>::get_zero();

    int k = blockIdx.z * TILE_WIDTH_K;  // -> I
    wmma::fragment<wmma::matrix_a, M, N, K, half_t, wmma::col_major> fA;
    wmma::fragment<wmma::matrix_b, M, N, K, half_t, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, M, N, K, cuda_type_t<scalar_t>> acc;
    for (int term = 0; term < T; term++) {
        wmma::fill_fragment(acc, zero);
#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
            int idx = tid + i;
            int aX = idx % TILE_WIDTH_M;
            int aY = idx / TILE_WIDTH_M;
            int bX = idx % TILE_WIDTH_K;
            int bY = idx / TILE_WIDTH_K;
            subTileA[aY][aX] = (((aRow + aX) < R) && ((k + aY) < I)) ? U2s.template get<half_t>(term, k + aY, aRow + aX) : __float2half(0.0f);
            if (term == 0) subTileB[bY][bX] = (((bCol + bY) < B) && ((k + bX) < I)) ? input.template get<half_t>(bCol + bY, k + bX) : __float2half(0.0f);
        }
        __syncthreads();
#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_K; i += K) {
            int subtileARow = M * (tx / warpSize);
            int subtileACol = i;
            int subtileBRow = i;
            int subtileBCol = N * ty;

            wmma::load_matrix_sync(fA, (half_t*)subTileA + subtileARow + subtileACol * TILE_WIDTH_M, TILE_WIDTH_M);
            wmma::load_matrix_sync(fB, (half_t*)subTileB + subtileBRow + subtileBCol * TILE_WIDTH_K, TILE_WIDTH_K);
            wmma::mma_sync(acc, fA, fB, acc);
        }

        int warpM = (blockIdx.x * blockDim.x + tx) / warpSize;
        int warpN = blockIdx.y * blockDim.y + ty;
        int cRow = warpM * M;
        int cCol = warpN * N;

        if (cRow < R && cCol < B) {
            wmma::store_matrix_sync(&interm_gradS2s(blockIdx.z, term, cRow, cCol), acc, B, wmma::mem_row_major);
        }
    }
}

template <typename scalar_t>
__global__ void sklinear_backward_grad_S2_output_wmma(
    const FlexibleTensorAccessor<scalar_t, 3> interm_gradS2s,  // [T, R, B]
    const FlexibleTensorAccessor<scalar_t, 2> grad_output,     // [B, O]
    FlexibleTensorAccessor<scalar_t, 3> grad_S2s,              // [T, R, O]
    int B, int R, int T, int O) {
    __shared__ half subTileA[TILE_WIDTH_K][TILE_WIDTH_M];  // interm_gradS2s
    __shared__ half subTileB[TILE_WIDTH_N][TILE_WIDTH_K];  // grad_output

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int aRow = blockIdx.x * TILE_WIDTH_M;  // aRow -> R
    int bCol = blockIdx.y * TILE_WIDTH_N;  // bCol -> O
    auto zero = cuda_type<scalar_t>::get_zero();

    int term = blockIdx.z;  // -> T
    wmma::fragment<wmma::matrix_a, M, N, K, half_t, wmma::col_major> fA;
    wmma::fragment<wmma::matrix_b, M, N, K, half_t, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, M, N, K, cuda_type_t<scalar_t>> acc;
    wmma::fill_fragment(acc, zero);
    for (int k = 0; k < B; k += TILE_WIDTH_K) {
#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
            int idx = tid + i;
            int aX = idx % TILE_WIDTH_M;
            int aY = idx / TILE_WIDTH_M;
            int bX = idx % TILE_WIDTH_K;
            int bY = idx / TILE_WIDTH_K;
            subTileA[aY][aX] = (((aRow + aX) < R) && ((k + aY) < B)) ? interm_gradS2s.template get<half_t>(term, aRow + aX, k + aY) : __float2half(0.0f);
            subTileB[bY][bX] = (((bCol + bY) < O) && ((k + bX) < B)) ? grad_output.template get<half_t>(k + bX, bCol + bY) : __float2half(0.0f);
        }
        __syncthreads();
#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_K; i += K) {
            int subtileARow = M * (tx / warpSize);
            int subtileACol = i;
            int subtileBRow = i;
            int subtileBCol = N * ty;

            wmma::load_matrix_sync(fA, (half_t*)subTileA + subtileARow + subtileACol * TILE_WIDTH_M, TILE_WIDTH_M);
            wmma::load_matrix_sync(fB, (half_t*)subTileB + subtileBRow + subtileBCol * TILE_WIDTH_K, TILE_WIDTH_K);
            wmma::mma_sync(acc, fA, fB, acc);
        }
    }

    int warpM = (blockIdx.x * blockDim.x + tx) / warpSize;
    int warpN = blockIdx.y * blockDim.y + ty;
    int cRow = warpM * M;
    int cCol = warpN * N;

    if (cRow < R && cCol < O) {
        wmma::store_matrix_sync(&grad_S2s(term, cRow, cCol), acc, O, wmma::mem_row_major);
    }
}

template <typename scalar_t>
__global__ void sklinear_backward_grad_input_wmma(
    const FlexibleTensorAccessor<scalar_t, 4> grad_interm,  // [2,T,B,R]
    const FlexibleTensorAccessor<scalar_t, 3> S1s,          // [T,I,R]
    const FlexibleTensorAccessor<scalar_t, 3> U2s,          // [T,I,R]
    FlexibleTensorAccessor<scalar_t, 2> grad_input,         // [B,I]
    int B, int I, int R, int T) {
    __shared__ half subTileA1[TILE_WIDTH_K][TILE_WIDTH_M];  // interm0
    __shared__ half subTileA2[TILE_WIDTH_K][TILE_WIDTH_M];  // interm1
    __shared__ half subTileB1[TILE_WIDTH_N][TILE_WIDTH_K];  // S1s
    __shared__ half subTileB2[TILE_WIDTH_N][TILE_WIDTH_K];  // U2s

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int aRow = blockIdx.x * TILE_WIDTH_M;  // aRow -> B
    int bCol = blockIdx.y * TILE_WIDTH_N;  // bCol -> I
    auto zero = cuda_type<scalar_t>::get_zero();
    wmma::fragment<wmma::matrix_a, M, N, K, half_t, wmma::col_major> fA1, fA2;
    wmma::fragment<wmma::matrix_b, M, N, K, half_t, wmma::col_major> fB1, fB2;
    wmma::fragment<wmma::accumulator, M, N, K, cuda_type_t<scalar_t>> acc1, acc2;
    wmma::fill_fragment(acc1, zero);
    // wmma::fill_fragment(acc2, zero);

    for (int term = 0; term < T; term++) {
        for (int k = 0; k < R; k += TILE_WIDTH_K) {
#pragma unroll 1
            for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
                int idx = tid + i;
                int aX = idx % TILE_WIDTH_M;
                int aY = idx / TILE_WIDTH_M;
                int bX = idx % TILE_WIDTH_K;
                int bY = idx / TILE_WIDTH_K;
                subTileA1[aY][aX] = (((aRow + aX) < B) && ((k + aY) < R)) ? grad_interm.template get<half_t>(0, term, aRow + aX, k + aY) : __float2half(0.0f);
                subTileA2[aY][aX] = (((aRow + aX) < B) && ((k + aY) < R)) ? grad_interm.template get<half_t>(1, term, aRow + aX, k + aY) : __float2half(0.0f);
                subTileB1[bY][bX] = (((bCol + bY) < I) && ((k + bX) < R)) ? S1s.template get<half_t>(term, bCol + bY, k + bX) : __float2half(0.0f);
                subTileB2[bY][bX] = (((bCol + bY) < I) && ((k + bX) < R)) ? U2s.template get<half_t>(term, bCol + bY, k + bX) : __float2half(0.0f);
            }
            __syncthreads();
#pragma unroll 1
            for (int i = 0; i < TILE_WIDTH_K; i += K) {
                int subtileARow = M * (tx / warpSize);
                int subtileACol = i;
                int subtileBRow = i;
                int subtileBCol = N * ty;

                wmma::load_matrix_sync(fA1, (half_t*)subTileA1 + subtileARow + subtileACol * TILE_WIDTH_M, TILE_WIDTH_M);
                wmma::load_matrix_sync(fA2, (half_t*)subTileA2 + subtileARow + subtileACol * TILE_WIDTH_M, TILE_WIDTH_M);
                wmma::load_matrix_sync(fB1, (half_t*)subTileB1 + subtileBRow + subtileBCol * TILE_WIDTH_K, TILE_WIDTH_K);
                wmma::load_matrix_sync(fB2, (half_t*)subTileB2 + subtileBRow + subtileBCol * TILE_WIDTH_K, TILE_WIDTH_K);
                wmma::mma_sync(acc1, fA1, fB1, acc1);
                wmma::mma_sync(acc1, fA2, fB2, acc1);
            }
        }
    }

    int warpM = (blockIdx.x * blockDim.x + tx) / warpSize;
    int warpN = blockIdx.y * blockDim.y + ty;
    int cRow = warpM * M;
    int cCol = warpN * N;

    if (cRow < B && cCol < I) {
        // for (int i = 0; i < acc1.num_elements; i++) {
        //     acc1.x[i] = cuda_type<scalar_t>::add(acc1.x[i], acc2.x[i]);
        // }
        wmma::store_matrix_sync(&grad_input(cRow, cCol), acc1, I, wmma::mem_row_major);
    }
}

template <typename scalar_t>
__global__ void sklinear_backward_grad_S1_wmma(
    const FlexibleTensorAccessor<scalar_t, 2> input,    // [B,I]
    const FlexibleTensorAccessor<scalar_t, 3> interm0,  // [T,B,R]
    FlexibleTensorAccessor<scalar_t, 3> grad_S1s,       // [T,I,R]
    int B, int I, int R, int T) {
    __shared__ half subTileA[TILE_WIDTH_K][TILE_WIDTH_M];  // input
    __shared__ half subTileB[TILE_WIDTH_N][TILE_WIDTH_K];  // interm0

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int aRow = blockIdx.x * TILE_WIDTH_M;  // aRow -> I
    int bCol = blockIdx.y * TILE_WIDTH_N;  // bCol -> R

    int term = blockIdx.z;  // -> T
    auto zero = cuda_type<scalar_t>::get_zero();

    wmma::fragment<wmma::matrix_a, M, N, K, half_t, wmma::col_major> fA;
    wmma::fragment<wmma::matrix_b, M, N, K, half_t, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, M, N, K, cuda_type_t<scalar_t>> acc;
    wmma::fill_fragment(acc, zero);
    for (int k = 0; k < B; k += TILE_WIDTH_K) {
#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
            int idx = tid + i;
            int aX = idx % TILE_WIDTH_M;
            int aY = idx / TILE_WIDTH_M;
            int bX = idx % TILE_WIDTH_K;
            int bY = idx / TILE_WIDTH_K;
            subTileA[aY][aX] = (((aRow + aX) < I) && ((k + aY) < B)) ? input.template get<half_t>(k + aY, aRow + aX) : __float2half(0.0f);
            subTileB[bY][bX] = (((bCol + bY) < R) && ((k + bX) < B)) ? interm0.template get<half_t>(term, k + bX, bCol + bY) : __float2half(0.0f);
        }
        __syncthreads();
#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_K; i += K) {
            int subtileARow = M * (tx / warpSize);
            int subtileACol = i;
            int subtileBRow = i;
            int subtileBCol = N * ty;

            wmma::load_matrix_sync(fA, (half_t*)subTileA + subtileARow + subtileACol * TILE_WIDTH_M, TILE_WIDTH_M);
            wmma::load_matrix_sync(fB, (half_t*)subTileB + subtileBRow + subtileBCol * TILE_WIDTH_K, TILE_WIDTH_K);
            wmma::mma_sync(acc, fA, fB, acc);
        }
    }

    int warpM = (blockIdx.x * blockDim.x + tx) / warpSize;
    int warpN = blockIdx.y * blockDim.y + ty;
    int cRow = warpM * M;
    int cCol = warpN * N;

    if (cRow < I && cCol < R) {
        wmma::store_matrix_sync(&grad_S1s(term, cRow, cCol), acc, R, wmma::mem_row_major);
    }
}

std::vector<torch::Tensor> sketched_linear_backward_cuda(
    const torch::Tensor& grad_output,  // [B, O]
    const torch::Tensor& input,        // [B, I]
    const torch::Tensor& S1s,          // [T, I, R]
    const torch::Tensor& S2s,          // [T, R, O]
    const torch::Tensor& U1s,          // [T, R, O]
    const torch::Tensor& U2s,          // [T, I, R]
    const bool has_bias) {
    TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf, "Input tensor must be float or half precision.");
    TORCH_CHECK(grad_output.scalar_type() == at::kFloat || grad_output.scalar_type() == at::kHalf, "Gradient output tensor must be float or half precision.");
    TORCH_CHECK(S1s.is_contiguous(), "S1s tensor must be contiguous.");
    TORCH_CHECK(U2s.is_contiguous(), "U2s tensor must be contiguous.");
    TORCH_CHECK(S2s.is_contiguous(), "S2s tensor must be contiguous.");
    TORCH_CHECK(U1s.is_contiguous(), "U1s tensor must be contiguous.");
    // g = grad_output.div(2 * num_terms)
    // t1 = g * U1s.T -> interm[0]
    // grad_input ->  interm[0]  * S1s.T +  interm[1]  * U2s.T
    // grad_input = ([g * U1s.T] * S1s.T + [g * S2s.T] * U2s.T).sum(0)
    // grad_S2s = U2s.T * input.T * g
    // grad_S1s = input.t * interm[0]
    auto device_id = input.get_device();

    int64_t I = input.size(1), O = grad_output.size(1);
    int64_t T = S1s.size(0), R = S1s.size(2), B = grad_output.size(0);
    int64_t numTilesO = (O + TILE_WIDTH_K - 1) / TILE_WIDTH_K;

    dim3 block(BLOCK_ROW_WARPS * 32, BLOCK_COL_WARPS);
    dim3 grid1((B + TILE_WIDTH_M - 1) / TILE_WIDTH_M,
               (R + TILE_WIDTH_N - 1) / TILE_WIDTH_N,
               numTilesO);
    at::cuda::CUDAStream torch_stream1 = at::cuda::getStreamFromPool(false, device_id);
    cudaStream_t stream1 = torch_stream1.stream();
    at::cuda::setCurrentCUDAStream(torch_stream1);
    auto g = grad_output.div(2.0f * T).contiguous();
    cudaEvent_t afterGcompute;
    cudaEventCreate(&afterGcompute);
    cudaEventRecord(afterGcompute, stream1);

    auto grad_intermediate = torch::zeros({numTilesO, 2, T, B, R}, S1s.options());

    AT_DISPATCH_FLOAT_AND_HALF(
        input.scalar_type(),
        "sklinear_backward_intermediate_wmma",
        [&] {
            sklinear_backward_intermediate_wmma<scalar_t><<<grid1, block, 0, stream1>>>(
                tensor_utils::buildAccessor<scalar_t, 2>(g),
                tensor_utils::buildAccessor<scalar_t, 3>(U1s),
                tensor_utils::buildAccessor<scalar_t, 3>(S2s),
                tensor_utils::buildAccessor<scalar_t, 5>(grad_intermediate),
                B, O, T, R);
        });

    int64_t numTilesI = (I + TILE_WIDTH_K - 1) / TILE_WIDTH_K;
    at::cuda::CUDAStream torch_stream2 = at::cuda::getStreamFromPool(false, device_id);
    cudaStream_t stream2 = torch_stream2.stream();
    at::cuda::setCurrentCUDAStream(torch_stream2);
    auto interm_gradS2s = torch::zeros({numTilesI, T, R, B}, S1s.options());

    dim3 grid2((R + TILE_WIDTH_M - 1) / TILE_WIDTH_M,
               (B + TILE_WIDTH_N - 1) / TILE_WIDTH_N,
               numTilesI);
    AT_DISPATCH_FLOAT_AND_HALF(
        input.scalar_type(),
        "sklinear_backward_grad_S2_interm_wmma",
        [&] {
            sklinear_backward_grad_S2_interm_wmma<scalar_t><<<grid2, block, 0, stream2>>>(
                tensor_utils::buildAccessor<scalar_t, 2>(input),
                tensor_utils::buildAccessor<scalar_t, 3>(U2s),
                tensor_utils::buildAccessor<scalar_t, 4>(interm_gradS2s),
                B, I, R, T);
        });

    at::cuda::setCurrentCUDAStream(torch_stream2);
    auto i_gradS2s = interm_gradS2s.sum(0).contiguous();
    auto grad_S2s = torch::zeros({T, R, O}, S1s.options());
    cudaStreamWaitEvent(stream2, afterGcompute);
    dim3 grid4((R + TILE_WIDTH_M - 1) / TILE_WIDTH_M,
               (O + TILE_WIDTH_N - 1) / TILE_WIDTH_N,
               T);
    AT_DISPATCH_FLOAT_AND_HALF(
        input.scalar_type(),
        "sklinear_backward_grad_S2_output_wmma",
        [&] {
            sklinear_backward_grad_S2_output_wmma<scalar_t><<<grid4, block, 0, stream2>>>(
                tensor_utils::buildAccessor<scalar_t, 3>(i_gradS2s),
                tensor_utils::buildAccessor<scalar_t, 2>(g),
                tensor_utils::buildAccessor<scalar_t, 3>(grad_S2s),
                B, R, T, O);
        });

    at::cuda::setCurrentCUDAStream(torch_stream1);
    auto interm = grad_intermediate.sum(0);

    cudaEvent_t afterIntermSum;
    cudaEventCreate(&afterIntermSum);
    cudaEventRecord(afterIntermSum, stream1);

    // auto grad_input = torch::zeros({B, I}, S1s.options());
    // dim3 grid3((B + TILE_WIDTH_M - 1) / TILE_WIDTH_M,
    //            (I + TILE_WIDTH_N - 1) / TILE_WIDTH_N);
    // AT_DISPATCH_FLOAT_AND_HALF(
    //     input.scalar_type(),
    //     "sklinear_backward_grad_input_wmma",
    //     [&] {
    //         sklinear_backward_grad_input_wmma<scalar_t><<<grid3, block, 0, stream1>>>(
    //             tensor_utils::buildAccessor<scalar_t, 4>(interm),
    //             tensor_utils::buildAccessor<scalar_t, 3>(S1s),
    //             tensor_utils::buildAccessor<scalar_t, 3>(U2s),
    //             tensor_utils::buildAccessor<scalar_t, 2>(grad_input),
    //             B, I, R, T);
    //     });
    auto grad_input = (interm[0].bmm(S1s.transpose(1, 2)) + interm[1].bmm(U2s.transpose(1, 2))).sum(0);

    at::cuda::CUDAStream torch_stream3 = at::cuda::getStreamFromPool(false, device_id);
    cudaStream_t stream3 = torch_stream3.stream();
    at::cuda::setCurrentCUDAStream(torch_stream3);
    auto grad_S1s = torch::zeros({T, I, R}, S1s.options());

    cudaStreamWaitEvent(stream3, afterIntermSum);

    auto interm0 = interm[0].contiguous();

    dim3 grid5((I + TILE_WIDTH_M - 1) / TILE_WIDTH_M,
               (R + TILE_WIDTH_N - 1) / TILE_WIDTH_N,
               T);
    AT_DISPATCH_FLOAT_AND_HALF(
        input.scalar_type(),
        "sklinear_backward_grad_S1_wmma",
        [&] {
            sklinear_backward_grad_S1_wmma<scalar_t><<<grid5, block, 0, stream3>>>(
                tensor_utils::buildAccessor<scalar_t, 2>(input),
                tensor_utils::buildAccessor<scalar_t, 3>(interm0),
                tensor_utils::buildAccessor<scalar_t, 3>(grad_S1s),
                B, I, R, T);
        });

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
