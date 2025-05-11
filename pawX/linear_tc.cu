#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

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

__global__ void sklinear_forward_intermediate_wmma(
    const torch::PackedTensorAccessor32<float_t, 2, torch::RestrictPtrTraits> input,  // [B,I]
    const torch::PackedTensorAccessor32<float_t, 3, torch::RestrictPtrTraits> S1s,    // [T,I,R]
    const torch::PackedTensorAccessor32<float_t, 3, torch::RestrictPtrTraits> U2s,    // [T,I,R]
    torch::PackedTensorAccessor32<float_t, 5, torch::RestrictPtrTraits> partial,      // [numTilesK,2,T,B,R]
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

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, M, N, K, half_t, wmma::col_major> fA;
    wmma::fragment<wmma::matrix_b, M, N, K, half_t, wmma::col_major> fB1, fB2;
    wmma::fragment<wmma::accumulator, M, N, K, float_t> acc1, acc2;

    for (int term = 0; term < T; term++) {
        wmma::fill_fragment(acc1, 0.0f);
        wmma::fill_fragment(acc2, 0.0f);
#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
            int idx = tid + i;
            int aX = idx % TILE_WIDTH_M;
            int aY = idx / TILE_WIDTH_M;
            int bX = idx % TILE_WIDTH_K;
            int bY = idx / TILE_WIDTH_K;
            if (term == 0) subTileA[aY][aX] = (((aRow + aX) < B) && ((k + aY) < I)) ? __float2half(static_cast<float>(input[aRow + aX][k + aY]) * DIVISOR) : __float2half(0.0f);
            subTileB1[bY][bX] = (((bCol + bY) < R) && ((k + bX) < I)) ? __float2half(static_cast<float>(S1s[term][k + bX][bCol + bY])) : __float2half(0.0f);
            subTileB2[bY][bX] = (((bCol + bY) < R) && ((k + bX) < I)) ? __float2half(static_cast<float>(U2s[term][k + bX][bCol + bY])) : __float2half(0.0f);
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
            wmma::store_matrix_sync(&partial[blockIdx.z][0][term][cRow][cCol], acc1, R, wmma::mem_row_major);
            wmma::store_matrix_sync(&partial[blockIdx.z][1][term][cRow][cCol], acc2, R, wmma::mem_row_major);
        }
    }
}

__global__ void sklinear_forward_output_wmma(
    const torch::PackedTensorAccessor32<float_t, 4, torch::RestrictPtrTraits> inter,  // [2,T,B,R]
    const torch::PackedTensorAccessor32<float_t, 3, torch::RestrictPtrTraits> U1s,    // [T,R,O]
    const torch::PackedTensorAccessor32<float_t, 3, torch::RestrictPtrTraits> S2s,    // [T,R,O]
    const torch::PackedTensorAccessor32<float_t, 1, torch::RestrictPtrTraits> bias,   // [O]
    torch::PackedTensorAccessor32<float_t, 2, torch::RestrictPtrTraits> out,          // [B,O]
    int B, int R, int T, int O) {
    __shared__ half subTileA1[TILE_WIDTH_K][TILE_WIDTH_M];
    __shared__ half subTileA2[TILE_WIDTH_K][TILE_WIDTH_M];
    __shared__ half subTileB1[TILE_WIDTH_N][TILE_WIDTH_K];
    __shared__ half subTileB2[TILE_WIDTH_N][TILE_WIDTH_K];
    __shared__ float_t shBias[TILE_WIDTH_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + threadIdx.x;

    int base_b = blockIdx.x * TILE_WIDTH_M;  // aRow
    int base_o = blockIdx.y * TILE_WIDTH_N;  // bCol

    wmma::fragment<wmma::matrix_a, M, N, K, half_t, wmma::col_major> fA1, fA2;
    wmma::fragment<wmma::matrix_b, M, N, K, half_t, wmma::col_major> fB1, fB2;
    wmma::fragment<wmma::accumulator, M, N, K, float_t> acc;
    wmma::fragment<wmma::accumulator, M, N, K, float_t> c_frag;
    wmma::fill_fragment(acc, 0.0f);

    // load bias into memory N
    for (int i = tx + ty * blockDim.x; i < N; i += blockDim.x * blockDim.y) {
        shBias[i] = (base_o + i) < O ? bias[base_o + i] : 0.0f;
    }

    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < R; k += TILE_WIDTH_K) {
            for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
                int idx = tid + i;
                int aX = idx % TILE_WIDTH_M;
                int aY = idx / TILE_WIDTH_M;
                int bX = idx % TILE_WIDTH_K;
                int bY = idx / TILE_WIDTH_K;
                subTileA1[aY][aX] = (((base_b + aX) < B) && ((k + aY) < R)) ? __float2half(static_cast<float>(inter[0][t][base_b + aX][k + aY])) : __float2half(0.0f);
                subTileA2[aY][aX] = (((base_b + aX) < B) && ((k + aY) < R)) ? __float2half(static_cast<float>(inter[1][t][base_b + aX][k + aY])) : __float2half(0.0f);
                subTileB1[bY][bX] = (((base_o + bY) < O) && ((k + bX) < R)) ? __float2half(static_cast<float>(U1s[t][k + bX][base_o + bY])) : __float2half(0.0f);
                subTileB2[bY][bX] = (((base_o + bY) < O) && ((k + bX) < R)) ? __float2half(static_cast<float>(S2s[t][k + bX][base_o + bY])) : __float2half(0.0f);
            }
            __syncthreads();

            for (int i = 0; i < TILE_WIDTH_K; i += K) {
                int subtileARow = M * (tx / warpSize);
                int subtileACol = i;
                int subtileBRow = i;
                int subtileBCol = N * ty;

                wmma::load_matrix_sync(fA1, (half_t*)subTileA1 + subtileARow + subtileACol * TILE_WIDTH_M, TILE_WIDTH_M);
                wmma::load_matrix_sync(fA2, (half_t*)subTileA2 + subtileARow + subtileACol * TILE_WIDTH_M, TILE_WIDTH_M);
                wmma::load_matrix_sync(fB1, (half_t*)subTileB1 + subtileBRow + subtileBCol * TILE_WIDTH_K, TILE_WIDTH_K);
                wmma::load_matrix_sync(fB2, (half_t*)subTileB2 + subtileBRow + subtileBCol * TILE_WIDTH_K, TILE_WIDTH_K);
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
        wmma::load_matrix_sync(c_frag, shBias, 0, wmma::mem_row_major);
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = acc.x[i] + c_frag.x[i];
        }
        wmma::store_matrix_sync(&out[cRow][cCol], c_frag, O, wmma::mem_row_major);
    }
}

// Host launcher: computes and passes shared_bytes for both kernels
torch::Tensor sketched_linear_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& S1s,
    const torch::Tensor& S2s,
    const torch::Tensor& U1s,
    const torch::Tensor& U2s,
    const torch::Tensor& bias) {
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only FP32 supported");
    TORCH_CHECK(input.is_contiguous() && S1s.is_contiguous() && U2s.is_contiguous());
    TORCH_CHECK(S2s.is_contiguous() && U1s.is_contiguous() && bias.is_contiguous());

    int B = input.size(0), I = input.size(1);
    int T = S1s.size(0), R = S1s.size(2), O = S2s.size(2);
    dim3 block(BLOCK_ROW_WARPS * 32, BLOCK_COL_WARPS);

    torch::Tensor interm;
    if (T > 1) {
        int64_t numTilesI = (I + TILE_WIDTH_K - 1) / TILE_WIDTH_K;
        auto partial = torch::zeros({numTilesI, 2, T, B, R}, input.options());
        dim3 grid1((B + TILE_WIDTH_M - 1) / TILE_WIDTH_M,
                   (R + TILE_WIDTH_N - 1) / TILE_WIDTH_N,
                   numTilesI);

        sklinear_forward_intermediate_wmma<<<grid1, block>>>(
            input.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            S1s.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            U2s.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            partial.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            B, I, R, T, (1.0f / (2.0f * T)));

        interm = partial.sum(0);
    } else {
        auto input_expanded = input.unsqueeze(0);
        interm = torch::stack({input_expanded.bmm(S1s), input_expanded.bmm(U2s)}, 0);
    }

    if (R <= 96) {
        auto out = torch::zeros({B, O}, input.options());
        dim3 grid2((B + TILE_WIDTH_M - 1) / TILE_WIDTH_M,
                   (O + TILE_WIDTH_N - 1) / TILE_WIDTH_N);

        sklinear_forward_output_wmma<<<grid2, block>>>(
            interm.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            U1s.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            S2s.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            bias.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            B, R, T, O);
        return out;
    } else {
        return (interm[0].bmm(U1s) + interm[1].bmm(S2s)).mean(0).div(2.0f) + bias;
    }
}

__global__ void sklinear_backward_intermediate_wmma(
    const torch::PackedTensorAccessor32<float_t, 2, torch::RestrictPtrTraits> grad_output,  // [B,O]
    const torch::PackedTensorAccessor32<float_t, 3, torch::RestrictPtrTraits> U1s,          // [T,R,O]
    const torch::PackedTensorAccessor32<float_t, 3, torch::RestrictPtrTraits> S2s,          // [T,R,O]
    torch::PackedTensorAccessor32<float_t, 5, torch::RestrictPtrTraits> grad_intermediate,  // [numTilesK,2,T,B,R]
    int B, int O, int T, int R) {
    __shared__ half subTileA[TILE_WIDTH_K][TILE_WIDTH_M];
    __shared__ half subTileB1[TILE_WIDTH_N][TILE_WIDTH_K];
    __shared__ half subTileB2[TILE_WIDTH_N][TILE_WIDTH_K];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int aRow = blockIdx.x * TILE_WIDTH_M;  // aRow
    int bCol = blockIdx.y * TILE_WIDTH_N;  // bCol

    int k = blockIdx.z * TILE_WIDTH_K;
    wmma::fragment<wmma::matrix_a, M, N, K, half_t, wmma::col_major> fA;
    wmma::fragment<wmma::matrix_b, M, N, K, half_t, wmma::col_major> fB1, fB2;
    wmma::fragment<wmma::accumulator, M, N, K, float_t> acc1, acc2;
    for (int term = 0; term < T; term++) {
        wmma::fill_fragment(acc1, 0.0f);
        wmma::fill_fragment(acc2, 0.0f);
#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
            int idx = tid + i;
            int aX = idx % TILE_WIDTH_M;
            int aY = idx / TILE_WIDTH_M;
            int bX = idx % TILE_WIDTH_K;
            int bY = idx / TILE_WIDTH_K;
            if (term == 0) subTileA[aY][aX] = (((aRow + aX) < B) && ((k + aY) < O)) ? __float2half(static_cast<float>(grad_output[aRow + aX][k + aY])) : __float2half(0.0f);
            subTileB1[bY][bX] = (((bCol + bY) < R) && ((k + bX) < O)) ? __float2half(static_cast<float>(U1s[term][bCol + bY][k + bX])) : __float2half(0.0f);
            subTileB2[bY][bX] = (((bCol + bY) < R) && ((k + bX) < O)) ? __float2half(static_cast<float>(S2s[term][bCol + bY][k + bX])) : __float2half(0.0f);
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
            wmma::store_matrix_sync(&grad_intermediate[blockIdx.z][0][term][cRow][cCol], acc1, R, wmma::mem_row_major);
            wmma::store_matrix_sync(&grad_intermediate[blockIdx.z][1][term][cRow][cCol], acc2, R, wmma::mem_row_major);
        }
    }
}

__global__ void sklinear_backward_grad_S2_interm_wmma(
    const torch::PackedTensorAccessor32<float_t, 2, torch::RestrictPtrTraits> input,     // [B,I]
    const torch::PackedTensorAccessor32<float_t, 3, torch::RestrictPtrTraits> U2s,       // [T,I,R]
    torch::PackedTensorAccessor32<float_t, 4, torch::RestrictPtrTraits> interm_gradS2s,  // [numTilesK, T, R, B]
    int B, int I, int R, int T) {
    __shared__ half subTileA[TILE_WIDTH_K][TILE_WIDTH_M];  // U2s
    __shared__ half subTileB[TILE_WIDTH_N][TILE_WIDTH_K];  // input

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int aRow = blockIdx.x * TILE_WIDTH_M;  // aRow -> R
    int bCol = blockIdx.y * TILE_WIDTH_N;  // bCol -> B

    int k = blockIdx.z * TILE_WIDTH_K;  // -> I
    wmma::fragment<wmma::matrix_a, M, N, K, half_t, wmma::col_major> fA;
    wmma::fragment<wmma::matrix_b, M, N, K, half_t, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, M, N, K, float_t> acc;
    for (int term = 0; term < T; term++) {
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
            int idx = tid + i;
            int aX = idx % TILE_WIDTH_M;
            int aY = idx / TILE_WIDTH_M;
            int bX = idx % TILE_WIDTH_K;
            int bY = idx / TILE_WIDTH_K;
            subTileA[aY][aX] = (((aRow + aX) < R) && ((k + aY) < I)) ? __float2half(static_cast<float>(U2s[term][k + aY][aRow + aX])) : __float2half(0.0f);
            if (term == 0) subTileB[bY][bX] = (((bCol + bY) < B) && ((k + bX) < I)) ? __float2half(static_cast<float>(input[bCol + bY][k + bX])) : __float2half(0.0f);
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
            wmma::store_matrix_sync(&interm_gradS2s[blockIdx.z][term][cRow][cCol], acc, B, wmma::mem_row_major);
        }
    }
}

__global__ void sklinear_backward_grad_S2_output_wmma(
    const torch::PackedTensorAccessor32<float_t, 3, torch::RestrictPtrTraits> interm_gradS2s,  // [T, R, B]
    const torch::PackedTensorAccessor32<float_t, 2, torch::RestrictPtrTraits> grad_output,     // [B, O]
    torch::PackedTensorAccessor32<float_t, 3, torch::RestrictPtrTraits> grad_S2s,              // [T, R, O]
    int B, int R, int T, int O) {
    __shared__ half subTileA[TILE_WIDTH_K][TILE_WIDTH_M];  // interm_gradS2s
    __shared__ half subTileB[TILE_WIDTH_N][TILE_WIDTH_K];  // grad_output

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int aRow = blockIdx.x * TILE_WIDTH_M;  // aRow -> R
    int bCol = blockIdx.y * TILE_WIDTH_N;  // bCol -> O

    int term = blockIdx.z;  // -> T
    wmma::fragment<wmma::matrix_a, M, N, K, half_t, wmma::col_major> fA;
    wmma::fragment<wmma::matrix_b, M, N, K, half_t, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, M, N, K, float_t> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int k = 0; k < B; k += TILE_WIDTH_K) {
#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
            int idx = tid + i;
            int aX = idx % TILE_WIDTH_M;
            int aY = idx / TILE_WIDTH_M;
            int bX = idx % TILE_WIDTH_K;
            int bY = idx / TILE_WIDTH_K;
            subTileA[aY][aX] = (((aRow + aX) < R) && ((k + aY) < B)) ? __float2half(static_cast<float>(interm_gradS2s[term][aRow + aX][k + aY])) : __float2half(0.0f);
            subTileB[bY][bX] = (((bCol + bY) < O) && ((k + bX) < B)) ? __float2half(static_cast<float>(grad_output[k + bX][bCol + bY])) : __float2half(0.0f);
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
        wmma::store_matrix_sync(&grad_S2s[term][cRow][cCol], acc, O, wmma::mem_row_major);
    }
}

__global__ void sklinear_backward_grad_input_wmma(
    const torch::PackedTensorAccessor32<float_t, 4, torch::RestrictPtrTraits> grad_interm,  // [2,T,B,R]
    const torch::PackedTensorAccessor32<float_t, 3, torch::RestrictPtrTraits> S1s,          // [T,I,R]
    const torch::PackedTensorAccessor32<float_t, 3, torch::RestrictPtrTraits> U2s,          // [T,I,R]
    torch::PackedTensorAccessor32<float_t, 2, torch::RestrictPtrTraits> grad_input,         // [B,I]
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
    wmma::fragment<wmma::matrix_a, M, N, K, half_t, wmma::col_major> fA1, fA2;
    wmma::fragment<wmma::matrix_b, M, N, K, half_t, wmma::col_major> fB1, fB2;
    wmma::fragment<wmma::accumulator, M, N, K, float_t> acc1, acc2;
    wmma::fill_fragment(acc1, 0.0f);
    wmma::fill_fragment(acc2, 0.0f);

    for (int term = 0; term < T; term++) {
        for (int k = 0; k < R; k += TILE_WIDTH_K) {
#pragma unroll 1
            for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
                int idx = tid + i;
                int aX = idx % TILE_WIDTH_M;
                int aY = idx / TILE_WIDTH_M;
                int bX = idx % TILE_WIDTH_K;
                int bY = idx / TILE_WIDTH_K;
                subTileA1[aY][aX] = (((aRow + aX) < B) && ((k + aY) < R)) ? __float2half(static_cast<float>(grad_interm[0][term][aRow + aX][k + aY])) : __float2half(0.0f);
                subTileA2[aY][aX] = (((aRow + aX) < B) && ((k + aY) < R)) ? __float2half(static_cast<float>(grad_interm[1][term][aRow + aX][k + aY])) : __float2half(0.0f);
                subTileB1[bY][bX] = (((bCol + bY) < I) && ((k + bX) < R)) ? __float2half(static_cast<float>(S1s[term][bCol + bY][k + bX])) : __float2half(0.0f);
                subTileB2[bY][bX] = (((bCol + bY) < I) && ((k + bX) < R)) ? __float2half(static_cast<float>(U2s[term][bCol + bY][k + bX])) : __float2half(0.0f);
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
                wmma::mma_sync(acc2, fA2, fB2, acc2);
            }
        }
    }

    int warpM = (blockIdx.x * blockDim.x + tx) / warpSize;
    int warpN = blockIdx.y * blockDim.y + ty;
    int cRow = warpM * M;
    int cCol = warpN * N;

    if (cRow < B && cCol < I) {
        for (int i = 0; i < acc1.num_elements; i++) {
            acc1.x[i] = acc1.x[i] + acc2.x[i];
        }
        wmma::store_matrix_sync(&grad_input[cRow][cCol], acc1, I, wmma::mem_row_major);
    }
}

__global__ void sklinear_backward_grad_S1_wmma(
    const torch::PackedTensorAccessor32<float_t, 2, torch::RestrictPtrTraits> input,    // [B,I]
    const torch::PackedTensorAccessor32<float_t, 3, torch::RestrictPtrTraits> interm0,  // [T,B,R]
    torch::PackedTensorAccessor32<float_t, 3, torch::RestrictPtrTraits> grad_S1s,       // [T,I,R]
    int B, int I, int R, int T) {
    __shared__ half subTileA[TILE_WIDTH_K][TILE_WIDTH_M];  // input
    __shared__ half subTileB[TILE_WIDTH_N][TILE_WIDTH_K];  // interm0

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;
    int aRow = blockIdx.x * TILE_WIDTH_M;  // aRow -> I
    int bCol = blockIdx.y * TILE_WIDTH_N;  // bCol -> R

    int term = blockIdx.z;  // -> T

    wmma::fragment<wmma::matrix_a, M, N, K, half_t, wmma::col_major> fA;
    wmma::fragment<wmma::matrix_b, M, N, K, half_t, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, M, N, K, float_t> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int k = 0; k < B; k += TILE_WIDTH_K) {
#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
            int idx = tid + i;
            int aX = idx % TILE_WIDTH_M;
            int aY = idx / TILE_WIDTH_M;
            int bX = idx % TILE_WIDTH_K;
            int bY = idx / TILE_WIDTH_K;
            subTileA[aY][aX] = (((aRow + aX) < I) && ((k + aY) < B)) ? __float2half(static_cast<float>(input[k + aY][aRow + aX])) : __float2half(0.0f);
            subTileB[bY][bX] = (((bCol + bY) < R) && ((k + bX) < B)) ? __float2half(static_cast<float>(interm0[term][k + bX][bCol + bY])) : __float2half(0.0f);
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
        wmma::store_matrix_sync(&grad_S1s[term][cRow][cCol], acc, R, wmma::mem_row_major);
    }
}

std::vector<torch::Tensor> sketched_linear_backward_cuda(
    const torch::Tensor& grad_output,  // [B, O]
    const torch::Tensor& input,        // [B, I]
    const torch::Tensor& S1s,          // [T, I, R]
    const torch::Tensor& S2s,          // [T, R, O]
    const torch::Tensor& U1s,          // [T, R, O]
    const torch::Tensor& U2s) {        // [T, I, R]
    TORCH_CHECK(input.scalar_type() == at::kFloat, "Only FP32 supported");
    TORCH_CHECK(input.is_contiguous() && S1s.is_contiguous() && U2s.is_contiguous());
    TORCH_CHECK(S2s.is_contiguous() && U1s.is_contiguous());
    TORCH_CHECK(grad_output.is_contiguous());
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

    auto grad_intermediate = torch::zeros({numTilesO, 2, T, B, R}, input.options());

    sklinear_backward_intermediate_wmma<<<grid1, block, 0, stream1>>>(
        g.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        U1s.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        S2s.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        grad_intermediate.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
        B, O, T, R);

    int64_t numTilesI = (I + TILE_WIDTH_K - 1) / TILE_WIDTH_K;
    at::cuda::CUDAStream torch_stream2 = at::cuda::getStreamFromPool(false, device_id);
    cudaStream_t stream2 = torch_stream2.stream();
    at::cuda::setCurrentCUDAStream(torch_stream2);
    auto interm_gradS2s = torch::zeros({numTilesI, T, R, B}, input.options());

    dim3 grid2((R + TILE_WIDTH_M - 1) / TILE_WIDTH_M,
               (B + TILE_WIDTH_N - 1) / TILE_WIDTH_N,
               numTilesI);
    sklinear_backward_grad_S2_interm_wmma<<<grid2, block, 0, stream2>>>(
        input.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        U2s.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        interm_gradS2s.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        B, I, R, T);

    at::cuda::setCurrentCUDAStream(torch_stream2);
    auto i_gradS2s = interm_gradS2s.sum(0).contiguous();
    auto grad_S2s = torch::zeros({T, R, O}, input.options());
    cudaStreamWaitEvent(stream2, afterGcompute);
    dim3 grid4((R + TILE_WIDTH_M - 1) / TILE_WIDTH_M,
               (O + TILE_WIDTH_N - 1) / TILE_WIDTH_N,
               T);
    sklinear_backward_grad_S2_output_wmma<<<grid4, block, 0, stream2>>>(
        i_gradS2s.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        g.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        grad_S2s.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        B, R, T, O);

    at::cuda::setCurrentCUDAStream(torch_stream1);
    auto interm = grad_intermediate.sum(0);

    cudaEvent_t afterIntermSum;
    cudaEventCreate(&afterIntermSum);
    cudaEventRecord(afterIntermSum, stream1);

    auto grad_input = torch::zeros({B, I}, input.options());
    dim3 grid3((B + TILE_WIDTH_M - 1) / TILE_WIDTH_M,
               (I + TILE_WIDTH_N - 1) / TILE_WIDTH_N);
    sklinear_backward_grad_input_wmma<<<grid3, block, 0, stream1>>>(
        interm.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        S1s.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        U2s.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        grad_input.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        B, I, R, T);
    // auto grad_input = (interm[0].bmm(S1s.transpose(1, 2)) + interm[1].bmm(U2s.transpose(1, 2))).sum(0);

    at::cuda::CUDAStream torch_stream3 = at::cuda::getStreamFromPool(false, device_id);
    cudaStream_t stream3 = torch_stream3.stream();
    at::cuda::setCurrentCUDAStream(torch_stream3);
    auto grad_S1s = torch::zeros({T, I, R}, input.options());

    cudaStreamWaitEvent(stream3, afterIntermSum);

    auto interm0 = interm[0].contiguous();

    dim3 grid5((I + TILE_WIDTH_M - 1) / TILE_WIDTH_M,
               (R + TILE_WIDTH_N - 1) / TILE_WIDTH_N,
               T);
    sklinear_backward_grad_S1_wmma<<<grid5, block, 0, stream3>>>(
        input.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        interm0.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        grad_S1s.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        B, I, R, T);

    at::cuda::CUDAStream torch_stream4 = at::cuda::getStreamFromPool(false, device_id);
    cudaStream_t stream4 = torch_stream4.stream();
    at::cuda::setCurrentCUDAStream(torch_stream4);
    auto grad_o = grad_output.sum(0);

    at::cuda::stream_synchronize(stream1);
    at::cuda::stream_synchronize(stream2);
    at::cuda::stream_synchronize(stream3);
    at::cuda::stream_synchronize(stream4);
    torch::cuda::synchronize(device_id);
    at::cuda::setCurrentCUDAStream(at::cuda::getDefaultCUDAStream(device_id));
    cudaEventDestroy(afterIntermSum);
    cudaEventDestroy(afterGcompute);

    return {grad_input, grad_S1s, grad_S2s, grad_o};
}
