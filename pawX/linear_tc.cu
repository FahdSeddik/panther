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
    torch::PackedTensorAccessor32<float_t, 4, torch::RestrictPtrTraits> inter,        // [2,T,B,R]
    int B, int I, int R, const float_t DIVISOR) {
    // Dynamic shared memory buffer:
    __shared__ half subTileA[TILE_WIDTH_K][TILE_WIDTH_M];
    __shared__ half subTileB1[TILE_WIDTH_N][TILE_WIDTH_K];
    __shared__ half subTileB2[TILE_WIDTH_N][TILE_WIDTH_K];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    int term = blockIdx.z;
    int aRow = blockIdx.x * TILE_WIDTH_M;  // aRow
    int bCol = blockIdx.y * TILE_WIDTH_N;  // bCol

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, M, N, K, half_t, wmma::col_major> fA;
    wmma::fragment<wmma::matrix_b, M, N, K, half_t, wmma::col_major> fB1, fB2;
    wmma::fragment<wmma::accumulator, M, N, K, float_t> acc1, acc2;
    wmma::fill_fragment(acc1, 0.0f);
    wmma::fill_fragment(acc2, 0.0f);

    for (int k = 0; k < I; k += TILE_WIDTH_K) {
        for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
            int idx = tid + i;
            int aX = idx % TILE_WIDTH_M;
            int aY = idx / TILE_WIDTH_M;
            int bX = idx % TILE_WIDTH_K;
            int bY = idx / TILE_WIDTH_K;
            subTileA[aY][aX] = (((aRow + aX) < B) && ((k + aY) < I)) ? __float2half(static_cast<float>(input[aRow + aX][k + aY]) * DIVISOR) : __float2half(0.0f);
            subTileB1[bY][bX] = (((bCol + bY) < R) && ((k + bX) < I)) ? __float2half(static_cast<float>(S1s[term][k + bX][bCol + bY])) : __float2half(0.0f);
            subTileB2[bY][bX] = (((bCol + bY) < R) && ((k + bX) < I)) ? __float2half(static_cast<float>(U2s[term][k + bX][bCol + bY])) : __float2half(0.0f);
        }
        __syncthreads();

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
    }

    int warpM = (blockIdx.x * blockDim.x + tx) / warpSize;
    int warpN = blockIdx.y * blockDim.y + ty;
    int cRow = warpM * M;
    int cCol = warpN * N;

    if (cRow < B && cCol < R) {
        wmma::store_matrix_sync(&inter[0][term][cRow][cCol], acc1, R, wmma::mem_row_major);
        wmma::store_matrix_sync(&inter[1][term][cRow][cCol], acc2, R, wmma::mem_row_major);
    }
}

__global__ void sklinear_forward_output_wmma(
    const torch::PackedTensorAccessor32<float_t, 4, torch::RestrictPtrTraits> inter,  // [2,T,B,R]
    const torch::PackedTensorAccessor32<float_t, 3, torch::RestrictPtrTraits> U1s,    // [T,R,O]
    const torch::PackedTensorAccessor32<float_t, 3, torch::RestrictPtrTraits> S2s,    // [T,R,O]
    const torch::PackedTensorAccessor32<float_t, 1, torch::RestrictPtrTraits> bias,   // [O]
    torch::PackedTensorAccessor32<float_t, 2, torch::RestrictPtrTraits> out,          // [B,O]
    int B, int R, int T, int O) {
    extern __shared__ char _shm2[];
    half_t* shA1 = reinterpret_cast<half_t*>(_shm2);
    half_t* shA2 = shA1 + TILE_WIDTH_M * TILE_WIDTH_K;
    half_t* shB1 = shA2 + TILE_WIDTH_M * TILE_WIDTH_K;
    half_t* shB2 = shB1 + TILE_WIDTH_K * TILE_WIDTH_N;
    float_t* shBias = reinterpret_cast<float_t*>(shB2 + TILE_WIDTH_K * TILE_WIDTH_N);  // [O]

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + threadIdx.x;

    int brow = blockIdx.y;
    int rcol = blockIdx.x;
    int base_b = brow * TILE_WIDTH_M;  // aRow
    int base_o = rcol * TILE_WIDTH_N;  // bCol

    wmma::fragment<wmma::matrix_a, M, N, K, half_t, wmma::col_major> fA1, fA2;
    wmma::fragment<wmma::matrix_b, M, N, K, half_t, wmma::col_major> fB1, fB2;
    wmma::fragment<wmma::accumulator, M, N, K, float_t> acc;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
    wmma::fill_fragment(acc, 0.0f);

    // load bias into memory TILE_WIDTH_N
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
                shA1[aY * TILE_WIDTH_K + aX] = (((base_b + aX) < B) && ((k + aY) < R)) ? __float2half(static_cast<float>(inter[0][t][base_b + aX][k + aY])) : __float2half(0.0f);
                shA2[aY * TILE_WIDTH_K + aX] = (((base_b + aX) < B) && ((k + aY) < R)) ? __float2half(static_cast<float>(inter[1][t][base_b + aX][k + aY])) : __float2half(0.0f);
                shB1[bY * TILE_WIDTH_N + bX] = (((base_o + bY) < O) && ((k + bX) < R)) ? __float2half(static_cast<float>(U1s[t][k + bX][base_o + bY])) : __float2half(0.0f);
                shB2[bY * TILE_WIDTH_N + bX] = (((base_o + bY) < O) && ((k + bX) < R)) ? __float2half(static_cast<float>(S2s[t][k + bX][base_o + bY])) : __float2half(0.0f);
            }
            __syncthreads();

            for (int i = 0; i < TILE_WIDTH_K; i += K) {
                int subtileARow = M * (tx / warpSize);
                int subtileACol = i;
                int subtileBRow = i;
                int subtileBCol = N * ty;

                wmma::load_matrix_sync(fA1, (half_t*)shA1 + subtileARow + subtileACol * TILE_WIDTH_M, TILE_WIDTH_M);
                wmma::load_matrix_sync(fA2, (half_t*)shA2 + subtileARow + subtileACol * TILE_WIDTH_M, TILE_WIDTH_M);
                wmma::load_matrix_sync(fB1, (half_t*)shB1 + subtileBRow + subtileBCol * TILE_WIDTH_N, TILE_WIDTH_N);
                wmma::load_matrix_sync(fB2, (half_t*)shB2 + subtileBRow + subtileBCol * TILE_WIDTH_N, TILE_WIDTH_N);
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
        wmma::store_matrix_sync(&out[cRow][cCol], c_frag, R, wmma::mem_row_major);
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

    auto interm = torch::zeros({2, T, B, R}, input.options());

    auto out = torch::zeros({B, O}, input.options());

    dim3 block(BLOCK_ROW_WARPS * 32, BLOCK_COL_WARPS);
    dim3 grid1((B + TILE_WIDTH_M - 1) / TILE_WIDTH_M,
               (R + TILE_WIDTH_N - 1) / TILE_WIDTH_N,
               T);

    sklinear_forward_intermediate_wmma<<<grid1, block>>>(
        input.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        S1s.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        U2s.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        interm.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        B, I, R, (1.0f / (2.0f * T)));

    // dim3 grid2((O + TILE_WIDTH_N - 1) / TILE_WIDTH_N,
    //            (B + TILE_WIDTH_M - 1) / TILE_WIDTH_M);
    // // final: same half tiles + one float scratch + bias per warp-col
    // size_t shared_bytes2 = (2 * TILE_WIDTH_M * TILE_WIDTH_K + 2 * TILE_WIDTH_K * TILE_WIDTH_N) * sizeof(half_t) + N * sizeof(float_t);

    // sklinear_forward_output_wmma<<<grid2, block, shared_bytes2>>>(
    //     interm.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
    //     U1s.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
    //     S2s.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
    //     bias.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
    //     out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
    //     B, R, T, O);

    return (interm[0].bmm(U1s) + interm[1].bmm(S2s)).sum(0) + bias;
}
