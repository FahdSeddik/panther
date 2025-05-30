#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <torch/extension.h>

#include "cuda_tensor_accessor.cuh"
#include "debug.cuh"
#include "kernel_launch.cuh"

constexpr int TILE_SIZE = 16;
constexpr int BLOCK_ROW_WARPS = 4;
constexpr int BLOCK_COL_WARPS = 4;
constexpr int WARPS_PER_BLOCK = BLOCK_ROW_WARPS * BLOCK_COL_WARPS;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

constexpr int TILE_WIDTH_M = TILE_SIZE * BLOCK_ROW_WARPS;
constexpr int TILE_WIDTH_N = TILE_SIZE * BLOCK_COL_WARPS;
constexpr int TILE_WIDTH_K = TILE_WIDTH_M;

using half_t = __half;
using float_t = float;
namespace wmma = nvcuda::wmma;

/**
 * @brief CUDA kernel for performing batch matrix multiplication using WMMA (Warp Matrix Multiply and Accumulate).
 *
 * This kernel computes the matrix multiplication of batched matrices using NVIDIA's WMMA API for efficient
 * half-precision tensor core operations. The computation is tiled and leverages shared memory for sub-tiles
 * of the input matrices.
 *
 * @tparam scalar_t The scalar type for computation (e.g., float, half).
 * @param A [in] FlexibleTensorAccessor<scalar_t, 3> - Input tensor of shape [B, M, K].
 * @param B [in] FlexibleTensorAccessor<scalar_t, 3> - Input tensor of shape [B, K, N].
 * @param C [out] FlexibleTensorAccessor<scalar_t, 3> - Output tensor of shape [B, M, N].
 * @param batch_size Batch size (first dimension of all tensors).
 * @param M Number of rows in matrix A.
 * @param N Number of columns in matrix B.
 * @param K Number of columns in matrix A / rows in matrix B.
 */
template <typename scalar_t>
__global__ void tiled_batch_matmul_tc_kernel(
    const FlexibleTensorAccessor<scalar_t, 3> A,  // [B, M, K]
    const FlexibleTensorAccessor<scalar_t, 3> B,  // [B, K, N]
    FlexibleTensorAccessor<scalar_t, 3> C,        // [B, M, N]
    int batch_size, int M, int N, int K) {
    __shared__ half_t As[TILE_WIDTH_K][TILE_WIDTH_M];
    __shared__ half_t Bs[TILE_WIDTH_N][TILE_WIDTH_K];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int b = blockIdx.z;

    const int row = by * TILE_WIDTH_M;
    const int col = bx * TILE_WIDTH_N;

    wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half_t, wmma::col_major> fA;
    wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half_t, wmma::col_major> fB;
    wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, cuda_type_t<scalar_t>> acc;
    auto zero = cuda_type<scalar_t>::get_zero();
    wmma::fill_fragment(acc, zero);

    for (int t = 0; t < (K + TILE_WIDTH_K - 1) / TILE_WIDTH_K; t++) {
#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
            int idx = tid + i;
            int aX = idx % TILE_WIDTH_M;
            int aY = idx / TILE_WIDTH_M;
            int bX = idx % TILE_WIDTH_K;
            int bY = idx / TILE_WIDTH_K;

            if (row + aX < M && t * TILE_WIDTH_K + aY < K) {
                As[aY][aX] = A.template get<half_t>(b, row + aX, t * TILE_WIDTH_K + aY);
            } else {
                As[aY][aX] = __float2half(0.0f);
            }

            if (col + bY < N && t * TILE_WIDTH_K + bX < K) {
                Bs[bY][bX] = B.template get<half_t>(b, t * TILE_WIDTH_K + bX, col + bY);
            } else {
                Bs[bY][bX] = __float2half(0.0f);
            }
        }
        __syncthreads();

#pragma unroll 1
        for (int i = 0; i < TILE_WIDTH_K; i += TILE_SIZE) {
            int subtileARow = TILE_SIZE * (tx / warpSize);
            int subtileACol = i;
            int subtileBRow = i;
            int subtileBCol = TILE_SIZE * ty;

            wmma::load_matrix_sync(fA, (half_t*)As + subtileARow + subtileACol * TILE_WIDTH_M, TILE_WIDTH_M);
            wmma::load_matrix_sync(fB, (half_t*)Bs + subtileBRow + subtileBCol * TILE_WIDTH_K, TILE_WIDTH_K);
            wmma::mma_sync(acc, fA, fB, acc);
        }
        __syncthreads();
    }

    int warpM = (blockIdx.x * blockDim.x + tx) / warpSize;
    int warpN = blockIdx.y * blockDim.y + ty;
    int cRow = warpM * TILE_SIZE;
    int cCol = warpN * TILE_SIZE;

    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(&C(b, cRow, cCol), acc, N, wmma::mem_row_major);
    }
}

/**
 * @brief Performs batch matrix multiplication using tensor cores.
 *
 * This function computes the matrix multiplication of batched matrices using NVIDIA's WMMA API
 * for efficient half-precision tensor core operations. It supports both float and half precision inputs.
 *
 * @param A Input tensor of shape (B, M, K).
 * @param B Input tensor of shape (B, K, N).
 * @return torch::Tensor Output tensor of shape (B, M, N).
 *
 * @note All input tensors must be contiguous. The function dispatches to a custom CUDA kernel
 *       that uses tensor cores for efficient matrix multiplication.
 */
torch::Tensor tiled_batch_matmul_tc_forward(
    const torch::Tensor& A,
    const torch::Tensor& B) {
    TORCH_CHECK(A.scalar_type() == at::kFloat || A.scalar_type() == at::kHalf,
                "Input tensor must be float or half precision.");
    TORCH_CHECK(A.is_contiguous(), "Input tensor A must be contiguous.");
    TORCH_CHECK(B.is_contiguous(), "Input tensor B must be contiguous.");

    const int batch_size = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    auto C = torch::zeros({batch_size, M, N}, A.options());

    dim3 block(BLOCK_ROW_WARPS * 32, BLOCK_COL_WARPS);
    dim3 grid(
        (N + TILE_WIDTH_N - 1) / TILE_WIDTH_N,
        (M + TILE_WIDTH_M - 1) / TILE_WIDTH_M,
        batch_size);

    AT_DISPATCH_FLOAT_AND_HALF(
        A.scalar_type(),
        "tiled_batch_matmul_tc_kernel",
        [&] {
            tiled_batch_matmul_tc_kernel<scalar_t><<<grid, block>>>(
                tensor_utils::buildAccessor<scalar_t, 3>(A),
                tensor_utils::buildAccessor<scalar_t, 3>(B),
                tensor_utils::buildAccessor<scalar_t, 3>(C),
                batch_size, M, N, K);
        });

    return C;
}