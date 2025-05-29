#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "kernel_launch.cuh"

// Constants for optimization
constexpr int TILE_SIZE = 16;    // Tile size for shared memory
constexpr int BLOCK_SIZE = 256;  // Threads per block
constexpr int NUM_THREADS = 16;  // Threads per tile dimension
constexpr int VECTOR_SIZE = 4;   // Vector load/store size

/**
 * @brief CUDA kernel for batched matrix multiplication using tiling and shared memory.
 *
 * This kernel computes C = A x B for a batch of matrices, where:
 *   - A has shape [batch_size, M, K]
 *   - B has shape [batch_size, K, N]
 *   - C has shape [batch_size, M, N]
 *
 * The computation is performed in tiles to optimize memory access and utilize shared memory.
 * Shared memory tiles are padded to avoid bank conflicts.
 *
 * @tparam scalar_t Data type of the matrix elements (e.g., float, double).
 * @param[in]  A           Pointer to input matrix A (batched).
 * @param[in]  B           Pointer to input matrix B (batched).
 * @param[out] C           Pointer to output matrix C (batched).
 * @param[in]  batch_size  Number of matrices in the batch.
 * @param[in]  M           Number of rows in each A and C matrix.
 * @param[in]  N           Number of columns in each B and C matrix.
 * @param[in]  K           Number of columns in A and rows in B.
 *
 * Each block computes a TILE_SIZE x TILE_SIZE submatrix of C for a specific batch index.
 * Threads within a block collaboratively load tiles of A and B into shared memory,
 * perform partial multiplications, and accumulate the results.
 *
 * Assumes grid and block dimensions are set such that:
 *   - blockDim.x == blockDim.y == TILE_SIZE
 *   - gridDim.x == ceil(N / TILE_SIZE)
 *   - gridDim.y == ceil(M / TILE_SIZE)
 *   - gridDim.z == batch_size
 *
 * @note TILE_SIZE and NUM_THREADS must be defined appropriately for the kernel to function correctly.
 */
template <typename scalar_t>
__global__ void tiled_batch_matmul_kernel(
    const scalar_t* __restrict__ A,  // [B, M, K]
    const scalar_t* __restrict__ B,  // [B, K, N]
    scalar_t* __restrict__ C,        // [B, M, N]
    const int batch_size,            // batch size
    const int M,                     // rows of A
    const int N,                     // cols of B
    const int K) {                   // cols of A / rows of B

    __shared__ scalar_t As[TILE_SIZE][TILE_SIZE + 1];  // +1 for bank conflict avoidance
    __shared__ scalar_t Bs[TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int b = blockIdx.z;

    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;

    scalar_t sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
#pragma unroll
        for (int i = 0; i < TILE_SIZE; i += NUM_THREADS) {
            if (row < M && t * TILE_SIZE + i < K) {
                As[ty + i][tx] = A[b * M * K + row * K + t * TILE_SIZE + i];
            } else {
                As[ty + i][tx] = 0.0f;
            }

            if (col < N && t * TILE_SIZE + i < K) {
                Bs[ty + i][tx] = B[b * K * N + (t * TILE_SIZE + i) * N + col];
            } else {
                Bs[ty + i][tx] = 0.0f;
            }
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[b * M * N + row * N + col] = sum;
    }
}

/**
 * @brief Vectorized and tiled batched matrix multiplication CUDA kernel.
 *
 * This kernel performs batched matrix multiplication C = A x B for a batch of matrices,
 * using shared memory tiling and vectorized memory access for improved performance.
 * Each thread block computes a TILE_SIZE x TILE_SIZE output tile for a single batch.
 * Vectorization is controlled by the VECTOR_WIDTH template parameter.
 *
 * @tparam scalar_t      Data type of matrix elements (e.g., float, double).
 * @tparam VECTOR_WIDTH  Number of elements processed per thread in the output tile (vectorization factor).
 *
 * @param[in]  A         Pointer to input matrix A of shape [batch_size, M, K].
 * @param[in]  B         Pointer to input matrix B of shape [batch_size, K, N].
 * @param[out] C         Pointer to output matrix C of shape [batch_size, M, N].
 * @param[in]  batch_size Number of matrices in the batch.
 * @param[in]  M         Number of rows in matrix A (and C).
 * @param[in]  N         Number of columns in matrix B (and C).
 * @param[in]  K         Number of columns in matrix A / rows in matrix B.
 *
 * Shared memory tiles As and Bs are used to cache sub-blocks of A and B, respectively.
 * The kernel loops over tiles of the K dimension, accumulating partial results in registers.
 * Each thread computes VECTOR_WIDTH output elements in the output tile.
 *
 * Grid and block configuration:
 *   - gridDim.x = ceil(N / (TILE_SIZE * VECTOR_WIDTH))
 *   - gridDim.y = ceil(M / TILE_SIZE)
 *   - gridDim.z = batch_size
 *   - blockDim.x = TILE_SIZE / VECTOR_WIDTH
 *   - blockDim.y = TILE_SIZE
 *
 * Requirements:
 *   - TILE_SIZE and NUM_THREADS must be defined elsewhere.
 *   - TILE_SIZE should be divisible by VECTOR_WIDTH.
 */
template <typename scalar_t, int VECTOR_WIDTH>
__global__ void vectorized_tiled_batch_matmul_kernel(
    const scalar_t* __restrict__ A,  // [B, M, K]
    const scalar_t* __restrict__ B,  // [B, K, N]
    scalar_t* __restrict__ C,        // [B, M, N]
    const int batch_size,            // batch size
    const int M,                     // rows of A
    const int N,                     // cols of B
    const int K) {                   // cols of A / rows of B

    __shared__ scalar_t As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ scalar_t Bs[TILE_SIZE][TILE_SIZE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int b = blockIdx.z;

    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx * VECTOR_WIDTH;

    scalar_t sum[VECTOR_WIDTH] = {0.0f};

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
#pragma unroll
        for (int i = 0; i < TILE_SIZE; i += NUM_THREADS) {
            if (row < M && t * TILE_SIZE + i < K) {
                As[ty + i][tx] = A[b * M * K + row * K + t * TILE_SIZE + i];
            } else {
                As[ty + i][tx] = 0.0f;
            }

            if (col < N && t * TILE_SIZE + i < K) {
#pragma unroll
                for (int v = 0; v < VECTOR_WIDTH; v++) {
                    if (col + v < N) {
                        Bs[ty + i][tx * VECTOR_WIDTH + v] =
                            B[b * K * N + (t * TILE_SIZE + i) * N + col + v];
                    } else {
                        Bs[ty + i][tx * VECTOR_WIDTH + v] = 0.0f;
                    }
                }
            }
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
#pragma unroll
            for (int v = 0; v < VECTOR_WIDTH; v++) {
                sum[v] += As[ty][i] * Bs[i][tx * VECTOR_WIDTH + v];
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
#pragma unroll
        for (int v = 0; v < VECTOR_WIDTH; v++) {
            if (col + v < N) {
                C[b * M * N + row * N + col + v] = sum[v];
            }
        }
    }
}

/**
 * @brief Performs a batched matrix multiplication using tiled CUDA kernels.
 *
 * This function dispatches to either a vectorized or standard tiled batch matrix multiplication CUDA kernel,
 * depending on the data type and alignment of the input matrices. It supports both float and half precision types.
 *
 * @param A Input tensor of shape (batch_size, M, K).
 * @param B Input tensor of shape (batch_size, K, N).
 * @return torch::Tensor Output tensor of shape (batch_size, M, N) containing the batched matrix multiplication results.
 *
 * The function automatically selects the appropriate kernel based on the scalar type of the input tensors and
 * whether the output dimension N is divisible by VECTOR_SIZE for vectorized computation.
 *
 * @throws AT_ERROR if the input data type is not supported.
 */
torch::Tensor tiled_batch_matmul_cuda_forward(
    const torch::Tensor& A,
    const torch::Tensor& B) {
    const int batch_size = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    auto C = torch::zeros({batch_size, M, N}, A.options());

    dim3 block(NUM_THREADS, NUM_THREADS);
    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        batch_size);

    switch (A.scalar_type()) {
        case at::ScalarType::Float:
            if (N % VECTOR_SIZE == 0) {
                vectorized_tiled_batch_matmul_kernel<float, VECTOR_SIZE><<<grid, block>>>(
                    A.data_ptr<float>(),
                    B.data_ptr<float>(),
                    C.data_ptr<float>(),
                    batch_size, M, N, K);
            } else {
                tiled_batch_matmul_kernel<float><<<grid, block>>>(
                    A.data_ptr<float>(),
                    B.data_ptr<float>(),
                    C.data_ptr<float>(),
                    batch_size, M, N, K);
            }
            break;
        case at::ScalarType::Half:
            if (N % VECTOR_SIZE == 0) {
                vectorized_tiled_batch_matmul_kernel<half, VECTOR_SIZE><<<grid, block>>>(
                    A.data_ptr<half>(),
                    B.data_ptr<half>(),
                    C.data_ptr<half>(),
                    batch_size, M, N, K);
            } else {
                tiled_batch_matmul_kernel<half><<<grid, block>>>(
                    A.data_ptr<half>(),
                    B.data_ptr<half>(),
                    C.data_ptr<half>(),
                    batch_size, M, N, K);
            }
            break;
        default:
            AT_ERROR("Unsupported data type");
    }

    return C;
}