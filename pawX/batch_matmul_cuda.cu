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

// Optimized tiled batch matrix multiplication kernel
template <typename scalar_t>
__global__ void tiled_batch_matmul_kernel(
    const scalar_t* __restrict__ A,  // [B, M, K]
    const scalar_t* __restrict__ B,  // [B, K, N]
    scalar_t* __restrict__ C,        // [B, M, N]
    const int batch_size,            // batch size
    const int M,                     // rows of A
    const int N,                     // cols of B
    const int K) {                   // cols of A / rows of B

    // Shared memory for tiles
    __shared__ scalar_t As[TILE_SIZE][TILE_SIZE + 1];  // +1 for bank conflict avoidance
    __shared__ scalar_t Bs[TILE_SIZE][TILE_SIZE + 1];

    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int b = blockIdx.z;

    // Calculate global indices
    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;

    // Initialize accumulator
    scalar_t sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
// Load tiles into shared memory
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

// Compute partial sum
#pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[b * M * N + row * N + col] = sum;
    }
}

// Vectorized version for better memory throughput
template <typename scalar_t, int VECTOR_WIDTH>
__global__ void vectorized_tiled_batch_matmul_kernel(
    const scalar_t* __restrict__ A,  // [B, M, K]
    const scalar_t* __restrict__ B,  // [B, K, N]
    scalar_t* __restrict__ C,        // [B, M, N]
    const int batch_size,            // batch size
    const int M,                     // rows of A
    const int N,                     // cols of B
    const int K) {                   // cols of A / rows of B

    // Shared memory for tiles
    __shared__ scalar_t As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ scalar_t Bs[TILE_SIZE][TILE_SIZE + 1];

    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int b = blockIdx.z;

    // Calculate global indices
    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx * VECTOR_WIDTH;

    // Initialize accumulators
    scalar_t sum[VECTOR_WIDTH] = {0.0f};

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
// Load tiles into shared memory using vector loads
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

// Compute partial sums
#pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
#pragma unroll
            for (int v = 0; v < VECTOR_WIDTH; v++) {
                sum[v] += As[ty][i] * Bs[i][tx * VECTOR_WIDTH + v];
            }
        }

        __syncthreads();
    }

    // Write results using vector stores
    if (row < M && col < N) {
#pragma unroll
        for (int v = 0; v < VECTOR_WIDTH; v++) {
            if (col + v < N) {
                C[b * M * N + row * N + col + v] = sum[v];
            }
        }
    }
}

// Wrapper function with automatic kernel selection
torch::Tensor tiled_batch_matmul_cuda_forward(
    const torch::Tensor& A,
    const torch::Tensor& B) {
    // Get dimensions
    const int batch_size = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    // Create output tensor
    auto C = torch::zeros({batch_size, M, N}, A.options());

    // Calculate grid and block dimensions
    dim3 block(NUM_THREADS, NUM_THREADS);
    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        batch_size);

    // Dispatch based on input type and size
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