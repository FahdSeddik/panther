#include <torch/extension.h>

template <typename scalar_t>
__global__ void sketched_linear_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> S1s,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> S2s,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> U1s,
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> U2s,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> bias,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    int batch_size, int input_dim, int num_terms, int low_rank_dim, int output_dim) {
    // Grid is organized so that each block computes one output element:
    //   blockIdx.x -> batch index, blockIdx.y -> output index.
    int b = blockIdx.x;  // batch index
    int o = blockIdx.y;  // output dimension index

    // We use a 1D block to parallelize over the large input_dim reduction.
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;

    // We'll accumulate the final result in a register (per block, only thread 0 will write out).
    // Each term is computed by first “reducing” the product over the input_dim for every low_rank index.
    // Then, the resulting low_rank-dim vector is dot-multiplied with the corresponding second factor.
    // We accumulate the two terms over all num_terms.
    scalar_t result = 0;

    // Temporary shared memory buffer used to reduce partial dot products.
    extern __shared__ float shared_data[];
    scalar_t* sharedDataCast = (scalar_t*)shared_data;
    int numWarps = (blockSize + warpSize - 1) / warpSize;
    scalar_t* reduction1 = sharedDataCast;
    scalar_t* reduction2 = sharedDataCast + numWarps;

    // Allocate shared memory for U1s and S2s for the current term.
    scalar_t* U1s_shared = reduction2 + numWarps;
    scalar_t* S2s_shared = U1s_shared + low_rank_dim * num_terms;

    // Each thread loads one element of the U1s and S2s matrices into shared memory.
    for (int idx = tid; idx < low_rank_dim * num_terms; idx += blockSize) {
        int k = idx / num_terms;
        int term = idx % num_terms;
        U1s_shared[idx] = U1s[term][k][o];
        S2s_shared[idx] = S2s[term][k][o];
    }

    __syncthreads();

    // Loop over each term (num_terms is very small)
    for (int term = 0; term < num_terms; term++) {
        // Loop over the low_rank (hidden) dimension.
        for (int k = 0; k < low_rank_dim; k++) {
            // Each thread computes a partial sum for the dot product over the input dimension.
            scalar_t partial1 = 0;  // for the first branch (input * S1s)
            scalar_t partial2 = 0;  // for the second branch (input * U2s)

            // Tile the reduction over j (input dimension)
            for (int i = tid; i < input_dim; i += blockSize) {
                scalar_t in_val = input[b][i];
                // multiply by S1s and U2s (each of shape [input_dim, low_rank_dim])
                partial1 += in_val * S1s[term][i][k];
                partial2 += in_val * U2s[term][i][k];
            }

            // Reduce within the warp (32 threads).
            for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                partial1 += __shfl_down_sync(0xFFFFFFFF, partial1, offset);
                partial2 += __shfl_down_sync(0xFFFFFFFF, partial2, offset);
            }
            __syncthreads();
            // Store the reduced values in shared memory.
            if (tid % warpSize == 0) {
                reduction1[tid / warpSize] = partial1;
                reduction2[tid / warpSize] = partial2;
            }
            __syncthreads();

            if (tid < numWarps) {
                // Reduce the values in shared memory across all warps.
                scalar_t inter1 = reduction1[tid];
                scalar_t inter2 = reduction2[tid];
                for (int offset = numWarps / 2; offset > 0; offset >>= 1) {
                    inter1 += __shfl_down_sync(0xFFFFFFFF, inter1, offset);
                    inter2 += __shfl_down_sync(0xFFFFFFFF, inter2, offset);
                }

                // Only one thread (tid==0) accumulates the contributions from this low_rank index.
                if (tid == 0) {
                    // First term: (input * S1s) dot U1s, second term: (input * U2s) dot S2s.
                    int idx = k * num_terms + term;
                    result += inter1 * U1s_shared[idx] + inter2 * S2s_shared[idx];
                }
            }
        }
    }

    // Write out the result (only one thread writes)
    if (tid == 0) {
        // Scale by 1/(2*num_terms) and add the bias.
        output[b][o] = bias[o] + result / (static_cast<scalar_t>(2 * num_terms));
    }
}

int64_t calcSharedMem(int64_t elementSize, int64_t blockSize, int64_t low_rank_dim, int64_t num_terms, int warpSize) {
    // Calculate the required shared memory size based on the block size and dimensions.
    int64_t numWarps = (blockSize + warpSize - 1) / warpSize;  // Number of warps in the block.
    return (numWarps * 2 + low_rank_dim * num_terms * 2) * elementSize;
}

int findSuitableBlockSize(const cudaDeviceProp& prop, int64_t elementSize, int64_t low_rank_dim, int64_t num_terms) {
    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    int regsPerSM = prop.regsPerMultiprocessor;
    int sharedMemPerSM = prop.sharedMemPerMultiprocessor;
    int sharedMemPerBlock = prop.sharedMemPerBlock;
    int maxBlocksPerSM = prop.maxBlocksPerMultiProcessor;

    const int desiredBlocksPerSM = 2;
    // This value has been retrieved through "--ptxas-options=-v"
    // and is specific to the kernel and device but its a good starting point.
    const int registersPerThread = 45;

    // Candidate selection variables.
    int idealBlockSize = 32;
    int minDiff = maxThreadsPerSM;  // Initialize with the maximum possible difference.
    int bestTotalThreads = 0;

    // Evaluate candidate block sizes (from 32 up to maxThreadsPerBlock, stepping by 32).
    for (int blockSize = 32; blockSize <= std::min(maxThreadsPerBlock, 1024); blockSize *= 2) {
        // 1. Calculate register usage and how many blocks can be resident by registers.
        int registersUsedPerBlock = blockSize * registersPerThread;
        int blocksByRegs = (registersUsedPerBlock > 0) ? (regsPerSM / registersUsedPerBlock) : 0;

        // 2. Calculate shared memory usage and how many blocks can be resident by shared memory.
        int sharedMemUsedPerBlock = calcSharedMem(elementSize, blockSize, low_rank_dim, num_terms, prop.warpSize);
        // Check if the shared memory usage exceeds the maximum allowed per block.
        if (sharedMemUsedPerBlock > sharedMemPerBlock) {
            continue;  // Skip this block size if it exceeds the limit.
        }
        int blocksBySharedMem = (sharedMemUsedPerBlock > 0) ? (sharedMemPerSM / sharedMemUsedPerBlock) : 0;

        // 3. Derived hardware limit: the number of blocks is also limited by the maximum threads per SM.
        int blocksByThreads = maxThreadsPerSM / blockSize;

        // The resident blocks for this candidate is the minimum of all constraints.
        int residentBlocks = std::min({blocksByRegs, blocksBySharedMem, maxBlocksPerSM, blocksByThreads});

        // Enforce the minimum desired resident blocks per SM.
        if (residentBlocks < desiredBlocksPerSM) {
            continue;
        }

        // Total resident threads for this candidate.
        int totalThreads = residentBlocks * blockSize;
        // Calculate the difference from the maximum threads per SM.
        int diff = maxThreadsPerSM - totalThreads;

        // Selection criteria:
        // Prefer the candidate that minimizes the difference.
        // If two candidates are similar (same difference), prefer the one with a higher blockSize.
        if (diff <= minDiff) {
            minDiff = diff;
            idealBlockSize = blockSize;
            bestTotalThreads = totalThreads;
        }
    }

    std::cout << "Ideal Block Size (1D): " << idealBlockSize << std::endl;
    std::cout << "Resident Threads per SM: " << bestTotalThreads << " (out of " << maxThreadsPerSM << " available)" << std::endl;

    return idealBlockSize;
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

    // Query current GPU properties.
    int device = input.device().index();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    // int chosenBlockSize = findSuitableBlockSize(prop, input.element_size(), low_rank_dim, num_terms);
    int chosenBlockSize = 256;

    // Define grid dimensions: one block per (batch, output) pair.
    dim3 grid(batch_size, output_dim);
    dim3 block(chosenBlockSize);

    // Allocate shared memory: one float per thread.
    size_t shared_mem_size = calcSharedMem(input.element_size(), chosenBlockSize, low_rank_dim, num_terms, prop.warpSize);

    // Launch the kernel using AT_DISPATCH_FLOATING_TYPES to handle different data types.
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "sketched_linear_forward_cuda",
        ([&] {
            sketched_linear_forward_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
                input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                S1s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                S2s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                U1s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                U2s.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                batch_size, input_dim, num_terms, low_rank_dim, output_dim);
        }));

    return output;
}