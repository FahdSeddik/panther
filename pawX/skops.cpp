#include "skops.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <vector>

torch::Tensor scaled_sign_sketch(int64_t m, int64_t n, c10::optional<torch::Device> device, c10::optional<torch::Dtype> dtype) {
    // Set tensor options
    auto options = torch::TensorOptions().device(device.value_or(torch::kCPU)).dtype(dtype.value_or(torch::kFloat));

    // Generate {-1, 1} directly using sign function and scale by precomputed factor
    return torch::sign(torch::rand({m, n}, options) - 0.5) * (1.0 / std::sqrt(static_cast<double>(m)));
}

torch::Tensor dense_sketch_operator(int64_t m, int64_t n, DistributionFamily distribution, c10::optional<torch::Device> device, c10::optional<torch::Dtype> dtype) {
    // Set tensor options
    auto options = torch::TensorOptions().device(device.value_or(torch::kCPU)).dtype(dtype.value_or(torch::kFloat));

    // Generate the sketch matrix based on the specified distribution
    if (distribution == DistributionFamily::Gaussian) {
        return torch::randn({m, n}, options);
    } else if (distribution == DistributionFamily::Uniform) {
        return (torch::rand({m, n}, options) - 0.5) * std::sqrt(static_cast<double>(12.0));
    } else {
        throw std::invalid_argument("Unsupported distribution family");
    }
}

// The sketch_tensor function:
// 1. Validates the axis and new_size constraints.
// 2. Generates a sketching operator S of size (new_size x original_size) along the chosen axis.
// 3. Permutes the input tensor so that the axis to sketch becomes the first dimension.
// 4. Flattens the remaining dimensions, applies S * input, and then reshapes and permutes back.
std::tuple<torch::Tensor, torch::Tensor> sketch_tensor(const torch::Tensor &input,
                                                       int64_t axis,
                                                       int64_t new_size,
                                                       DistributionFamily distribution,
                                                       c10::optional<torch::Device> device,
                                                       c10::optional<torch::Dtype> dtype) {
    // Check that axis is valid.
    TORCH_CHECK(axis >= 0 && axis < input.dim(), "Axis index out of bounds");

    // Get size along the axis.
    int64_t original_size = input.size(axis);
    TORCH_CHECK(new_size > 0 && new_size <= original_size,
                "new_size must be > 0 and <= the size of the dimension to sketch");

    // Create the sketching operator: dimensions (new_size x original_size)
    torch::Tensor sketch_matrix = dense_sketch_operator(new_size, original_size, distribution, device, dtype);

    return std::make_tuple(sketch_tensor(input, axis, new_size, sketch_matrix, device, dtype), sketch_matrix);
}

torch::Tensor sketch_tensor(const torch::Tensor &input,
                            int64_t axis,
                            int64_t new_size,
                            const torch::Tensor &sketch_matrix,
                            c10::optional<torch::Device> device,
                            c10::optional<torch::Dtype> dtype) {
    // Check that axis is valid.
    TORCH_CHECK(axis >= 0 && axis < input.dim(), "Axis index out of bounds");

    // Get size along the axis.
    int64_t original_size = input.size(axis);
    TORCH_CHECK(new_size > 0 && new_size <= original_size,
                "new_size must be > 0 and <= the size of the dimension to sketch");

    // Build permutation vector to bring the axis to be sketched to the front.
    std::vector<int64_t> perm;
    perm.push_back(axis);
    for (int64_t i = 0; i < input.dim(); i++) {
        if (i != axis) {
            perm.push_back(i);
        }
    }
    // Permute the input tensor accordingly.
    torch::Tensor input_permuted = input.permute(perm);

    // Collapse all dimensions except the first into a single dimension.
    torch::Tensor reshaped = input_permuted.reshape({original_size, -1});

    // Apply the sketching operator.
    // S is (new_size x original_size) and input reshaped is (original_size x rest)
    torch::Tensor output_flat = torch::matmul(sketch_matrix, reshaped);

    // Determine the new shape:
    // Replace the sketched axis with new_size while keeping all other dimensions the same.
    std::vector<int64_t> rest_dims;
    for (int64_t i = 0; i < input.dim(); i++) {
        if (i != axis) {
            rest_dims.push_back(input.size(i));
        }
    }
    std::vector<int64_t> new_shape;
    new_shape.push_back(new_size);
    new_shape.insert(new_shape.end(), rest_dims.begin(), rest_dims.end());

    // Reshape the result back to the permuted shape.
    torch::Tensor output_permuted = output_flat.reshape(new_shape);

    // Compute inverse permutation to restore original ordering.
    std::vector<int64_t> inv_perm(perm.size());
    for (size_t i = 0; i < perm.size(); i++) {
        inv_perm[perm[i]] = i;
    }
    torch::Tensor output = output_permuted.permute(inv_perm);

    return output;
}

/**
 * @brief Generates random sparse matrix entries using repeated Fisher–Yates shuffling.
 *
 * This function fills the provided `rows`, `cols`, and `vals` vectors with `vec_nnz` nonzero entries
 * per minor dimension, for a matrix of size (`dim_major` x `dim_minor`). For each minor index,
 * it selects `vec_nnz` unique major indices using the Fisher–Yates shuffle, and assigns each entry
 * a random Rademacher value (+1 or -1).
 *
 * @param vec_nnz        Number of nonzero entries to generate per minor index.
 * @param dim_major      Size of the major dimension (number of rows or columns, depending on layout).
 * @param dim_minor      Size of the minor dimension (number of columns or rows, depending on layout).
 * @param rows_are_major If true, rows are the major dimension; otherwise, columns are major.
 * @param rows           Output vector to store row indices of nonzero entries.
 * @param cols           Output vector to store column indices of nonzero entries.
 * @param vals           Output vector to store values (+1 or -1) of nonzero entries.
 */
static void repeated_fisher_yates(
    int64_t vec_nnz,
    int64_t dim_major,
    int64_t dim_minor,
    bool rows_are_major,
    std::vector<int64_t> &rows,
    std::vector<int64_t> &cols,
    std::vector<float> &vals) {
    std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<int> coin_flip(0, 1);

    // Temporary working array for Fisher–Yates:
    std::vector<int64_t> vec_work(dim_major);

    for (int64_t i = 0; i < dim_minor; ++i) {
        // 1) Initialize vec_work = [0, 1, 2, ..., (dim_major-1)]
        for (int64_t j = 0; j < dim_major; ++j) {
            vec_work[j] = j;
        }

        // 2) Perform the first vec_nnz steps of Fisher–Yates:
        for (int64_t j = 0; j < vec_nnz; ++j) {
            // pick l uniformly from [j .. dim_major-1]
            std::uniform_int_distribution<int64_t> dist_ell(j, dim_major - 1);
            int64_t l = dist_ell(rng);

            // swap vec_work[l] <--> vec_work[j]
            int64_t chosen = vec_work[l];
            vec_work[l] = vec_work[j];
            vec_work[j] = chosen;

            // Build a Rademacher ±1
            float val = (coin_flip(rng) == 0 ? +1.0f : -1.0f);

            // Map "major_idx = chosen", "minor_idx = i" -> (row, col)
            if (rows_are_major) {
                rows.push_back(chosen);
                cols.push_back(i);
            } else {
                rows.push_back(i);
                cols.push_back(chosen);
            }
            vals.push_back(val);
        }
    }
}

/**
 * @brief Samples indices and values for a sparse vector with i.i.d. uniform distribution.
 *
 * This function fills the provided index and value vectors with randomly sampled indices
 * and Rademacher ±1 values. Indices are sampled uniformly at random from the range [0, dim_major),
 * and values are assigned randomly as +1 or -1. Duplicate indices may occur; the actual
 * merging and value assignment for duplicates is handled in a later step
 * (laso_merge_long_axis_vector_coo_data).
 *
 * @param dim_major The upper bound (exclusive) for index sampling.
 * @param vec_nnz The number of non-zero elements to sample.
 * @param idxs_lax_l Output vector to store sampled indices (resized to vec_nnz).
 * @param vals_l Output vector to store sampled values (resized to vec_nnz).
 * @param rng Random number generator to use for sampling.
 * @param uni_major Uniform integer distribution for index sampling.
 * @param coin_flip Uniform integer distribution for Rademacher ±1 value assignment.
 */
static void sample_indices_iid_uniform(
    int64_t dim_major,
    int64_t vec_nnz,
    std::vector<int64_t> &idxs_lax_l,
    std::vector<float> &vals_l,
    std::mt19937_64 &rng,
    std::uniform_int_distribution<int64_t> &uni_major,
    std::uniform_int_distribution<int> &coin_flip) {
    idxs_lax_l.resize(vec_nnz);
    vals_l.resize(vec_nnz);

    for (int64_t j = 0; j < vec_nnz; ++j) {
        int64_t l = uni_major(rng);
        idxs_lax_l[j] = l;
        // assign a Rademacher ±1 if this is the first time l appears,
        // but since we don't know duplicates yet, just store a placeholder
        // (we'll actually merge+reassign in laso_merge_long_axis_vector_coo_data).
        //
        // For simplicity, store a random ±1 at each position; merging step
        // will re‐compute sqrt(count) * sign_of_first_appearance.
        vals_l[j] = (coin_flip(rng) == 0 ? +1.0f : -1.0f);
    }
}

/**
 * @brief Merges a sparse vector into COO (Coordinate) format along a long axis, aggregating duplicate indices.
 *
 * For each unique index in the input vector, this function counts the number of occurrences and records the sign
 * (value) from its first appearance. It then computes the final value for each unique index as:
 *     final_value = (sign from first appearance) * sqrt(count of occurrences)
 * and appends a single triplet (row, col, value) to the output COO arrays.
 *
 * @param vec_nnz        Number of non-zero elements in the input vector.
 * @param idxs_lax_l     Indices of the non-zero elements along the long axis.
 * @param vals_l         Values corresponding to the indices in idxs_lax_l.
 * @param i              The fixed index along the short axis (row or column, depending on rows_are_major).
 * @param rows_are_major If true, output triplets as (row=l, col=i); otherwise, as (row=i, col=l).
 * @param rows           Output vector to append row indices of the resulting COO triplets.
 * @param cols           Output vector to append column indices of the resulting COO triplets.
 * @param vals           Output vector to append values of the resulting COO triplets.
 */
static void laso_merge_long_axis_vector_coo_data(
    int64_t vec_nnz,
    const std::vector<int64_t> &idxs_lax_l,
    const std::vector<float> &vals_l,
    int64_t i,
    bool rows_are_major,
    std::vector<int64_t> &rows,
    std::vector<int64_t> &cols,
    std::vector<float> &vals) {
    // loc2count[l] = number of times l appeared
    // loc2sign[l]  = the ±1 assigned the *first* time l appeared
    std::unordered_map<int64_t, int64_t> loc2count;
    std::unordered_map<int64_t, float> loc2sign;

    loc2count.reserve(vec_nnz);
    loc2sign.reserve(vec_nnz);

    // 1) Count occurrences, record sign on first appearance
    for (int64_t j = 0; j < vec_nnz; ++j) {
        int64_t l = idxs_lax_l[j];
        auto it = loc2count.find(l);
        if (it == loc2count.end()) {
            loc2count[l] = 1;
            loc2sign[l] = vals_l[j];  // first‐appearance sign
        } else {
            it->second += 1;
        }
    }

    // 2) For each unique l, compute final value = sign * sqrt(count), append a single triplet
    for (auto &kv : loc2count) {
        int64_t l = kv.first;
        int64_t c = kv.second;
        float magnitude = std::sqrt(static_cast<float>(c));
        float v = loc2sign[l] * magnitude;

        if (rows_are_major) {
            // l is the row index, i is the column
            rows.push_back(l);
            cols.push_back(i);
        } else {
            // l is the column index, i is the row
            rows.push_back(i);
            cols.push_back(l);
        }
        vals.push_back(v);
    }
}

/**
 * @brief Fills the provided row, column, and value vectors with the coordinates and values
 *        of a randomly generated sparse matrix in coordinate (COO) format.
 *
 * The sparsity pattern and value generation follow the conventions of RandBLAS's SparseDist.
 * The function supports two modes, determined by the `major_axis` parameter:
 *   - Axis::Short: The "short" axis (rows or columns, whichever is smaller) is treated as the major axis.
 *                  For each minor index, exactly `vec_nnz` nonzeros are generated using a repeated Fisher-Yates shuffle.
 *   - Axis::Long:  The "long" axis (rows or columns, whichever is larger) is treated as the major axis.
 *                  For each minor index, `vec_nnz` indices are sampled with replacement, then duplicates are merged.
 *
 * @param m         Number of rows in the matrix.
 * @param n         Number of columns in the matrix.
 * @param vec_nnz   Number of nonzeros per minor index (per row or column, depending on axis).
 * @param major_axis Specifies whether the major axis is the short or long axis (Axis::Short or Axis::Long).
 * @param rows      Output vector to be filled with the row indices of nonzero entries.
 * @param cols      Output vector to be filled with the column indices of nonzero entries.
 * @param vals      Output vector to be filled with the values of nonzero entries.
 *
 * @note The output vectors are reserved to accommodate the worst-case number of nonzeros.
 *       The actual number of nonzeros may be less than or equal to (dim_minor * vec_nnz)
 *       depending on the axis and duplicate merging.
 */
static void fill_sparse(
    int64_t m,
    int64_t n,
    int64_t vec_nnz,
    Axis major_axis,
    std::vector<int64_t> &rows,
    std::vector<int64_t> &cols,
    std::vector<float> &vals) {
    //
    //  1) Determine (dim_major, dim_minor) exactly as RandBLAS's SparseDist:
    //
    int64_t dim_major, dim_minor;
    bool rows_are_major;

    if (major_axis == Axis::Short) {
        if (m <= n) {
            dim_major = m;
            dim_minor = n;
            rows_are_major = true;  // "short axis" == rows
        } else {
            dim_major = n;
            dim_minor = m;
            rows_are_major = false;  // "short axis" == columns
        }
    } else {
        // major_axis == Axis::Long
        if (m >= n) {
            dim_major = m;
            dim_minor = n;
            rows_are_major = true;  // "long axis" == rows
        } else {
            dim_major = n;
            dim_minor = m;
            rows_are_major = false;  // "long axis" == columns
        }
    }

    // Reserve worst‐case: dim_minor * vec_nnz
    rows.reserve(dim_minor * vec_nnz);
    cols.reserve(dim_minor * vec_nnz);
    vals.reserve(dim_minor * vec_nnz);

    if (major_axis == Axis::Short) {
        // Short‐axis -> call repeated_fisher_yates
        repeated_fisher_yates(vec_nnz, dim_major, dim_minor, rows_are_major, rows, cols, vals);
        // -> exactly (dim_minor * vec_nnz) nonzeros
    } else {
        // Long‐axis -> for each minor index i, draw vec_nnz with replacement, then merge
        std::mt19937_64 rng{std::random_device{}()};
        std::uniform_int_distribution<int64_t> uni_major(0, dim_major - 1);
        std::uniform_int_distribution<int> coin_flip(0, 1);

        std::vector<int64_t> idxs_lax_l;
        std::vector<float> vals_l;

        idxs_lax_l.reserve(vec_nnz);
        vals_l.reserve(vec_nnz);

        for (int64_t i = 0; i < dim_minor; ++i) {
            // 2.1) Draw raw "long‐axis" indices (+ placeholder ±1)
            sample_indices_iid_uniform(
                dim_major,
                vec_nnz,
                idxs_lax_l,
                vals_l,
                rng,
                uni_major,
                coin_flip);

            // 2.2) Merge duplicates and append exactly one triplet per unique index
            laso_merge_long_axis_vector_coo_data(
                vec_nnz,
                idxs_lax_l,
                vals_l,
                i,
                rows_are_major,
                rows,
                cols,
                vals);
        }
        // -> final nnz <= (dim_minor * vec_nnz)
    }
}

/**
 * @brief Constructs a sparse sketch operator as a PyTorch sparse COO tensor.
 *
 * This function generates a sparse matrix of shape (m, n) with a specified number of non-zeros per vector (vec_nnz),
 * using the provided major axis and distribution family. The sparse matrix is constructed on the CPU and can be moved
 * to a specified device and data type.
 *
 * @param m Number of rows in the output sparse matrix.
 * @param n Number of columns in the output sparse matrix.
 * @param vec_nnz Number of non-zero entries per vector (row or column, depending on major_axis).
 * @param major_axis Axis along which non-zeros are distributed (e.g., row-major or column-major).
 * @param distribution Distribution family used to generate the non-zero values.
 * @param device (Optional) Target device for the output tensor (e.g., CPU or CUDA). Defaults to CPU if not specified.
 * @param dtype (Optional) Desired data type for the output tensor. Defaults to float32 if not specified.
 * @return torch::Tensor A sparse COO tensor representing the sketch operator, on the specified device and dtype.
 */
torch::Tensor sparse_sketch_operator(
    int64_t m,
    int64_t n,
    int64_t vec_nnz,
    Axis major_axis,
    c10::optional<torch::Device> device,
    c10::optional<torch::Dtype> dtype) {
    // 1) Gather all (row, col, value) triplets on CPU
    std::vector<int64_t> rows;
    std::vector<int64_t> cols;
    std::vector<float> vals;
    fill_sparse(m, n, vec_nnz, major_axis, rows, cols, vals);

    int64_t final_nnz = static_cast<int64_t>(vals.size());

    // 2) Build a [2 x final_nnz] int64 tensor for indices on CPU
    auto indices = torch::empty({2, final_nnz}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    auto acc = indices.accessor<int64_t, 2>();
    for (int64_t i = 0; i < final_nnz; ++i) {
        acc[0][i] = rows[i];
        acc[1][i] = cols[i];
    }

    // 3) Build a [final_nnz] float32 tensor for values on CPU, then cast if needed
    // clone so that PyTorch owns the data
    auto values = torch::from_blob(vals.data(), {final_nnz}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();

    torch::Dtype out_dtype = dtype.value_or(torch::kFloat32);
    torch::Device out_device = device.value_or(torch::kCPU);

    if (out_dtype != torch::kFloat32) {
        values = values.to(out_dtype);
    }

    // 4) Create a sparse COO tensor on CPU, then move to out_device
    auto sparse = torch::sparse_coo_tensor(indices, values, values.options().layout(torch::kSparse));

    if (out_device != torch::kCPU) {
        sparse = sparse.to(out_device);
    }
    return sparse;
}

/**
 * @brief Generates a Gaussian (normal) random matrix with scaled variance.
 *
 * This function creates a tensor of shape (m, d) filled with values drawn from a standard normal distribution,
 * and scales the result by 1/sqrt(m). The tensor is created on the specified device and with the specified data type,
 * if provided.
 *
 * @param m Number of rows in the output tensor.
 * @param d Number of columns in the output tensor.
 * @param device (Optional) The device on which to allocate the tensor (e.g., CPU or CUDA).
 * @param dtype (Optional) The desired data type of the returned tensor.
 * @return torch::Tensor A (m x d) tensor of scaled Gaussian random values.
 */
torch::Tensor gaussian_skop(int64_t m, int64_t d, c10::optional<torch::Device> device, c10::optional<torch::Dtype> dtype) {
    auto options = torch::TensorOptions().dtype(dtype.value_or(torch::kFloat32)).device(device.value_or(torch::kCPU));
    return torch::randn({m, d}, options) / std::sqrt(m);
}

/**
 * @brief Generates a sparse random matrix with ±1 entries in each column.
 *
 * Constructs an m x d matrix S, where for each column j, a single row index h[j] is randomly selected,
 * and S[h[j]][j] is set to either +1 or -1 (chosen randomly). All other entries are zero.
 *
 * @param m Number of rows in the output tensor.
 * @param d Number of columns in the output tensor.
 * @param device (Optional) The device on which to allocate the tensor (e.g., CPU or CUDA).
 * @param dtype (Optional) The desired data type of the returned tensor.
 * @return torch::Tensor The generated m x d sparse random matrix.
 */
torch::Tensor count_skop(int64_t m, int64_t d, c10::optional<torch::Device> device, c10::optional<torch::Dtype> dtype) {
    auto options = torch::TensorOptions().dtype(dtype.value_or(torch::kFloat32)).device(device.value_or(torch::kCPU));
    auto S = torch::zeros({m, d}, options);
    auto h = torch::randint(0, m, {d}, options.dtype(torch::kInt64));  // Row indices for each column
    auto signs = torch::bernoulli(torch::full({d}, 0.5)) * 2 - 1;      // ±1

    for (int64_t j = 0; j < d; ++j) {
        S[h[j]][j] = signs[j].item<float>();
    }
    return S;
}

/**
 * @brief Generates a Sparse Johnson-Lindenstrauss Transform (SJLT) matrix.
 *
 * This function creates an m x d sparse matrix suitable for dimensionality reduction
 * using the Johnson-Lindenstrauss lemma. Each column contains a specified number of non-zero
 * entries (given by `sparsity`), randomly positioned and assigned random signs (+1 or -1).
 * The matrix is normalized by dividing by sqrt(sparsity).
 *
 * @param m Number of rows in the output matrix.
 * @param d Number of columns in the output matrix.
 * @param sparsity Number of non-zero entries per column (default: 2).
 * @param device (Optional) The device on which to allocate the tensor (e.g., CPU or CUDA).
 * @param dtype (Optional) The data type of the tensor (e.g., torch::kFloat32).
 * @return torch::Tensor The generated SJLT matrix of shape (m, d).
 */
torch::Tensor sjlt_skop(int64_t m, int64_t d, int64_t sparsity, c10::optional<torch::Device> device, c10::optional<torch::Dtype> dtype) {
    auto options = torch::TensorOptions().dtype(dtype.value_or(torch::kFloat32)).device(device.value_or(torch::kCPU));
    auto S = torch::zeros({m, d}, options);

    for (int64_t j = 0; j < d; ++j) {
        auto rows = torch::randperm(m).slice(0, 0, sparsity);
        auto signs = torch::bernoulli(torch::full({sparsity}, 0.5)) * 2 - 1;

        for (int64_t i = 0; i < sparsity; ++i) {
            S[rows[i]][j] = signs[i].item<float>();
        }
    }
    return S / std::sqrt(sparsity);
}

/**
 * @brief Computes the Fast Walsh-Hadamard Transform (FWHT) of a 1D tensor.
 *
 * This function performs the in-place Fast Walsh-Hadamard Transform on the input tensor `x`.
 * The input tensor must be one-dimensional and its size must be a power of two.
 *
 * @param x A 1D torch::Tensor of size n (where n is a power of two) to be transformed.
 * @return torch::Tensor The transformed tensor after applying FWHT.
 *
 * @note The input tensor is modified in-place.
 */
torch::Tensor fwht(const torch::Tensor &x) {
    int n = x.size(0);
    int log_n = static_cast<int>(std::log2(n));
    for (int i = 0; i < log_n; ++i) {
        int step = 1 << i;
        for (int j = 0; j < n; j += 2 * step) {
            for (int k = 0; k < step; ++k) {
                auto u = x[j + k].clone();
                auto v = x[j + k + step].clone();
                x[j + k] = u + v;
                x[j + k + step] = u - v;
            }
        }
    }
    return x;
}

/**
 * @brief Applies the Subsampled Randomized Hadamard Transform (SRHT) to the input tensor.
 *
 * This function performs the following steps:
 *   1. Multiplies the input tensor by random ±1 signs.
 *   2. Applies the Fast Walsh-Hadamard Transform (FWHT) to the signed tensor and normalizes the result.
 *   3. Uniformly subsamples m rows from the transformed tensor.
 *
 * @param x Input 1D tensor of size n (must be a power of 2), on CPU, and convertible to float32.
 * @param m Number of rows to subsample (must satisfy m <= n).
 * @return torch::Tensor The resulting tensor of shape (m,) after SRHT.
 *
 * @throws c10::Error if input is not on CPU, input size is not a power of 2, or m > n.
 */
torch::Tensor srht(torch::Tensor x, int m) {
    // Ensure input is float32 and on CPU
    TORCH_CHECK(x.device().is_cpu(), "Input must be on CPU.");
    x = x.to(torch::kFloat32);

    int n = x.size(0);
    TORCH_CHECK((n & (n - 1)) == 0, "Input size must be power of 2.");  // Check power of 2
    TORCH_CHECK(m <= n, "Cannot subsample more rows than input size.");

    // Step 1: Multiply by random ±1 signs
    auto signs = torch::randint(0, 2, {n}, torch::kFloat32) * 2 - 1;  // ±1
    x = x * signs;

    // Step 2: Apply Hadamard transform
    x = fwht(x) / std::sqrt(static_cast<float>(n));  // Normalize

    // Step 3: Uniform subsample m rows
    auto indices = torch::randperm(n).slice(0, 0, m);
    return x.index_select(0, indices);
}