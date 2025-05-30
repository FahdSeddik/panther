#include "cqrrpt.h"

extern "C" {
#include <lapacke_config.h>
}

extern "C" {
#include <lapacke.h>
}

extern "C" {
#include <lapack.h>
}

extern "C" {
#include <cblas.h>
}

#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

template <typename T>
void qrcp_impl(int64_t d, int64_t n, T* M_sk_data, int64_t lda, std::vector<lapack_int>& jpvt, std::vector<T>& tau);

/**
 * @brief Performs QR factorization with column pivoting for a matrix of type float.
 *
 * This template specialization of qrcp_impl uses LAPACKE_sgeqp3 to compute the QR decomposition
 * with column pivoting of a matrix stored in row-major order.
 *
 * @param d         The number of rows of the matrix.
 * @param n         The number of columns of the matrix.
 * @param M_sk_data Pointer to the matrix data (row-major order).
 * @param lda       Leading dimension of the matrix (typically equal to n).
 * @param jpvt      Vector that will contain the pivot indices.
 * @param tau       Vector that will contain the scalar factors of the elementary reflectors.
 */
template <>
void qrcp_impl<float>(int64_t d, int64_t n, float* M_sk_data, int64_t lda,
                      std::vector<lapack_int>& jpvt, std::vector<float>& tau) {
    LAPACKE_sgeqp3(LAPACK_ROW_MAJOR, d, n, M_sk_data, lda, jpvt.data(), tau.data());
}

/**
 * @brief Performs QR factorization with column pivoting for a matrix of type double.
 *
 * This template specialization of qrcp_impl uses LAPACKE_dgeqp3 to compute the QR decomposition
 * with column pivoting of a real matrix. The results are stored in-place in the input matrix and
 * the provided vectors for pivot indices and scalar factors of the elementary reflectors.
 *
 * @param d         The number of rows of the matrix M_sk_data.
 * @param n         The number of columns of the matrix M_sk_data.
 * @param M_sk_data Pointer to the matrix data stored in row-major order.
 * @param lda       The leading dimension of the matrix M_sk_data.
 * @param jpvt      Reference to a vector that will contain the pivot indices.
 * @param tau       Reference to a vector that will contain the scalar factors of the elementary reflectors.
 */
template <>
void qrcp_impl<double>(int64_t d, int64_t n, double* M_sk_data, int64_t lda,
                       std::vector<lapack_int>& jpvt, std::vector<double>& tau) {
    LAPACKE_dgeqp3(LAPACK_ROW_MAJOR, d, n, M_sk_data, lda, jpvt.data(), tau.data());
}

/**
 * @brief Computes a column-QR with randomized projection and pivoted QR (CQRRPT) decomposition.
 *
 * This function performs a randomized sketch of the input matrix M using the sketching matrix S,
 * followed by a QR decomposition with column pivoting (QRCP) on the sketched matrix. It then
 * estimates the numerical rank, refines it using a Cholesky-based criterion, and computes the
 * final Q and R factors along with the column permutation indices.
 *
 * @param M [Tensor] The input matrix of shape (m, n), must be 2D and of type float32 or float64.
 * @param S [Tensor] The sketching matrix of shape (d, m), must be 2D and of the same dtype as M.
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
 *         - Qk: Orthonormal matrix of shape (m, new_rank) after rank refinement.
 *         - Rk: Upper triangular matrix of shape (new_rank, n).
 *         - J: 1D tensor of shape (n,) containing the column permutation indices.
 *
 * @throws torch::Error if input validation fails or rank deficiency is detected.
 *
 * @details
 * Steps performed:
 *  1. Validates input dimensions and types.
 *  2. Computes a randomized sketch of M via S * M.
 *  3. Performs QR decomposition with column pivoting (QRCP) on the sketched matrix.
 *  4. Extracts the R factor and permutation indices.
 *  5. Estimates the rank using a tolerance based on the data type.
 *  6. Selects leading columns and solves a triangular system.
 *  7. Computes a Cholesky factorization for further rank refinement.
 *  8. Refines the rank based on the Cholesky diagonal.
 *  9. Computes the final Q and R matrices for the refined rank.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cqrrpt_core(
    const torch::Tensor& M, const torch::Tensor& S) {
    // Validate inputs
    TORCH_CHECK(M.dim() == 2, "M must be 2D tensor");
    TORCH_CHECK(S.dim() == 2, "S must be 2D tensor");
    TORCH_CHECK(M.scalar_type() == S.scalar_type(), "M and S must have same data type");

    const auto dtype = M.scalar_type();
    TORCH_CHECK(dtype == torch::kFloat32 || dtype == torch::kFloat64,
                "Only float32 and float64 supported");

    // Single contiguous copy if needed
    auto M_contig = M.contiguous();

    // Compute sketch on original device
    auto M_sk = torch::mm(S, M_contig);
    const int64_t d = M_sk.size(0);
    const int64_t n = M_sk.size(1);
    const int64_t min_dn = std::min(d, n);

    // Move to CPU for LAPACK QRCP
    auto M_sk_cpu = M_sk.to(torch::kCPU).contiguous();
    std::vector<lapack_int> jpvt(n, 0);
    torch::Tensor tau;

    // Type-specific QRCP
    if (dtype == torch::kFloat32) {
        std::vector<float> tau_vec(min_dn);
        qrcp_impl<float>(d, n, M_sk_cpu.data_ptr<float>(), n, jpvt, tau_vec);
        tau = torch::from_blob(tau_vec.data(), {min_dn}, torch::kFloat32).clone();
    } else {
        std::vector<double> tau_vec(min_dn);
        qrcp_impl<double>(d, n, M_sk_cpu.data_ptr<double>(), n, jpvt, tau_vec);
        tau = torch::from_blob(tau_vec.data(), {min_dn}, torch::kFloat64).clone();
    }

    // Extract R factor and permutation
    auto R_sk = M_sk_cpu.triu().to(M.device());
    std::vector<int64_t> J_vec(n);
    for (int64_t i = 0; i < n; ++i)
        J_vec[i] = jpvt[i] - 1;
    auto J = torch::tensor(J_vec, M.options().dtype(torch::kInt64));

    // Rank estimation with type-specific tolerance
    const auto eps = dtype == torch::kFloat32 ? 1e-6f : 1e-12;
    auto diag = R_sk.diag().abs();
    int64_t k = (diag > eps).sum().item().toLong();
    TORCH_CHECK(k > 0, "Rank deficiency detected");

    // Column selection and triangular solve
    auto Mk = M_contig.index({torch::indexing::Slice(), J.index({torch::indexing::Slice(0, k)})});
    auto Ak_sk = R_sk.index({torch::indexing::Slice(0, k),
                             torch::indexing::Slice(0, k)});

    auto M_pre = torch::linalg_solve_triangular(Ak_sk, Mk, /*upper=*/true,
                                                /*left=*/false, /*unitriangular=*/false);

    // Cholesky factorization
    auto G = torch::matmul(M_pre.t(), M_pre);
    auto L = torch::linalg_cholesky(G, /*upper=*/false);

    // Rank refinement
    auto L_diag = L.diag().abs();
    const auto u = dtype == torch::kFloat32 ? std::numeric_limits<float>::epsilon() : std::numeric_limits<double>::epsilon();
    const auto threshold = std::sqrt(eps / u);

    double running_max = L_diag[0].item().toDouble();
    double running_min = running_max;
    int64_t new_rank = k;

    for (int64_t i = 0; i < k; ++i) {
        const auto curr = L_diag[i].item().toDouble();
        running_max = std::max(running_max, curr);
        running_min = std::min(running_min, curr);
        if (running_max / running_min >= threshold) {
            new_rank = i;
            break;
        }
    }
    new_rank = std::max(new_rank, int64_t(1));

    // Final Q and R computation
    auto L_T = L.transpose(0, 1).index({torch::indexing::Slice(0, new_rank),
                                        torch::indexing::Slice(0, new_rank)});
    auto Qk = torch::linalg_solve_triangular(L_T,
                                             M_pre.index({torch::indexing::Slice(), torch::indexing::Slice(0, new_rank)}),
                                             /*upper=*/true, /*left=*/false, /*unitriangular=*/false);

    auto Rk = torch::matmul(L_T, R_sk.index({torch::indexing::Slice(0, new_rank),
                                             torch::indexing::Slice()}));

    return {Qk.contiguous(), Rk.contiguous(), J};
}

/**
 * @brief Computes a Cholesky QR decomposition with randomized projection.
 *
 * This function applies a sketching operator to the input matrix M based on the specified
 * distribution family, and then performs a core CQRRPT (Cholesky QR with Random Projection and Truncation)
 * operation. Currently, only the Gaussian distribution family is supported for the sketching operator.
 *
 * @param M The input matrix as a torch::Tensor of shape (m, n).
 * @param gamma The compression ratio, used to determine the sketch size (d = ceil(gamma * n)).
 * @param F The distribution family for the sketching operator (only DistributionFamily::Gaussian is supported).
 * @return A tuple of three torch::Tensor objects resulting from the CQRRPT core operation.
 *
 * @throws torch::TorchCheckError if an unsupported distribution family is provided.
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cqrrpt(const torch::Tensor& M, double gamma, const DistributionFamily& F) {
    const int64_t n = M.size(1);
    const int64_t m = M.size(0);
    const int64_t d = static_cast<int64_t>(std::ceil(gamma * n));
    const auto opts = M.options().dtype(M.scalar_type());

    torch::Tensor S;
    if (F == DistributionFamily::Gaussian) {
        S = sparse_sketch_operator(d, m, 4, Axis::Short, opts.device(), M.scalar_type());
    } else {
        TORCH_CHECK(false, "Unsupported distribution family");
    }

    return cqrrpt_core(M, S);
}