#include "rsvd.h"

#include <algorithm>
#include <cmath>
#include <tuple>

// Helper: QR-based orthonormalization
/**
 * @brief Orthonormalizes the input tensor using QR decomposition.
 *
 * This function computes the reduced QR decomposition of the input tensor X,
 * returning a tuple containing the orthonormal matrix Q and the upper triangular matrix R.
 *
 * @param X The input tensor to be orthonormalized.
 * @return A std::tuple containing:
 *         - torch::Tensor Q: The orthonormal matrix.
 *         - torch::Tensor R: The upper triangular matrix.
 */
static std::tuple<torch::Tensor, torch::Tensor> orthonormalize(const torch::Tensor& X) {
    return torch::linalg_qr(X, /*mode=*/"reduced");
}

/**
 * @brief Computes a randomized sketch of the input matrix using the power iteration method.
 *
 * This function generates a sketch matrix by repeatedly multiplying the input matrix (and its transpose)
 * with a random Gaussian matrix, optionally orthonormalizing the result at specified intervals for numerical stability.
 * The process is controlled by the number of passes and stabilization frequency.
 *
 * @param A         The input matrix (torch::Tensor) of shape (m, n).
 * @param k         The target sketch dimension (number of columns in the sketch).
 * @param passes    The number of power iteration passes to perform.
 * @param stab_freq The frequency (in iterations) at which to orthonormalize the sketch for stability (default: 1).
 * @return torch::Tensor The resulting sketch matrix (Omega) after the specified number of passes.
 */
torch::Tensor powerSketch(const torch::Tensor& A, int64_t k, int64_t passes, int64_t stab_freq = 1) {
    auto m = A.size(0);
    auto n = A.size(1);
    // initialize Omega
    torch::Tensor Omega = torch::randn({n, k}, A.options());
    torch::Tensor Y;
    for (int64_t i = 0; i < passes; ++i) {
        if (i % 2 == 0) {
            // Y = A * Omega
            Y = torch::matmul(A, Omega);
        } else {
            // Y = A^T * Omega
            Y = torch::matmul(A.transpose(0, 1), Omega);
        }
        Omega = Y;
        // optional stabilization every stab_freq iterations
        if (stab_freq > 0 && ((i + 1) % stab_freq) == 0) {
            Omega = std::get<0>(orthonormalize(Omega));
        }
    }
    return Omega;
}

// Returns Q with orthonormal columns spanning A's approximate range
/**
 * @brief Computes an approximate orthonormal basis for the range of matrix A using randomized projections.
 *
 * This function uses a randomized range finder algorithm to efficiently approximate the column space of the input matrix A.
 * It first generates a sketch matrix using power iterations, then forms a sample matrix by multiplying A with the sketch,
 * and finally orthonormalizes the result to produce the basis.
 *
 * @param A            The input matrix (torch::Tensor) whose range is to be approximated.
 * @param k            Target rank or number of basis vectors to compute.
 * @param power_iters  Number of power iterations to improve approximation accuracy (default: 2).
 * @return             An orthonormal basis (torch::Tensor) for the approximate range of A.
 */
torch::Tensor rangeFinder(const torch::Tensor& A, int64_t k, int64_t power_iters = 2) {
    // Sketch
    auto Omega = powerSketch(A, k, power_iters);
    // Form sample matrix Y = A * Omega
    auto Y = torch::matmul(A, Omega);
    return std::get<0>(orthonormalize(Y));
}

/**
 * @brief Computes a blocked randomized QB factorization of a matrix.
 *
 * This function implements a blocked version of the randomized QB decomposition,
 * which approximates the input matrix A as A ≈ Q * B, where Q has orthonormal columns
 * and B is a smaller matrix. The algorithm proceeds in blocks, adaptively building
 * up the approximation until the specified rank k or error tolerance is reached.
 *
 * @param A           The input matrix (torch::Tensor) to be factorized, of shape (m, n).
 * @param k           Target rank for the approximation.
 * @param block_size  Size of each block to use in the iterative process (default: 64).
 * @param tol         Error tolerance for the approximation (default: 1e-6).
 * @return            A tuple (Q, B) where:
 *                      - Q: Orthonormal basis matrix (torch::Tensor) of shape (m, <=k).
 *                      - B: Coefficient matrix (torch::Tensor) of shape (<=k, n).
 *
 * @note The function uses a range finder (rangeFinder) and orthonormalization (orthonormalize)
 *       routines, which must be defined elsewhere. The process terminates early if the
 *       approximation error stops decreasing or falls below the specified tolerance.
 */
std::tuple<torch::Tensor, torch::Tensor> blockedQB(
    const torch::Tensor& A,
    int64_t k,
    int64_t block_size = 64,
    double tol = 1e-6) {
    auto m = A.size(0);
    auto n = A.size(1);
    double eps = 100 * std::numeric_limits<double>::epsilon();
    tol = std::max(tol, eps);

    torch::Tensor A_res = A.clone();
    torch::Tensor Q = torch::empty({m, 0}, A.options());
    torch::Tensor B;
    int64_t curr = 0;

    double normA = A.norm().item<double>();
    double normB = 0;
    double approx_err = 0, prev_err = 0;

    while (curr < k) {
        int64_t bs = std::min(block_size, k - curr);
        // RangeFinder on residual
        auto Q_i = rangeFinder(A_res, bs);
        // Re-orthogonalize against Q
        if (curr > 0) {
            auto proj = torch::matmul(Q.transpose(0, 1), Q_i);
            Q_i = Q_i - torch::matmul(Q, proj);
            Q_i = std::get<0>(orthonormalize(Q_i));
        }
        // Compute B_i = Q_i^T * A_res
        auto B_i = torch::matmul(Q_i.transpose(0, 1), A_res);
        // Update Q, B
        Q = torch::cat({Q, Q_i}, 1);
        B = (curr == 0 ? B_i : torch::cat({B, B_i}, 0));
        // Update error
        double normBi = B_i.norm().item<double>();
        normB = std::hypot(normB, normBi);
        prev_err = approx_err;
        approx_err = std::sqrt(std::abs(normA * normA - normB * normB)) * (std::sqrt(normA * normA + normB * normB) / normA);
        // Check termination
        if (curr > 0 && approx_err > prev_err) {
            break;  // error growing
        }
        if (approx_err < tol) {
            curr += bs;
            break;
        }
        // Update residual A_res = A_res - Q_i * B_i
        A_res = A_res - torch::matmul(Q_i, B_i);
        curr += bs;
    }
    return {Q, B};
}

/**
 * @brief Computes a randomized Singular Value Decomposition (SVD) of the input matrix.
 *
 * This function performs a randomized SVD on the given matrix `A`, approximating the top `k` singular values and vectors.
 * It first computes a blocked QB decomposition to obtain an orthonormal basis `Q` and a smaller matrix `B`.
 * Then, it performs a full SVD on `B` and reconstructs the approximate left singular vectors.
 *
 * @param A The input tensor (matrix) to decompose.
 * @param k The target rank (number of singular values/vectors to compute).
 * @param tol Tolerance for convergence in the blocked QB step (default: 1e-6).
 * @return A tuple containing:
 *         - U: The approximate left singular vectors (torch::Tensor).
 *         - S: The singular values (torch::Tensor).
 *         - V: The approximate right singular vectors (torch::Tensor, transposed).
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> randomized_svd(
    const torch::Tensor& A,
    int64_t k,
    double tol = 1e-6) {
    // Blocked QB
    auto [Q, B] = blockedQB(A, k, /*block_size=*/std::min<int64_t>(64, k), tol);
    // SVD on small B
    auto svd_result = torch::linalg_svd(B, /*full_matrices=*/false);
    auto U_tilde = std::get<0>(svd_result);
    auto S = std::get<1>(svd_result);
    auto Vt = std::get<2>(svd_result);
    // Compute U = Q * U_tilde
    auto U = torch::matmul(Q, U_tilde);
    return {U, S, Vt.transpose(0, 1)};
}
