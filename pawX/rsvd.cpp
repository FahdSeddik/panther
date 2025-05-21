#include "rsvd.h"

#include <algorithm>
#include <cmath>
#include <tuple>

// Helper: QR-based orthonormalization
static std::tuple<torch::Tensor, torch::Tensor> orthonormalize(const torch::Tensor& X) {
    return torch::linalg_qr(X, /*mode=*/"reduced");
}

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
torch::Tensor rangeFinder(const torch::Tensor& A, int64_t k, int64_t power_iters = 2) {
    // Sketch
    auto Omega = powerSketch(A, k, power_iters);
    // Form sample matrix Y = A * Omega
    auto Y = torch::matmul(A, Omega);
    return std::get<0>(orthonormalize(Y));
}

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
