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

// cqrrpt_core implements Algorithm 2 from the manuscript.
// Input:  M (m x n) and S (d x m) as const torch::Tensor& objects (assumed double, CPU)
// Output: tuple (Q_k, R_k, J) where:
//         Q_k is m x k, R_k is k x n, and J is the length-n permutation vector.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cqrrpt_core(
    const torch::Tensor& M, const torch::Tensor& S) {
    // Create local copies ensuring double precision and contiguity.
    torch::Tensor M_local = M.to(torch::kDouble).contiguous();
    torch::Tensor S_local = S.to(torch::kDouble).contiguous();

    // Step 3: Compute the sketch M_sk = S_local * M_local.
    torch::Tensor M_sk = torch::mm(S_local, M_local);  // dimensions: d x n
    int64_t d = M_sk.size(0);
    int64_t n = M_sk.size(1);

    // Step 4: Compute a QR factorization with column pivoting on M_sk using LAPACKE_dgeqp3.
    M_sk = M_sk.contiguous();  // ensure contiguity
    double* M_sk_data = M_sk.data_ptr<double>();

    // jpvt: permutation indices (length n); initialize to zeros.
    std::vector<lapack_int> jpvt(n, 0);
    // tau: scalar factors for Householder reflectors, length = min(d, n)
    int min_dn = std::min((int)d, (int)n);
    std::vector<double> tau(min_dn);

    // Call LAPACKE_dgeqp3.
    int info = LAPACKE_dgeqp3(LAPACK_ROW_MAJOR, d, n, M_sk_data, n, jpvt.data(), tau.data());
    if (info != 0) {
        throw std::runtime_error("LAPACKE_dgeqp3 failed");
    }
    // After dgeqp3, the upper-triangular part of M_sk contains R_sk.
    torch::Tensor R_sk = M_sk.triu();

    // Step 6: Determine numerical rank k from R_sk’s diagonal.
    double tol = 1e-12;  // tolerance (can be adjusted)
    int k = 0;
    int min_dim = std::min((int)d, (int)n);
    auto R_acc = R_sk.accessor<double, 2>();
    for (int i = 0; i < min_dim; i++) {
        if (std::abs(R_acc[i][i]) > tol) {
            k++;
        }
    }
    if (k == 0) {
        throw std::runtime_error("Detected rank zero in R_sk.");
    }

    // Step 4 (continued): Extract permutation vector J.
    // LAPACKE returns 1-indexed pivot indices; convert them to 0-indexed.
    std::vector<int64_t> J_vec(n);
    for (int i = 0; i < n; i++) {
        J_vec[i] = jpvt[i] - 1;
    }
    torch::Tensor J = torch::tensor(J_vec, torch::dtype(torch::kInt64));

    // Step 7: Extract M_k = M_local(:, J[0:k]) -- the first k pivoted columns of M_local.
    torch::Tensor indices = J.slice(0, 0, k);
    torch::Tensor M_k = M_local.index_select(1, indices);  // dimensions: m x k

    // Step 8: Extract A_k_sk = R_sk[0:k, 0:k] from the QR of the sketch.
    torch::Tensor A_k_sk = R_sk.slice(0, 0, k).slice(1, 0, k).contiguous();

    // Step 9: Precondition M_k by solving M_pre = M_k * inv(A_k_sk)
    // Using cblas_dtrsm to solve X * A_k_sk = M_k for X.
    M_k = M_k.contiguous();
    int m_dim = M_k.size(0);  // number of rows in M_local (m)
    int k_dim = M_k.size(1);  // equals k
    cblas_dtrsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                m_dim, k_dim, 1.0, A_k_sk.data_ptr<double>(), k_dim, M_k.data_ptr<double>(), k_dim);
    // Now M_k has been overwritten with M_pre.
    torch::Tensor M_pre = M_k;  // dimensions: m x k

    // Step 10: Compute G = M_pre^T * M_pre.
    torch::Tensor G = torch::mm(M_pre.transpose(0, 1), M_pre);  // dimensions: k x k

    // Step 11: Compute Cholesky decomposition of G: G = L * L^T.
    torch::Tensor L = torch::linalg_cholesky(G, /*upper=*/false);

    // Step 12: Compute Q_k = M_pre * inv(L).
    torch::Tensor Q_k = M_pre.clone();
    cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit,
                m_dim, k_dim, 1.0, L.data_ptr<double>(), k_dim, Q_k.data_ptr<double>(), k_dim);

    // Step 13: Undo preconditioning: Compute R_k = L * (R_sk[0:k, :]).
    torch::Tensor R_k_sk = R_sk.slice(0, 0, k).contiguous();  // dimensions: k x n
    torch::Tensor R_k = torch::mm(L, R_k_sk);                 // dimensions: k x n

    return std::make_tuple(Q_k, R_k, J);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cqrrpt(const torch::Tensor& M, double gamma, const std::string& F) {
    // Step 5: Set d = ceil(gamma * n)
    int64_t n = M.size(1);
    int64_t d = static_cast<int64_t>(std::ceil(gamma * n));

    // Step 6: Generate a d x m sketching matrix S
    int64_t m = M.size(0);
    torch::Tensor S;
    if (F == "default") {
        // Default to a sparse random matrix
        S = torch::randn({d, m}, M.options());
    } else {
        // Handle other distribution families if needed
        TORCH_CHECK(false, "Unsupported distribution family: ", F);
    }

    // Step 7: Compute [Q, R, J] = cqrrpt_core(M, S)
    auto [Q, R, J] = cqrrpt_core(M, S);

    // Step 8: Return Q, R, J
    return std::make_tuple(Q, R, J);
}