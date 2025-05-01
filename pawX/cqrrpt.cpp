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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cqrrpt_core(
    const torch::Tensor& M, const torch::Tensor& S) {
    // Ensure inputs are in double precision and contiguous.
    torch::Tensor M_local = M.to(torch::kDouble).contiguous();
    torch::Tensor S_local = S.to(torch::kDouble).contiguous();

    // Step 3: Compute the sketch: M_sk = S_local * M_local.
    torch::Tensor M_sk = torch::mm(S_local, M_local);  // dimensions: d x n
    int64_t d = M_sk.size(0);
    int64_t n = M_sk.size(1);

    // Step 4: Compute a QR factorization with column pivoting on M_sk.
    M_sk = M_sk.contiguous();  // ensure contiguity
    double* M_sk_data = M_sk.data_ptr<double>();

    // Allocate permutation vector.
    std::vector<lapack_int> jpvt(n, 0);
    int min_dn = std::min(static_cast<int>(d), static_cast<int>(n));
    std::vector<double> tau(min_dn);

    // Call LAPACKE_dgeqp3 (note: using row-major storage).
    int info = LAPACKE_dgeqp3(LAPACK_ROW_MAJOR, d, n, M_sk_data, n, jpvt.data(), tau.data());
    if (info != 0) {
        throw std::runtime_error("LAPACKE_dgeqp3 failed");
    }
    // The upper triangular part of M_sk now contains R_sk.
    torch::Tensor R_sk = M_sk.triu();

    // Step 6: Determine numerical rank k from the diagonal of R_sk.
    double tol = 1e-12;
    int k = 0;
    int min_dim = std::min(static_cast<int>(d), static_cast<int>(n));
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
    std::vector<int64_t> J_vec(n);
    for (int i = 0; i < n; i++) {
        J_vec[i] = jpvt[i] - 1;  // convert 1-indexed to 0-indexed.
    }
    torch::Tensor J = torch::tensor(J_vec, torch::dtype(torch::kInt64));

    // Step 7: Extract M_k = M_local(:, J[0:k]) -- the first k pivoted columns.
    torch::Tensor indices = J.slice(0, 0, k);
    torch::Tensor M_k = M_local.index_select(1, indices);  // dimensions: m x k

    // Step 8: Extract A_k_sk = R_sk[0:k, 0:k] from the QR of the sketch.
    torch::Tensor A_k_sk = R_sk.slice(0, 0, k).slice(1, 0, k).contiguous();

    // Step 9: Precondition M_k by solving M_pre = M_k * inv(A_k_sk)
    // Solve X * A_k_sk = M_k for X.
    M_k = M_k.contiguous();
    int m_dim = M_k.size(0);  // m
    int k_dim = M_k.size(1);  // equals k
    cblas_dtrsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                m_dim, k_dim, 1.0, A_k_sk.data_ptr<double>(), k_dim, M_k.data_ptr<double>(), k_dim);
    // Now, M_k has been overwritten with M_pre.
    torch::Tensor M_pre = M_k;  // dimensions: m x k

    // Step 10: Compute G = M_pre^T * M_pre.
    torch::Tensor G = torch::mm(M_pre.transpose(0, 1), M_pre);  // dimensions: k x k

    // Step 11: Compute Cholesky decomposition of G.
    // Here, we compute the lower triangular factor L such that G = L * L^T.
    torch::Tensor L = torch::linalg_cholesky(G, /*upper=*/false);

    // Step 12: Compute Q_k = M_pre * inv(L^T)
    // We solve X * L^T = M_pre for X.
    torch::Tensor Q_k = M_pre.clone();
    cblas_dtrsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                m_dim, k_dim, 1.0, L.transpose(0, 1).data_ptr<double>(), k_dim,
                Q_k.data_ptr<double>(), k_dim);

    // Step 13: Undo preconditioning: Compute R_k = L^T * R_sk[0:k, :].
    torch::Tensor R_k_sk = R_sk.slice(0, 0, k).contiguous();   // dimensions: k x n
    torch::Tensor R_k = torch::mm(L.transpose(0, 1), R_k_sk);  // dimensions: k x n

    return std::make_tuple(Q_k, R_k, J);
}

//
// Revised high-level cqrrpt function.
//
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cqrrpt(
    const torch::Tensor& M, double gamma, const DistributionFamily& F) {
    // Set d = ceil(gamma * n)
    int64_t n = M.size(1);
    int64_t d = static_cast<int64_t>(std::ceil(gamma * n));

    // Generate a d x m sketching matrix S.
    int64_t m = M.size(0);
    torch::Tensor S;
    if (F == DistributionFamily::Gaussian) {
        // Default distribution: Gaussian (can be replaced with a sparse variant if desired)
        S = torch::randn({d, m}, M.options());
    } else {
        TORCH_CHECK(false, "Unsupported distribution family: ", F);
    }

    // Compute [Q, R, J] using the revised core.
    auto [Q, R, J] = cqrrpt_core(M, S);
    return std::make_tuple(Q, R, J);
}