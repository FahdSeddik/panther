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

template <>
void qrcp_impl<float>(int64_t d, int64_t n, float* M_sk_data, int64_t lda,
                      std::vector<lapack_int>& jpvt, std::vector<float>& tau) {
    LAPACKE_sgeqp3(LAPACK_ROW_MAJOR, d, n, M_sk_data, lda, jpvt.data(), tau.data());
}

template <>
void qrcp_impl<double>(int64_t d, int64_t n, double* M_sk_data, int64_t lda,
                       std::vector<lapack_int>& jpvt, std::vector<double>& tau) {
    LAPACKE_dgeqp3(LAPACK_ROW_MAJOR, d, n, M_sk_data, lda, jpvt.data(), tau.data());
}

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