#include "rsvd.h"

extern "C" {
#include <lapacke_config.h>
}

extern "C" {
#include <lapacke.h>
}

extern "C" {
#include <lapack.h>
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> randomized_svd(const torch::Tensor& A, int64_t k, double tol) {
    // Check input properties: A must be a double 2D tensor.
    TORCH_CHECK(A.dtype() == torch::kFloat64, "A must be a double tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D matrix");
    int64_t m = A.size(0);
    int64_t n = A.size(1);

    // Step 1: Generate a random Gaussian matrix Omega (n x k)
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(A.device());
    torch::Tensor Omega = torch::randn({n, k}, options);

    // Compute Y = A * Omega, Y is (m x k)
    torch::Tensor Y = torch::mm(A, Omega);

    // Step 2: Compute the QR factorization of Y using LAPACKE_dgeqp3
    // We want an orthonormal basis Q for the range of Y.
    // Ensure Y is contiguous in memory.
    torch::Tensor Y_copy = Y.clone().contiguous();
    int mY = Y_copy.size(0);  // equals m
    int nY = Y_copy.size(1);  // equals k
    int min_mn = std::min(mY, nY);

    // Prepare the pivot array and the tau vector for the factorization.
    std::vector<int> jpvt(nY, 0);
    std::vector<double> tau(min_mn);

    // Workspace query for dgeqp3_work (row-major layout is used)
    int lwork = -1;
    double work_query;
    int info = LAPACKE_dgeqp3_work(LAPACK_ROW_MAJOR, mY, nY,
                                   Y_copy.data_ptr<double>(), nY,
                                   jpvt.data(), tau.data(),
                                   &work_query, lwork);
    if (info != 0) {
        throw std::runtime_error("LAPACKE_dgeqp3_work workspace query failed");
    }
    lwork = static_cast<int>(work_query);
    std::vector<double> work(lwork);

    // Compute the QR factorization with column pivoting.
    info = LAPACKE_dgeqp3_work(LAPACK_ROW_MAJOR, mY, nY,
                               Y_copy.data_ptr<double>(), nY,
                               jpvt.data(), tau.data(),
                               work.data(), lwork);
    if (info != 0) {
        throw std::runtime_error("LAPACKE_dgeqp3_work failed");
    }

    // Generate Q from the factorization using LAPACKE_dorgqr_work.
    lwork = -1;
    double work_query2;
    info = LAPACKE_dorgqr_work(LAPACK_ROW_MAJOR, mY, nY, min_mn,
                               Y_copy.data_ptr<double>(), nY,
                               tau.data(), &work_query2, lwork);
    if (info != 0) {
        throw std::runtime_error("LAPACKE_dorgqr_work workspace query failed");
    }
    lwork = static_cast<int>(work_query2);
    std::vector<double> work2(lwork);
    info = LAPACKE_dorgqr_work(LAPACK_ROW_MAJOR, mY, nY, min_mn,
                               Y_copy.data_ptr<double>(), nY,
                               tau.data(), work2.data(), lwork);
    if (info != 0) {
        throw std::runtime_error("LAPACKE_dorgqr_work failed");
    }
    // Now Y_copy holds the orthonormal matrix Q (m x k)
    torch::Tensor Q = Y_copy;

    // Step 3: Compute the small projected matrix B = Qᵀ * A (of size k x n)
    torch::Tensor B = torch::mm(Q.transpose(0, 1), A);

    // Compute the SVD of B. Here we use torch::svd.
    torch::Tensor U_B, S, V;
    std::tie(U_B, S, V) = torch::svd(B, /* some compute_uv flag */ true);

    // Step 4: Back-project to form U = Q * U_B (of size m x k)
    torch::Tensor U = torch::mm(Q, U_B);

    // Optionally, you could enforce the tolerance tol here to truncate singular values.
    // For simplicity, we return the full computed SVD.

    return std::make_tuple(U, S, V);
}