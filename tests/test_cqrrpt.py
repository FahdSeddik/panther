import pytest
import torch

import pawX


# Helper function to check reconstruction accuracy.
def check_qr_decomposition(M, Q, R, J, tol=1e-9):
    """
    Checks that after applying the permutation J to M's columns,
    the product Q*R reconstructs M correctly.
    """
    # Permute columns of M using J.
    M_permuted = M[:, J]
    error = torch.norm(M_permuted - torch.mm(Q, R))
    return error.item()


# Helper function to check that Q is orthonormal.
def check_q_orthonormality(Q, tol=1e-9):
    I = torch.eye(Q.shape[1], dtype=Q.dtype, device=Q.device)  # noqa: E741
    error = torch.norm(torch.mm(Q.T, Q) - I)
    return error.item()


#########################################
# Basic tests (from the original suite)
#########################################
def test_cqrrpt():
    torch.manual_seed(42)  # reproducibility

    # Tall matrix: m > n
    m, n = 10, 5
    M = torch.randn(m, n, dtype=torch.double)

    # Run the CQRRPT function
    Q, R, J = pawX.cqrrpt(M)

    # Check shapes.
    k = R.shape[0]
    assert Q.shape == (m, k), f"Unexpected Q shape: got {Q.shape}, expected ({m}, {k})"
    assert R.shape == (k, n), f"Unexpected R shape: got {R.shape}, expected ({k}, {n})"
    assert J.shape == (n,), f"Unexpected J shape: got {J.shape}, expected ({n},)"


def test_cqrrpt_wide_matrix():
    torch.manual_seed(42)
    # Wide matrix (m < n)
    m, n = 5, 10
    M = torch.randn(m, n, dtype=torch.double)
    Q, R, J = pawX.cqrrpt(M)
    # In a wide matrix the rank cannot exceed m.
    assert R.shape[0] <= m, "Computed rank exceeds expected maximum for wide matrix!"


#########################################
# Parameterized tests for reconstruction.
#########################################
@pytest.mark.parametrize("m,n", [(10, 5), (20, 10), (15, 15), (5, 10)])
def test_cqrrpt_reconstruction(m, n):
    torch.manual_seed(42)
    M = torch.randn(m, n, dtype=torch.double)
    Q, R, J = pawX.cqrrpt(M)
    err = check_qr_decomposition(M, Q, R, J)
    assert err < 1e-9, f"Reconstruction error {err} too high for matrix size ({m},{n})!"


#########################################
# Test orthonormality of Q.
#########################################
@pytest.mark.parametrize("m,n", [(10, 5), (20, 10), (15, 15), (5, 10)])
def test_q_orthonormality(m, n):
    torch.manual_seed(42)
    M = torch.randn(m, n, dtype=torch.double)
    Q, R, J = pawX.cqrrpt(M)
    err = check_q_orthonormality(Q)
    assert err < 1e-9, f"Q is not orthonormal: error {err}"


#########################################
# Test varying the gamma parameter.
#########################################
@pytest.mark.parametrize("gamma", [1.0, 1.25, 1.5, 2.0])
def test_varying_gamma(gamma):
    torch.manual_seed(42)
    m, n = 20, 10
    M = torch.randn(m, n, dtype=torch.double)
    Q, R, J = pawX.cqrrpt(M, gamma)
    err = check_qr_decomposition(M, Q, R, J)
    assert err < 1e-9, f"Reconstruction error {err} too high for gamma={gamma}!"


#########################################
# Test low-rank input matrices.
#########################################
def test_low_rank_matrix():
    torch.manual_seed(42)
    m, n, r = 50, 30, 5
    A = torch.randn(m, r, dtype=torch.double)
    B = torch.randn(r, n, dtype=torch.double)
    M = torch.mm(A, B)
    Q, R, J = pawX.cqrrpt(M)
    err = check_qr_decomposition(M, Q, R, J)
    assert err < 1e-9, f"Reconstruction error {err} too high for low-rank matrix!"


#########################################
# Statistical consistency test.
#########################################
def test_statistical_consistency():
    torch.manual_seed(42)
    m, n = 50, 30
    num_trials = 20
    errors = []
    for trial in range(num_trials):
        M = torch.randn(m, n, dtype=torch.double)
        Q, R, J = pawX.cqrrpt(M)
        err = check_qr_decomposition(M, Q, R, J)
        errors.append(err)
    avg_err = sum(errors) / len(errors)
    max_err = max(errors)
    print(f"Average error over {num_trials} trials: {avg_err}")
    print(f"Max error over {num_trials} trials: {max_err}")
    assert avg_err < 1e-8, f"Average error {avg_err} is too high!"
    assert max_err < 1e-8, f"Max error {max_err} is too high!"


#########################################
# Test with an ill-conditioned matrix.
#########################################
def test_ill_conditioned_matrix():
    torch.manual_seed(42)
    m, n = 30, 30
    U, _ = torch.linalg.qr(torch.randn(m, m, dtype=torch.double))
    V, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.double))
    singular_values = torch.logspace(0, -8, n, dtype=torch.double)
    S = torch.diag(singular_values)
    M = U[:, :n] @ S @ V.T
    Q, R, J = pawX.cqrrpt(M)
    err = check_qr_decomposition(M, Q, R, J, tol=1e-7)
    # Allow a somewhat looser tolerance for ill-conditioned matrices.
    assert (
        err < 1e-6
    ), f"Reconstruction error {err} too high for ill-conditioned matrix!"


#########################################
# Test validity of the permutation vector.
#########################################
def test_permutation_vector():
    torch.manual_seed(42)
    m, n = 20, 10
    M = torch.randn(m, n, dtype=torch.double)
    _, _, J = pawX.cqrrpt(M)
    sorted_J = torch.sort(J).values
    expected = torch.arange(0, n, dtype=J.dtype)
    assert torch.allclose(
        sorted_J, expected
    ), "Permutation vector J is not a valid permutation!"
