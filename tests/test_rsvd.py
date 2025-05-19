import pytest
import torch

from panther.linalg import randomized_svd  # Adjust the module name as needed


def test_shape_and_reconstruction():
    # Create a random matrix A of shape (m, n)
    m, n = 100, 50
    k = 20
    tol = 1e-5
    A = torch.randn(m, n, dtype=torch.float64)

    # Compute the RSVD decomposition
    U, S, V = randomized_svd(A, k, tol)

    # Check output shapes: U (m x k), S (k,), V (n x k)
    assert U.shape == (m, k), f"Expected U.shape to be {(m, k)}, got {U.shape}"
    assert S.shape == (k,), f"Expected S.shape to be {(k,)}, got {S.shape}"
    assert V.shape == (n, k), f"Expected V.shape to be {(n, k)}, got {V.shape}"

    # Reconstruct the matrix approximation: A_approx = U * diag(S) * V^T
    A_approx = torch.mm(U, torch.mm(torch.diag(S), V.t()))

    # Since RSVD returns a rank-k approximation, the error should be small.
    error = torch.norm(A - A_approx, p="fro").item()
    assert error < 60.0, f"Reconstruction error too high: {error}"


def test_orthonormality_of_U():
    m, n = 100, 50
    k = 20
    tol = 1e-5
    A = torch.randn(m, n, dtype=torch.float64)

    # Compute the RSVD decomposition
    U, _, _ = randomized_svd(A, k, tol)

    # Verify that columns of U are orthonormal: U^T U ~ I
    U_product = torch.mm(U.t(), U)
    eye = torch.eye(k, dtype=torch.float64)
    diff = torch.norm(U_product - eye, p="fro").item()
    assert diff < 1e-5, f"Columns of U are not orthonormal, difference: {diff}"


if __name__ == "__main__":
    pytest.main([__file__])
