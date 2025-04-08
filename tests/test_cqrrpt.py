import torch

import pawX


def test_cqrrpt():
    torch.manual_seed(42)  # For reproducibility

    # Generate a random m x n matrix M (tall matrix, m > n)
    m, n = 10, 5
    M = torch.randn(m, n, dtype=torch.double)

    # Run the CQRRPT function
    Q, R, J = pawX.cqrrpt(M)

    # Check the shapes of outputs
    k = R.shape[0]
    assert Q.shape == (m, k), f"Unexpected shape for Q: {Q.shape}, expected ({m}, {k})"
    assert R.shape == (k, n), f"Unexpected shape for R: {R.shape}, expected ({k}, {n})"
    assert J.shape == (n,), f"Unexpected shape for J: {J.shape}, expected ({n},)"


def test_cqrrpt_wide_matrix():
    torch.manual_seed(42)
    m, n = 5, 10  # Wide matrix (not typical but still testable)
    M = torch.randn(m, n, dtype=torch.double)
    Q, R, J = pawX.cqrrpt(M)
    assert R.shape[0] == min(m, n), "Unexpected rank in wide matrix case!"
