"""Core linear algebra operations for the panther package."""

from typing import Tuple

import torch

from pawX import DistributionFamily
from pawX import cqrrpt as _cqrrpt
from pawX import randomized_svd as _randomized_svd


def cqrrpt(
    M: torch.Tensor,
    gamma: float = 1.25,
    F: DistributionFamily = DistributionFamily.Gaussian,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs CholeskyQR with randomization and pivoting (CQRRPT) for tall matrices.

    This function implements a numerically stable QR decomposition for tall matrices using
    Cholesky factorization, randomization, and column pivoting. It is particularly useful
    for large-scale problems where standard QR decomposition may be computationally expensive
    or numerically unstable.

    Args:
        M (torch.Tensor): The input tall matrix of shape (m, n) with m >= n.
        gamma (float, optional): Oversampling parameter to improve numerical stability.
            Default is 1.25.
        F (DistributionFamily, optional): The distribution family used for random projections.
            Default is DistributionFamily.Gaussian.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - Q (torch.Tensor): Orthonormal matrix of shape (m, n).
            - R (torch.Tensor): Upper triangular matrix of shape (n, n).
            - P (torch.Tensor): Permutation matrix or indices representing column pivoting.

    Example:
        >>> import torch
        >>> from panther.linalg import cqrrpt, DistributionFamily
        >>> # Create a random tall matrix
        >>> m, n = 100, 20
        >>> M = torch.randn(m, n)
        >>> # Perform CQRRPT
        >>> Q, R, P = cqrrpt(M, gamma=1.25, F=DistributionFamily.Gaussian)
        >>> # Q is (m, n), R is (n, n), P is (n,)
        >>> # Verify the decomposition: Q @ R should approximate M[:, P]
        >>> M_permuted = M[:, P]
        >>> reconstruction = Q @ R
        >>> print("Reconstruction error:", torch.norm(reconstruction - M_permuted))
        >>> # Q should be orthonormal: Q.T @ Q ≈ I
        >>> print("Q orthogonality error:", torch.norm(Q.T @ Q - torch.eye(n)))
        >>> # R should be upper triangular
        >>> print("R upper triangular:", torch.allclose(R, torch.triu(R)))
        >>> # Optionally, check relative error
        >>> rel_error = torch.norm(reconstruction - M_permuted) / torch.norm(M_permuted)
        >>> print("Relative error:", rel_error.item())

    References:
        - Martinsson, P. G., Tropp, J. A. (2020). "Randomized Numerical Linear Algebra: Foundations & Algorithms."
        - Demmel, J., Dumitriu, I., & Holtz, O. (2007). "Fast linear algebra is stable." Numerische Mathematik, 108(1), 59-91.
        - https://github.com/pytorch/pytorch/issues/16763 (for discussions on randomized QR in PyTorch)

    Notes:
        - This method is especially effective for matrices where m >> n.
        - The randomization step improves the conditioning of the matrix before Cholesky factorization.
        - Pivoting ensures numerical stability and accurate rank-revealing properties.
    """
    return _cqrrpt(M, gamma, F)


def randomized_svd(
    A: torch.Tensor, k: int, tol: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes a truncated randomized Singular Value Decomposition (SVD) of a matrix.

    This function efficiently approximates the singular value decomposition of a given matrix `A` using randomized algorithms, which are particularly useful for large-scale matrices where traditional SVD is computationally expensive.

    Parameters
    ----------
    A : torch.Tensor
        The input matrix of shape (m, n) to decompose.
    k : int
        The number of singular values and vectors to compute (rank of the approximation).
    tol : float
        Tolerance for convergence. Smaller values yield more accurate results but may require more computation.

    Returns
    -------
    U : torch.Tensor
        Left singular vectors of shape (m, k).
    S : torch.Tensor
        Singular values of shape (k,).
    V : torch.Tensor
        Right singular vectors of shape (n, k).

    Examples
    --------
    >>> import torch
    >>> from panther.linalg.randomized_svd import randomized_svd
    >>> A = torch.randn(100, 50)
    >>> U, S, V = randomized_svd(A, k=10, tol=1e-5)
    >>> # Reconstruct the approximation
    >>> A_approx = U @ torch.diag(S) @ V.T
    >>> print(A_approx.shape)
    torch.Size([100, 50])

    References
    ----------
    - Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions." SIAM review, 53(2), 217-288.
    https://epubs.siam.org/doi/abs/10.1137/090771806
    - Scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html
    - PyTorch SVD documentation: https://pytorch.org/docs/stable/generated/torch.linalg.svd.html

    Notes
    -----
    Randomized SVD is especially useful for dimensionality reduction, principal component analysis (PCA), and latent semantic analysis (LSA) on large datasets.
    """
    return _randomized_svd(A, k, tol)
