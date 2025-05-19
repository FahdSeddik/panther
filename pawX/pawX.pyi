from enum import Enum
from typing import Optional, Tuple, overload

import torch

class DistributionFamily(Enum):
    Gaussian = "Gaussian"
    Uniform = "Uniform"

def test_tensor_accessor(tensor: torch.Tensor) -> None: ...
def scaled_sign_sketch(
    m: int,
    n: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
        Generates a scaled sign sketch matrix as a PyTorch tensor.

        A scaled sign sketch is a random projection matrix where each entry is independently set to +1 or -1,
        scaled by a normalization factor. This is commonly used in randomized linear algebra and sketching algorithms.

        Args:
            m (int): Number of rows in the output tensor (sketch dimension).
            n (int): Number of columns in the output tensor (original dimension).
            device (Optional[torch.device], optional): The device on which to create the tensor. Defaults to None (uses current device).
            dtype (Optional[torch.dtype], optional): The desired data type of returned tensor. Defaults to None (uses default dtype).

        Returns:
            torch.Tensor: A tensor of shape (m, n) containing the scaled sign sketch matrix.
    l;
        Example:
            >>> import torch
            >>> from panther.sketch import scaled_sign_sketch
            >>> m, n = 32, 128
            >>> S = scaled_sign_sketch(m, n)
            >>> print(S.shape)
            torch.Size([32, 128])
            >>> # Each entry is either +1/sqrt(m) or -1/sqrt(m)
            >>> print(torch.unique(S))
            tensor([-0.1768,  0.1768])  # For m=32, 1/sqrt(32) ≈ 0.1768

        References:
            - Woodruff, D. P. (2014). "Sketching as a Tool for Numerical Linear Algebra." Foundations and Trends® in Theoretical Computer Science, 10(1–2), 1–157.
            - Liberty, E., et al. (2008). "Randomized algorithms for the low-rank approximation of matrices." Proceedings of the National Academy of Sciences, 104(51), 20167–20172.
            - https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma
    """
    pass

def sketched_linear_forward(
    input: torch.Tensor,
    S1s: torch.Tensor,
    S2s: torch.Tensor,
    U1s: torch.Tensor,
    U2s: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    use_gpu: bool = False,
) -> torch.Tensor: ...
def sketched_linear_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    S1s: torch.Tensor,
    S2s: torch.Tensor,
    U1s: torch.Tensor,
    U2s: torch.Tensor,
    has_bias: bool = False,
    use_gpu: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
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
    --------
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
    pass

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
    pass

def dense_sketch_operator(
    m: int,
    n: int,
    distribution: DistributionFamily,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor: ...
@overload
def sketch_tensor(
    input: torch.Tensor,
    axis: int,
    new_size: int,
    distribution: DistributionFamily,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
@overload
def sketch_tensor(
    input: torch.Tensor,
    axis: int,
    new_size: int,
    sketch_matrix: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor: ...
def causal_numerator_forward(
    qs: torch.Tensor,
    ks: torch.Tensor,
    vs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def causal_numerator_backward(
    res_grad: torch.Tensor,
    sums: torch.Tensor,
    qs: torch.Tensor,
    ks: torch.Tensor,
    vs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def causal_denominator_forward(
    qs: torch.Tensor,
    ks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def causal_denominator_backward(
    res_grad: torch.Tensor,
    sums: torch.Tensor,
    qs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def rmha_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    Wq: torch.Tensor,
    Wk: torch.Tensor,
    Wv: torch.Tensor,
    W0: torch.Tensor,
    num_heads: int,
    embed_dim: int,
    kernel_fn: str,
    causal: bool,
    attention_mask: Optional[torch.Tensor],
    bq: Optional[torch.Tensor] = None,
    bk: Optional[torch.Tensor] = None,
    bv: Optional[torch.Tensor] = None,
    b0: Optional[torch.Tensor] = None,
    projection_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor: ...
def create_projection_matrix(
    m: int,
    d: int,
    seed: int = 42,
    scaling: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor: ...
def sketched_conv2d_forward(
    x: torch.Tensor,
    S1s: torch.Tensor,
    U1s: torch.Tensor,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    kernel_size: Tuple[int, int],
    bias: torch.Tensor | None = None,
) -> torch.Tensor: ...
def sketched_conv2d_backward(
    input: torch.Tensor,
    S1s: torch.Tensor,
    S2s: torch.Tensor,
    U1s: torch.Tensor,
    U2s: torch.Tensor,
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    kernel_size: Tuple[int, int],
    in_shape: Tuple[int, int],
    grad_out: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
