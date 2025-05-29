from typing import Optional, Tuple, Union

import torch

from pawX import Axis, DistributionFamily
from pawX import count_skop as _count_skop
from pawX import dense_sketch_operator as _dense_sketch_operator
from pawX import gaussian_skop as _gaussian_skop
from pawX import scaled_sign_sketch as _scaled_sign_sketch
from pawX import sjlt_skop as _sjlt_skop
from pawX import sketch_tensor as _sketch_tensor
from pawX import sparse_sketch_operator as _sparse_sketch_operator
from pawX import srht as _srht


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

    """
    return _scaled_sign_sketch(m, n, device=device, dtype=dtype)


def dense_sketch_operator(
    m: int,
    n: int,
    distribution: DistributionFamily,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Creates a dense sketch operator matrix with entries sampled from a specified distribution.
    This function generates a dense random matrix of shape `(m, n)` where each entry is drawn independently from the given `distribution`. Such sketch operators are commonly used in randomized linear algebra, dimensionality reduction, and compressed sensing to project high-dimensional data into a lower-dimensional space while preserving certain geometric properties.

    Args:
        m : int
            The number of rows of the sketch operator (i.e., the target dimension after sketching).
        n : int
            The number of columns of the sketch operator (i.e., the original dimension of the data).
        distribution : DistributionFamily
            The distribution family from which to sample the entries of the sketch operator.
            This could be, for example, a standard normal distribution, a uniform distribution, or any other supported distribution.
        device : Optional[torch.device], default=None
            The device on which to allocate the resulting tensor (e.g., 'cpu' or 'cuda'). If None, defaults to the current device.
        dtype : Optional[torch.dtype], default=None
            The desired data type of the returned tensor. If None, defaults to the default dtype of the current torch device.
    Returns:
        torch.Tensor
            A dense tensor of shape `(m, n)` with entries sampled from the specified distribution.

    Example:
        >>> import torch
        >>> from panther.sketch import dense_sketch_operator
        >>> from panther.sketch import DistributionFamily
        >>> m, n = 100, 500
        >>> sketch = dense_sketch_operator(m, n, DistributionFamily.GAUSSIAN)
        >>> print(sketch.shape)
        torch.Size([100, 500])


    Notes
    -----
    - The choice of distribution affects the properties of the sketch operator. For example, using a standard normal distribution yields a Johnson-Lindenstrauss transform.
    - For large matrices, consider using sparse sketch operators for improved efficiency.
    """
    return _dense_sketch_operator(m, n, distribution, device=device, dtype=dtype)


def sketch_tensor(
    input: torch.Tensor,
    axis: int,
    new_size: int,
    distribution: Optional[DistributionFamily] = None,
    sketch_matrix: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Sketches a given input tensor along a specified axis using a randomized projection technique.

    This function reduces the dimensionality of the input tensor along the specified axis to `new_size`
    by applying a sketching (random projection) operation. The type of random projection is determined
    by the `distribution_or_sketch_matrix` parameter, which can be either a distribution family or a precomputed sketching matrix.
    If a distribution family is provided, the function returns a tuple (sketched_tensor, sketch_matrix).
    If a sketching matrix (torch.Tensor) is provided, only the sketched tensor is returned.

    Parameters:
        input (torch.Tensor):
            The input tensor to be sketched. Can be of any shape, but the dimension along `axis` will be reduced.
        axis (int):
            The axis along which to apply the sketching operation. Must be a valid axis for `input`.
        new_size (int):
            The target size for the specified axis after sketching. Must be less than or equal to the original size.
        distribution_or_sketch_matrix (Union[DistributionFamily, torch.Tensor]):
            Either the distribution family to use for generating the sketching matrix (e.g., Gaussian, Rademacher),
            or a precomputed sketching matrix (torch.Tensor) to use directly.
        device (Optional[torch.device], default=None):
            The device on which to perform the computation and allocate the sketching matrix. If None, uses the device of `input`.
        dtype (Optional[torch.dtype], default=None):
            The desired data type of the output tensor and sketching matrix. If None, uses the dtype of `input`.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            If a distribution family is provided, returns a tuple (sketched_tensor, sketch_matrix).
            If a sketching matrix is provided, returns only the sketched tensor.

    Example:
        >>> import torch
        >>> from panther.sketch.core import sketch_tensor, DistributionFamily
        >>> x = torch.randn(10, 100)
        >>> # Using a distribution family
        >>> sketched_x, sketch_matrix = sketch_tensor(
        ...     input=x,
        ...     axis=1,
        ...     new_size=20,
        ...     distribution_or_sketch_matrix=DistributionFamily.GAUSSIAN
        ... )
        >>> print(sketched_x.shape)
        torch.Size([10, 20])
        >>> print(sketch_matrix.shape)
        torch.Size([20, 100])
        >>> # Using a precomputed sketching matrix
        >>> from panther.sketch import dense_sketch_operator
        >>> S = dense_sketch_operator(100, 20, DistributionFamily.GAUSSIAN)
        >>> sketched_x = sketch_tensor(
        ...     input=x,
        ...     axis=1,
        ...     new_size=20,
        ...     distribution_or_sketch_matrix=S
        ... )
        >>> print(sketched_x.shape)
        torch.Size([10, 20])

    Notes:
        - The sketching operation is commonly used for dimensionality reduction, speeding up computations,
          and preserving essential structure in large-scale machine learning and signal processing tasks.
        - The choice of distribution affects the quality and properties of the sketch.
        - The returned sketching matrix can be reused for consistent projections or analysis.
    """
    # enforce exactly one path
    if distribution is not None and sketch_matrix is not None:
        raise ValueError(
            "Cannot specify both `distribution` and `sketch_matrix`, "
            f"got distribution={distribution} and sketch_matrix={sketch_matrix}"
        )

    # dispatch to the low-level binding
    if distribution is not None:
        # distribution path: return both sketched tensor and sketch matrix
        return _sketch_tensor(
            input=input,
            axis=axis,
            new_size=new_size,
            distribution=distribution,
            device=device,
            dtype=dtype,
        )
    elif sketch_matrix is not None:
        # sketch_matrix path: only return sketched tensor
        return _sketch_tensor(
            input=input,
            axis=axis,
            new_size=new_size,
            sketch_matrix=sketch_matrix,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(
            "Must specify exactly one of `distribution` or `sketch_matrix`, "
            f"got distribution={distribution} and sketch_matrix={sketch_matrix}"
        )


def sparse_sketch_operator(
    m: int,
    n: int,
    vec_nnz: int,
    axis: Axis,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Creates a sparse sketch operator matrix with specified number of non-zero entries per vector.

    This function generates a sparse random matrix of shape `(m, n)` where each row or column (depending on the `axis`)
    has exactly `vec_nnz` non-zero entries. This is commonly used in randomized linear algebra and sketching algorithms
    to project high-dimensional data into a lower-dimensional space while preserving certain geometric properties.

    Args:
        m : int
            The number of rows of the sketch operator (i.e., the target dimension after sketching).
        n : int
            The number of columns of the sketch operator (i.e., the original dimension of the data).
        vec_nnz : int
            The number of non-zero entries per vector (row or column) in the sketch operator.
        axis : Axis
            The axis along which to create the sparse sketch operator. Can be either `Axis.Short` or `Axis.Long`.
        device : Optional[torch.device], default=None
            The device on which to allocate the resulting tensor (e.g., 'cpu' or 'cuda'). If None, defaults to the current device.
        dtype : Optional[torch.dtype], default=None
            The desired data type of the returned tensor. If None, defaults to the default dtype of the current torch device.

    Returns:
        torch.Tensor
            A sparse COO tensor of shape `(m, n)` with exactly `vec_nnz` non-zero entries per vector along the specified axis.

    Example:
        >>> import torch
        >>> from panther.sketch import sparse_sketch_operator, Axis
        >>> m, n = 100, 500
        >>> vec_nnz = 5
        >>> sketch = sparse_sketch_operator(m, n, vec_nnz, Axis.Short)
        >>> print(sketch.shape)
        torch.Size([100, 500])
        >>> print(sketch._nnz())  # Number of non-zero entries in the sparse tensor

    """
    return _sparse_sketch_operator(m, n, vec_nnz, axis, device=device, dtype=dtype)


def srht(x: torch.Tensor, m: int) -> torch.Tensor:
    """
    Subsampled Randomized Hadamard Transform (SRHT).

    This function computes a Subsampled Randomized Hadamard Transform of a 1D input tensor `x` of length `n`
    (where `n` must be a power of 2). It performs the following steps:

    1. Multiply the input by a random diagonal matrix with ±1 entries (random sign flipping).
    2. Apply the Fast Walsh-Hadamard Transform (FWHT).
    3. Uniformly subsample `m` rows from the result.

    This is commonly used in randomized numerical linear algebra and compressed sensing to reduce
    dimensionality while approximately preserving distances.

    Parameters
    ----------
    x : torch.Tensor
        A 1D input tensor of shape `(n,)` where `n` is a power of 2.
        The tensor must reside on the CPU and be of floating point type.

    m : int
        The number of rows to subsample from the Hadamard-transformed output.
        Must satisfy `0 < m <= n`.

    Returns
    -------
    torch.Tensor
        A 1D tensor of shape `(m,)` containing the subsampled rows from the Hadamard-transformed vector.

    Raises
    ------
    RuntimeError
        If `x` is not on CPU, is not 1-dimensional, or `n` is not a power of 2.
        If `m` is greater than `n`.

    Example
    -------
    >>> import torch
    >>> from panther.sketch import srht
    >>> x = torch.randn(8)
    >>> y = srht(x, 4)
    >>> print(y.shape)
    torch.Size([4])
    """
    return _srht(x, m)


def count_skop(
    m: int,
    d: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Generates a CountSketch sketching matrix of shape (m, d).

    Each column of the sketching matrix has exactly one non-zero entry,
    which is randomly assigned to a row via a simple hash (uniform random),
    and assigned a value of ±1 with equal probability. The result is a sparse
    sketching operator used to project high-dimensional data into a lower-dimensional
    space while approximately preserving inner products.

    This is commonly used in streaming and randomized linear algebra algorithms.

    Args:
        m (int): Number of rows in the sketching matrix (target dimension).
        d (int): Number of columns in the sketching matrix (original dimension).

    Returns:
        torch.Tensor: A sparse sketching matrix of shape (m, d) with exactly one
                      non-zero entry per column (either +1 or -1).

    Example:
        >>> import torch
        >>> from panther.sketch import count_sketch_operator
        >>> m, d = 100, 1000
        >>> S = count_skop(m, d)
        >>> X = torch.randn(50, d)  # Input data
        >>> X_sketched = X @ S.t()
        >>> X_sketched.shape
        torch.Size([50, 100])
    """
    return _count_skop(m, d, device=device, dtype=dtype)


def gaussian_skop(
    m: int,
    d: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Generates a Gaussian random projection matrix of shape (m, d).

    Each entry in the sketching matrix is an independent sample from
    the normal distribution N(0, 1/m). This type of projection approximately
    preserves Euclidean distances and inner products with high probability.

    Args:
        m (int): Number of rows in the sketching matrix (target dimension).
        d (int): Number of columns in the sketching matrix (original dimension).

    Returns:
        torch.Tensor: A dense sketching matrix of shape (m, d) with entries ~ N(0, 1/m).

    Example:
        >>> S = gaussian_skop(100, 1000)
        >>> X = torch.randn(64, 1000)
        >>> X_sketched = X @ S.t()
        >>> X_sketched.shape
        torch.Size([64, 100])
    """
    return _gaussian_skop(m, d, device=device, dtype=dtype)


def sjlt_skop(
    m: int,
    d: int,
    sparsity: int = 2,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Generates a sparse Johnson-Lindenstrauss Transform (SJLT) sketching matrix.

    Each column of the matrix has exactly `sparsity` non-zero entries, chosen uniformly
    at random among the rows, with values ±1. The matrix is scaled by 1/√sparsity.
    This sketch is ideal for fast projections with low memory usage.

    Args:
        m (int): Number of rows in the sketching matrix (target dimension).
        d (int): Number of columns in the sketching matrix (original dimension).
        sparsity (int, optional): Number of non-zero entries per column. Defaults to 2.

    Returns:
        torch.Tensor: A sparse sketching matrix of shape (m, d), entries ∈ {±1/√sparsity}.

    Example:
        >>> S = sjlt_skop(200, 1000, sparsity=3)
        >>> X = torch.randn(32, 1000)
        >>> X_sketched = X @ S.t()
        >>> X_sketched.shape
        torch.Size([32, 200])
    """
    return _sjlt_skop(m, d, sparsity=sparsity, device=device, dtype=dtype)
