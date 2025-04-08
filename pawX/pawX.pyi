from enum import Enum
from typing import Optional, Tuple, overload

import torch

def scaled_sign_sketch(
    m: int,
    n: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor: ...
def sketched_linear_forward(
    input: torch.Tensor,
    S1s: torch.Tensor,
    S2s: torch.Tensor,
    U1s: torch.Tensor,
    U2s: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor: ...
def sketched_linear_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    S1s: torch.Tensor,
    S2s: torch.Tensor,
    U1s: torch.Tensor,
    U2s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, torch.Tensor]: ...
def cqrrpt(
    M: torch.Tensor, gamma: float = 1.25, F: str = "default"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def randomized_svd(
    A: torch.Tensor, k: int, tol: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class DistributionFamily(Enum):
    Gaussian = "Gaussian"
    Uniform = "Uniform"

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
