from enum import Enum
from typing import Optional, Tuple, overload

import torch

class DistributionFamily(Enum):
    Gaussian = "Gaussian"
    Uniform = "Uniform"

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
    use_tensor_core: bool = False,
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
    M: torch.Tensor,
    gamma: float = 1.25,
    F: DistributionFamily = DistributionFamily.Gaussian,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def randomized_svd(
    A: torch.Tensor, k: int, tol: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
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
def causal_numerator_apply(
    query_prime: torch.Tensor,
    key_prime: torch.Tensor,
    value_prime: torch.Tensor,
) -> torch.Tensor: ...
def causal_denominator_apply(
    query_prime: torch.Tensor,
    key_prime: torch.Tensor,
) -> torch.Tensor: ...
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
    bias: torch.Tensor,
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
