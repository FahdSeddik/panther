from typing import Optional, Tuple

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
