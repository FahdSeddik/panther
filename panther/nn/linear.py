import math
import warnings
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import init

from panther.nn.pawXimpl import sketched_linear_backward, sketched_linear_forward
from panther.sketch import scaled_sign_sketch as gen_U
from panther.utils.compatibility import has_tensor_core_support


class SketchedLinearFunction(Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(
        input: torch.Tensor,
        S1s: torch.Tensor,
        S2s: torch.Tensor,
        U1s: torch.Tensor,
        U2s: torch.Tensor,
        bias: torch.Tensor,
        use_gpu: bool = False,
    ):
        return sketched_linear_forward(input, S1s, S2s, U1s, U2s, bias, use_gpu)

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any):
        input, S1s, S2s, U1s, U2s, bias, use_gpu = inputs
        ctx.save_for_backward(
            input,
            S1s,
            S2s,
            U1s,
            U2s,
            torch.tensor(use_gpu, dtype=torch.bool),
            torch.tensor(bias is not None, dtype=torch.bool),
        )

    @staticmethod
    def backward(ctx: Any, *grad_output: Any) -> Any:
        # dl/dS2_i = U1_i g h_in^T / 2 * l
        # dl/dS1_i = g h_in^T U2_i^T / 2 * l
        # dl/dh_in = 1/(2*l) * (sum_{i=1}^{l} (S1_i^T U1_i g) + sum_{i=1}^{l} (U2_i^T S2_i g))
        # dl/db = g
        input, S1s, S2s, U1s, U2s, use_gpu_tensor, has_bias = ctx.saved_tensors
        use_gpu = use_gpu_tensor.item()
        grads = sketched_linear_backward(
            grad_output=grad_output[0],
            input=input,
            S1s=S1s,
            S2s=S2s,
            U1s=U1s,
            U2s=U2s,
            has_bias=has_bias.item(),
            use_gpu=use_gpu,
        )
        return (
            grads[0],  # h_in
            grads[1],  # S1s
            grads[2],  # S2s
            None,  # U1s
            None,  # U2s
            grads[3] if has_bias.item() else None,  # bias
            None,  # boolean for use_tensor_core
        )


class SKLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "num_terms", "low_rank"]
    in_features: int
    out_features: int
    num_terms: int
    low_rank: int
    S1s: torch.Tensor
    S2s: torch.Tensor
    U1s: torch.Tensor
    U2s: torch.Tensor
    WARNING_MSG_TC = (
        "Tensor Core not utilized. Ensure 'in_features', 'out_features', "
        "and 'low_rank' are multiples of 16 (current: {}, {}, {})."
    )
    WARNING_MSG_BATCH = (
        "Batch size {} is not a multiple of 16. Tensor Core optimizations disabled."
    )
    WARNING_MSG_FC = (
        "The sketching layer uses more parameters than a fully connected layer. "
        "Consider reducing 'num_terms' or 'low_rank' for efficiency."
    )

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_terms: int,
        low_rank: int,
        W_init=None,
        bias: bool = True,
        dtype=None,
        device=None,
    ):
        if (
            2 * num_terms * low_rank * (out_features + in_features)
            > out_features * in_features
        ):
            warnings.warn(
                self.WARNING_MSG_FC,
                category=UserWarning,
                stacklevel=2,
            )
        self.has_tensor_core = has_tensor_core_support()
        self.use_gpu = True
        if self.has_tensor_core and (
            in_features % 16 != 0 or out_features % 16 != 0 or low_rank % 16 != 0
        ):
            # Log warning to user
            self.use_gpu = False
            warnings.warn(
                self.WARNING_MSG_TC.format(in_features, out_features, low_rank),
                UserWarning,
                stacklevel=2,
            )

        factory_kwargs = {"dtype": dtype, "device": device}
        super(SKLinear, self).__init__()

        self.num_terms = num_terms
        self.low_rank = low_rank
        self.out_features = out_features
        self.in_features = in_features

        # Register U1s and U2s as buffers since they are not learnable
        self.register_buffer(
            "U1s",
            torch.stack(
                [
                    gen_U(self.low_rank, self.out_features, **factory_kwargs)
                    for _ in range(num_terms)
                ]
            ).contiguous(),
        )  # kxd1
        self.register_buffer(
            "U2s",
            torch.stack(
                [
                    gen_U(self.low_rank, self.in_features, **factory_kwargs).T
                    for _ in range(num_terms)
                ]
            ).contiguous(),
        )  # d2xk

        # W is used to only initialize S
        if W_init is None:
            W = torch.empty(self.in_features, self.out_features, **factory_kwargs)
            init.kaiming_uniform_(W, a=math.sqrt(5))
        else:
            W = W_init.T.detach().clone()

        # S1s and S2s are precomputed but not updated in the backward pass
        self.S1s = nn.Parameter(
            torch.stack([torch.matmul(W, self.U1s[i].T) for i in range(num_terms)])
        ).contiguous()  # d2xk
        self.S2s = nn.Parameter(
            torch.stack([torch.matmul(self.U2s[i].T, W) for i in range(num_terms)])
        ).contiguous()  # kxd1

        # Bias term initialized with a small standard deviation
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_features, **factory_kwargs)
            ).contiguous()
            fan_in, _ = init._calculate_fan_in_and_fan_out(W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, h_in: torch.Tensor) -> torch.Tensor:
        def _reshape_input(h_in):
            was_1d = h_in.dim() == 1
            if was_1d:
                h_in = h_in.unsqueeze(0)
            orig_shape = h_in.shape[:-1]
            h_in = h_in.view(-1, self.in_features)
            return h_in, orig_shape, was_1d

        h_in, orig_shape, was_1d = _reshape_input(h_in)
        batch_size = h_in.shape[0]
        use_gpu = self.use_gpu

        if h_in.is_cuda and self.has_tensor_core and use_gpu and (batch_size % 16 != 0):
            use_gpu = False
            warnings.warn(
                self.WARNING_MSG_BATCH.format(batch_size),
                UserWarning,
                stacklevel=2,
            )

        out = SketchedLinearFunction.apply(
            h_in, self.S1s, self.S2s, self.U1s, self.U2s, self.bias, use_gpu
        )
        return (
            out.view(self.out_features)
            if was_1d
            else out.view(*orig_shape, self.out_features)
        )


if __name__ == "__main__":
    in_features = 8192
    out_features = 32768
    num_terms = 3
    low_rank = 512
    batch_size = 64
    linear = SKLinear(
        in_features=in_features,
        out_features=out_features,
        num_terms=num_terms,
        low_rank=low_rank,
        dtype=torch.float32,
        device=torch.device("cuda:0"),
    )
    input = torch.randn(
        batch_size, in_features, dtype=torch.float32, device=torch.device("cuda:0")
    )
    output = linear(input)
    output_expected = torch.zeros(
        batch_size, out_features, dtype=torch.float32, device=torch.device("cuda:0")
    )
