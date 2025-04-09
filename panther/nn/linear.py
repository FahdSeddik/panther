import math
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import init

from panther.sketch import scaled_sign_sketch as gen_U
from pawX import sketched_linear_backward, sketched_linear_forward


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
    ):
        return sketched_linear_forward(input, S1s, S2s, U1s, U2s, bias)

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any):
        input, S1s, S2s, U1s, U2s, bias = inputs
        ctx.save_for_backward(input, S1s, S2s, U1s, U2s, bias)

    @staticmethod
    def backward(ctx: Any, *grad_output: Any) -> Any:
        # dl/dS2_i = U1_i g h_in^T / 2 * l
        # dl/dS1_i = g h_in^T U2_i^T / 2 * l
        # dl/dh_in = 1/(2*l) * (sum_{i=1}^{l} (S1_i^T U1_i g) + sum_{i=1}^{l} (U2_i^T S2_i g))
        # dl/db = g
        input, S1s, S2s, U1s, U2s, _ = ctx.saved_tensors
        grads = sketched_linear_backward(
            grad_output=grad_output[0],
            input=input,
            S1s=S1s,
            S2s=S2s,
            U1s=U1s,
            U2s=U2s,
        )
        return (
            grads[0],  # h_in
            grads[1],  # S1s
            grads[2],  # S2s
            None,  # U1s
            None,  # U2s
            grads[3],  # bias
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
        factory_kwargs = {"dtype": dtype, "device": device}
        super(SKLinear, self).__init__()

        # if (
        #     2 * num_terms * low_rank * (out_features + in_features)
        #     > out_features * in_features
        # ):
        #     raise ValueError(
        #         "The number of parameters in the sketching layer is larger "
        #         + "than the number of parameters in the fully connected layer."
        #     )

        self.num_terms = num_terms
        self.low_rank = low_rank
        self.out_features = out_features
        self.in_features = in_features

        # Register U1s and U2s as buffers since they are not learnable
        self.register_buffer(
            "U1s",
            torch.stack(
                [
                    gen_U(low_rank, out_features, **factory_kwargs)
                    for _ in range(num_terms)
                ]
            ),
        )  # kxd1
        self.register_buffer(
            "U2s",
            torch.stack(
                [
                    gen_U(low_rank, in_features, **factory_kwargs).T
                    for _ in range(num_terms)
                ]
            ),
        )  # d2xk

        # W is used to only initialize S
        if W_init is None:
            W = torch.empty(in_features, out_features, **factory_kwargs)
            init.kaiming_uniform_(W, a=math.sqrt(5))
        else:
            W = W_init.T.detach().clone()

        # S1s and S2s are precomputed but not updated in the backward pass
        self.S1s = nn.Parameter(
            torch.stack([torch.matmul(W, self.U1s[i].T) for i in range(num_terms)])
        )  # d2xk
        self.S2s = nn.Parameter(
            torch.stack([torch.matmul(self.U2s[i].T, W) for i in range(num_terms)])
        )  # kxd1

        # Bias term initialized with a small standard deviation
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            fan_in, _ = init._calculate_fan_in_and_fan_out(W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, h_in):
        return SketchedLinearFunction.apply(
            h_in, self.S1s, self.S2s, self.U1s, self.U2s, self.bias
        )


if __name__ == "__main__":
    linear = SKLinear(
        in_features=10,
        out_features=10,
        num_terms=1,
        low_rank=1,
        dtype=torch.float32,
        device=torch.device("cuda:0"),
    )
    input = torch.randn(1, 10, dtype=torch.float32, device=torch.device("cuda:0"))
    output = linear(input)
    print(output)
    print(output.shape)
