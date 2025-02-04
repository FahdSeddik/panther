from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function


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
        d1, num_terms = S2s.shape[1], S2s.shape[0]
        tot = torch.zeros((d1, input.shape[1]), device=input.device)
        # Efficiently perform the sum over all l terms
        for i in range(num_terms):
            term1 = U1s[i].T.mm(S1s[i].mm(input))
            term2 = S2s[i].mm(U2s[i].mm(input))
            tot += term1 + term2
        return tot / (2 * num_terms) + bias.reshape(-1, 1)

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
        num_terms = S2s.shape[0]
        g = grad_output[0] / (2 * num_terms)

        grad = torch.zeros(input.shape, device=input.device)
        grad_S1s = torch.zeros(S1s.shape, device=input.device)
        grad_S2s = torch.zeros(S2s.shape, device=input.device)
        for i in range(num_terms):
            grad += S1s[i].T.mm(U1s[i].mm(g)) + U2s[i].T.mm(S2s[i].T.mm(g))
            grad_S2s[i] = g.mm(input.T.mm(U2s[i].T))
            grad_S1s[i] = U1s[i].mm(g).mm(input.T)

        return (
            grad / (2 * num_terms),
            grad_S1s,
            grad_S2s,
            None,
            None,
            # sum g on batch dimension input.shape[1]
            g.reshape(-1, input.shape[1]).sum(1),
        )


class SKLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_terms: int,
        low_rank: int,
        bias_init_std=0.01,
        W=None,
    ):
        super(SKLinear, self).__init__()

        if 2 * num_terms * low_rank * (output_dim + input_dim) > output_dim * input_dim:
            raise ValueError(
                "The number of parameters in the sketching layer is larger "
                + "than the number of parameters in the fully connected layer."
            )

        self.l = num_terms
        self.k = low_rank
        self.d1 = output_dim
        self.d2 = input_dim

        def generate_U(k: int, d: int) -> torch.Tensor:
            """
            Generate a random matrix U with orthonormal rows.
            """
            return (
                torch.randint(0, 2, (k, d), dtype=torch.float32) * 2 - 1
            ) / torch.sqrt(torch.tensor(k, dtype=torch.float32))

        # Register U1s and U2s as buffers since they are not learnable
        self.register_buffer(
            "U1s",
            torch.stack([generate_U(low_rank, output_dim) for _ in range(num_terms)]),
        )
        self.register_buffer(
            "U2s",
            torch.stack([generate_U(low_rank, input_dim) for _ in range(num_terms)]),
        )

        # W is used to only initialize S
        W = torch.randn(output_dim, input_dim) if W is None else W.clone().detach()

        # S1s and S2s are precomputed but not updated in the backward pass
        self.S1s = nn.Parameter(
            torch.stack([torch.matmul(self.U1s[i], W) for i in range(num_terms)])
        )
        self.S2s = nn.Parameter(
            torch.stack([torch.matmul(W, self.U2s[i].T) for i in range(num_terms)])
        )

        # Bias term initialized with a small standard deviation
        self.bias = nn.Parameter(torch.randn(output_dim) * bias_init_std)

    def forward(self, h_in):
        """
        Forward pass calculation as per the given formula:
        a = (1/(2*l)) * sum_{i=1}^{l} (U1_i^T S1_i h_in)
            + (1/(2*l)) * sum_{i=1}^{l} (S2_i U2_i h_in)
            + b
        """
        return SketchedLinearFunction.apply(
            h_in, self.S1s, self.S2s, self.U1s, self.U2s, self.bias
        )
