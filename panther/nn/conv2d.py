import math
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn import init

from panther.sketch import scaled_sign_sketch as gen_U


def mode4_unfold(tensor: torch.Tensor) -> torch.Tensor:
    """Computes mode-4 matricization (unfolding along the last dimension)."""
    return tensor.reshape(-1, tensor.shape[-1])  # (I4, I1 * I2 * I3)


class SketchedConv2dFunction(Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(
        input: torch.Tensor,
        S1s: torch.Tensor,
        S2s: torch.Tensor,
        U1s: torch.Tensor,
        U2s: torch.Tensor,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        kernelSize: Tuple[int, int],
        inshape,
        bias: torch.Tensor,
    ):
        # in_channels, height, width = input.shape
        _, dout = U1s[0].shape
        hout = (inshape[2] + 2 * padding[0] - kernelSize[0]) // stride[0] + 1
        wout = (inshape[3] + 2 * padding[1] - kernelSize[1]) // stride[1] + 1
        input.transpose_(1, 2)
        t = (
            torch.einsum("nab,lbc,lcd->nlad", input, S1s, U1s)
            + torch.einsum("nab,lbc,lcd->nlad", input, U2s.transpose(1, 2), S2s)
        ).mean(dim=1)
        t = t.view(inshape[0], dout, hout, wout)
        return t + bias.view(1, dout, 1, 1)

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any):
        input, S1s, S2s, U1s, U2s, stride, padding, kernelSize, inshap, bias = inputs
        ctx.save_for_backward(
            input,
            S1s,
            S2s,
            U1s,
            U2s,
            torch.tensor(stride),
            torch.tensor(padding),
            torch.tensor(kernelSize),
            torch.tensor(inshap),
            bias,
        )

    @staticmethod
    def backward(ctx: Any, *grad_output: Any) -> Any:
        input, S1s, S2s, U1s, U2s, stride, padding, kernelSize, inshape, bias = (
            ctx.saved_tensors
        )
        input.transpose_(1, 2)
        num_terms, _, __ = S2s.shape
        hout = grad_output[0].shape[2]
        wout = grad_output[0].shape[3]
        g_bias = grad_output[0].sum(dim=(0, 2, 3))
        grad_output = grad_output[0].view(
            grad_output[0].shape[0],
            hout * wout,
            grad_output[0].shape[1],
        )
        grad_output /= 2 * num_terms
        g_S1s = torch.zeros_like(S1s)
        g_S2s = torch.zeros_like(S2s)
        g_S1s = torch.einsum(
            "nab,nbc,lcd->lad", input, grad_output, U1s.transpose(1, 2)
        )
        g_S2s = torch.einsum("lab,nbc,ncd->lad", U2s, input, grad_output)
        gout = torch.einsum(
            "nab,lbc,lcd->nad", grad_output, U1s.transpose(1, 2), S1s.transpose(1, 2)
        ) + torch.einsum("nab,lbc,lcd->nad", grad_output, S2s.transpose(1, 2), U2s)
        fold = nn.Fold(
            output_size=(inshape[2], inshape[3]),
            kernel_size=(kernelSize[0], kernelSize[1]),
            stride=stride,
            padding=padding,
        )
        gout = gout.transpose(1, 2)
        gout = fold(gout)

        return (gout, g_S1s, g_S2s, None, None, None, None, None, None, g_bias)


class SKConv2d(nn.Module):
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
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple = (3, 3),
        stride: Tuple = (1, 1),
        padding: Tuple = (1, 1),
        num_terms: int = 6,
        low_rank: int = 8,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super(SKConv2d, self).__init__()
        self.num_terms = num_terms
        self.low_rank = low_rank
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.register_buffer(
            "U1s",
            torch.stack(
                [
                    gen_U(low_rank, out_channels, **factory_kwargs)
                    for _ in range(num_terms)
                ]
            ),
        )  # kxd1
        self.register_buffer(
            "U2s",
            torch.stack(
                [
                    gen_U(
                        low_rank * self.kernel_size[0] * self.kernel_size[1],
                        in_channels * self.kernel_size[0] * self.kernel_size[1],
                        **factory_kwargs,
                    )
                    for _ in range(num_terms)
                ]
            ),
        )  # k h w x d2 h w
        kernels = nn.Parameter(
            torch.empty(
                (in_channels, *self.kernel_size, out_channels), **factory_kwargs
            )
        )  # doutxdinxhxw
        init.kaiming_uniform_(kernels, a=math.sqrt(5))
        self.S1s = nn.Parameter(
            torch.stack(
                [
                    mode4_unfold(torch.matmul(kernels, self.U1s[i].T))
                    for i in range(num_terms)
                ]
            )
        )  # d2xk
        K_mat4 = kernels.view(
            in_channels * self.kernel_size[0] * self.kernel_size[1], out_channels
        )
        self.S2s = nn.Parameter(
            torch.stack(
                [
                    mode4_unfold(
                        torch.matmul(self.U2s[i], K_mat4).view(
                            low_rank, *self.kernel_size, out_channels
                        )
                    )
                    for i in range(num_terms)
                ]
            )
        )  #
        self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        fan_in, _ = init._calculate_fan_in_and_fan_out(kernels)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

        # Register U1s and U2s as buffers since they are not learnable

    def forward(self, x):
        """Forward pass of the SKConv2d layer."""
        # padd x
        B, C, H, W = x.shape
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = F.pad(
                x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
            )
        H_out = (x.shape[2] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (x.shape[3] - self.kernel_size[1]) // self.stride[1] + 1
        x_strided = x.as_strided(
            size=(
                x.shape[0],
                x.shape[1],
                H_out,
                W_out,
                self.kernel_size[0],
                self.kernel_size[1],
            ),
            stride=(
                x.stride(0),
                x.stride(1),
                x.stride(2) * self.stride[0],
                x.stride(3) * self.stride[1],
                x.stride(2),
                x.stride(3),
            ),
        )
        x_windows = x_strided.permute(0, 2, 3, 1, 4, 5)

        x_windows = x_windows.reshape(
            -1, self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        )
        out1 = (
            torch.einsum("nd,tdr,tro->no", x_windows, self.S1s, self.U1s)
            / self.num_terms
        ) + self.bias
        out2 = (
            torch.einsum(
                "nd,tdr,tro->no", x_windows, self.U2s.transpose(1, 2), self.S2s
            )
            / self.num_terms
        )
        return (
            (out1 + out2 + self.bias)
            .view(B, H_out, W_out, self.out_channels)
            .permute(0, 3, 1, 2)
        )
