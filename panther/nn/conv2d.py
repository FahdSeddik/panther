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
        ctx: Any,
        x: torch.Tensor,
        S1s: torch.Tensor,
        S2s: torch.Tensor,
        U1s: torch.Tensor,
        U2s: torch.Tensor,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        kernel_size: Tuple[int, int],
        bias: torch.Tensor,
    ):
        B, C, H, W = x.shape
        L, _, out_channels = U1s.shape
        if padding[0] > 0 or padding[1] > 0:
            x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]))
        H_out = (x.shape[2] - kernel_size[0]) // stride[0] + 1
        W_out = (x.shape[3] - kernel_size[1]) // stride[1] + 1
        x_strided = x.as_strided(
            size=(
                x.shape[0],
                x.shape[1],
                H_out,
                W_out,
                kernel_size[0],
                kernel_size[1],
            ),
            stride=(
                x.stride(0),
                x.stride(1),
                x.stride(2) * stride[0],
                x.stride(3) * stride[1],
                x.stride(2),
                x.stride(3),
            ),
        )
        x_windows = x_strided.permute(0, 2, 3, 1, 4, 5)

        x_windows = x_windows.reshape(B, -1, kernel_size[0] * kernel_size[1] * C)

        out1 = torch.einsum("bnd,tdr,tro->bno", x_windows, S1s, U1s) / (2 * L)
        out2 = torch.einsum("bnd,tdr,tro->bno", x_windows, U2s.transpose(1, 2), S2s) / (
            2 * L
        )
        ctx.save_for_backward(
            x_windows,
            S1s,
            S2s,
            U1s,
            U2s,
            torch.tensor(stride),
            torch.tensor(padding),
            torch.tensor(kernel_size),
            torch.tensor((B, C, H, W)),
        )
        return (
            (out1 + out2 + bias).view(B, H_out, W_out, out_channels).permute(0, 3, 1, 2)
        )

    @staticmethod
    def backward(ctx: Any, *grad_output: Any) -> Any:
        input, S1s, S2s, U1s, U2s, stride, padding, kernelSize, inshape = (
            ctx.saved_tensors
        )
        num_terms, _, __ = S2s.shape
        hout = grad_output[0].shape[2]
        wout = grad_output[0].shape[3]
        g_bias = grad_output[0].sum(dim=(0, 2, 3))
        print(grad_output[0].shape)
        grad_output = grad_output[0].reshape(
            grad_output[0].shape[0],
            hout * wout,
            grad_output[0].shape[1],
        )
        grad_output = grad_output / (2 * num_terms)
        g_S1s = torch.einsum(
            "nab,nbc,lcd->lad", input.transpose(1, 2), grad_output, U1s.transpose(1, 2)
        )
        g_S2s = torch.einsum(
            "lab,nbc,ncd->lad", U2s, input.transpose(1, 2), grad_output
        )
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

        return (gout, g_S1s, g_S2s, None, None, None, None, None, g_bias)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SKConv2d layer."""
        return SketchedConv2dFunction.apply(
            x,
            self.S1s,
            self.S2s,
            self.U1s,
            self.U2s,
            self.stride,
            self.padding,
            self.kernel_size,
            self.bias,
        )
