import math
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import init

from panther.sketch import scaled_sign_sketch as gen_U
from pawX import sketched_conv2d_backward, sketched_conv2d_forward


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
        out, x_windows = sketched_conv2d_forward(
            x, S1s, S2s, U1s, U2s, stride, padding, kernel_size, bias
        )
        ctx.save_for_backward(
            x_windows,
            S1s,
            S2s,
            U1s,
            U2s,
            torch.tensor(stride, dtype=torch.int64),
            torch.tensor(padding, dtype=torch.int64),
            torch.tensor(kernel_size, dtype=torch.int64),
            torch.tensor([x.shape[2], x.shape[3]], dtype=torch.int64),
        )
        return out

    @staticmethod
    def backward(ctx: Any, *grad_output: Any) -> Any:
        input, S1s, S2s, U1s, U2s, stride, padding, kernelSize, inshape = (
            ctx.saved_tensors
        )
        gout, g_S1s, g_S2s, g_bias = sketched_conv2d_backward(
            input,
            S1s,
            S2s,
            U1s,
            U2s,
            (stride[0].item(), stride[1].item()),
            (padding[0].item(), padding[1].item()),
            (kernelSize[0].item(), kernelSize[1].item()),
            (inshape[0].item(), inshape[1].item()),
            grad_output[0],
        )
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

        def mode4_unfold(tensor: torch.Tensor) -> torch.Tensor:
            """Computes mode-4 matricization (unfolding along the last dimension)."""
            return tensor.reshape(-1, tensor.shape[-1])  # (I4, I1 * I2 * I3)

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
