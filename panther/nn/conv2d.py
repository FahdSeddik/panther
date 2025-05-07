import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

# import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import init

from panther.sketch import scaled_sign_sketch as gen_U
from pawX import sketched_conv2d_forward


def mode4_unfold(tensor: torch.Tensor) -> torch.Tensor:
    """Computes mode-4 matricization (unfolding along the last dimension)."""
    return tensor.reshape(-1, tensor.shape[-1])  # (I4, I1 * I2 * I3)


# def sketched_conv2d_forward(
#     x: torch.Tensor,
#     S1s: torch.Tensor,
#     U1s: torch.Tensor,
#     stride: Tuple[int, int],
#     padding: Tuple[int, int],
#     kernel_size: Tuple[int, int],
#     bias: torch.Tensor,
# ) -> torch.Tensor:
#     B, C, H, W = x.shape
#     L, K, D1 = U1s.shape
#     Kh, Kw = kernel_size
#     H_out = (H + 2 * padding[0] - Kh) // stride[0] + 1
#     W_out = (W + 2 * padding[1] - Kw) // stride[1] + 1
#     temp = F.conv2d(x, S1s, stride=stride, padding=padding).view(B, L, K, H_out, W_out)
#     out = torch.einsum("blkhw,lko->blhwo", temp, U1s).mean(dim=1) + bias
#     return out.view(B, H_out, W_out, D1).permute(0, 3, 1, 2)


class SketchedConv2dFunction(Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        S1s: torch.Tensor,
        U1s: torch.Tensor,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        kernel_size: Tuple[int, int],
        bias: torch.Tensor,
    ):
        return sketched_conv2d_forward(x, S1s, U1s, stride, padding, kernel_size, bias)


class SKConv2d(nn.Module):
    __constants__ = ["in_features", "out_features", "num_terms", "low_rank"]
    in_features: int
    out_features: int
    num_terms: int
    low_rank: int
    S1s: torch.Tensor
    U1s: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple = (3, 3),
        stride: Tuple = (1, 1),
        padding: str | Tuple | int = (1, 1),
        num_terms: int = 6,
        low_rank: int = 8,
        layer: Optional[nn.Conv2d] = None,
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
        if isinstance(padding, str):
            if padding == "same":
                self.padding = (
                    (kernel_size[0] - 1) // 2,
                    (kernel_size[1] - 1) // 2,
                )
            elif padding == "valid":
                self.padding = (0, 0)
            else:
                raise ValueError(
                    f"Invalid padding type: {padding}. Use 'same', 'valid', or a tuple."
                )
        elif isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple):
            self.padding = padding
        else:
            raise ValueError(
                f"Invalid padding type: {padding}. Use 'same', 'valid', or a tuple."
            )

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
                    for _ in range(num_terms * 2)
                ]
            ),
        )  # kxd1

        if layer is None:
            kernels = torch.empty(
                (in_channels, *self.kernel_size, out_channels), **factory_kwargs
            )
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
            fan_in, _ = init._calculate_fan_in_and_fan_out(kernels)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        elif isinstance(layer, nn.Conv2d):
            assert layer.groups == 1, "Groups must be 1 for SKConv2d"
            assert layer.dilation == (1, 1), "Dilation must be (1, 1) for SKConv2d"
            kernels = layer.weight.data.clone().detach()

            if layer.transposed:
                kernels = kernels.permute(0, 2, 3, 1)
            else:
                kernels = kernels.permute(1, 2, 3, 0)

            if layer.bias is not None:
                self.bias = nn.Parameter(layer.bias.data.clone().detach())
            else:
                self.register_parameter("bias", None)
        else:
            raise ValueError(
                "Layer must be a torch.nn.Conv2d layer or None. If None, a new kernel will be created."
            )

        init.kaiming_uniform_(kernels, a=math.sqrt(5))

        def mode4_unfold(tensor: torch.Tensor) -> torch.Tensor:
            """Computes mode-4 matricization (unfolding along the last dimension)."""
            return tensor.reshape(-1, tensor.shape[-1])  # (I4, I1 * I2 * I3)

        self.S1s = nn.Parameter(
            torch.stack(
                [
                    mode4_unfold(torch.matmul(kernels, self.U1s[i].T))
                    for i in range(num_terms * 2)
                ]
            )
            .permute(0, 2, 1)
            .reshape(
                self.num_terms * 2 * self.low_rank,
                self.in_channels,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        )  # d2xk

    @staticmethod
    def fromTorch(layer: nn.Conv2d, num_terms: int, low_rank: int) -> "SKConv2d":
        assert isinstance(layer, nn.Conv2d), "Layer must be a torch.nn.Conv2d layer"
        return SKConv2d(
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            num_terms=num_terms,
            low_rank=low_rank,
            layer=layer,
            dtype=layer.weight.dtype,
            device=layer.weight.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SKConv2d layer."""
        return SketchedConv2dFunction.apply(
            x,
            self.S1s,
            self.U1s,
            self.stride,
            self.padding,
            self.kernel_size,
            self.bias,
        )
