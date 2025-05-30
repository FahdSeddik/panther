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
    """
    Unfolds a 4D tensor along its last (4th) mode, returning its mode-4 matricization.

    Args:
        tensor (torch.Tensor): A 4-dimensional input tensor of shape (I1, I2, I3, I4).

    Returns:
        torch.Tensor: A 2D tensor of shape (I1 * I2 * I3, I4), where the last mode (I4) is preserved as columns,
                      and the product of the other dimensions forms the rows.

    Example:
        >>> x = torch.randn(2, 3, 4, 5)
        >>> y = mode4_unfold(x)
        >>> y.shape
        torch.Size([24, 5])
    """
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
    """
    A low-rank, sketched convolutional 2D layer.

    SKConv2d approximates a standard 2D convolution using a low-rank decomposition and randomized sketching.
    This reduces the number of parameters and computation, making it suitable for efficient deep learning models.

    The layer can be initialized from scratch or from an existing `torch.nn.Conv2d` layer. It uses a set of
    learnable parameters (`S1s`) and fixed sketch matrices (`U1s`) to represent the convolutional kernel.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Tuple[int, int], optional): Size of the convolutional kernel. Defaults to (3, 3).
        stride (Tuple[int, int], optional): Stride of the convolution. Defaults to (1, 1).
        padding (str | Tuple[int, int] | int, optional): Padding type or size. Can be 'same', 'valid', a tuple, or an integer. Defaults to (1, 1).
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        num_terms (int, optional): Number of terms for the low-rank decomposition. Defaults to 6.
        low_rank (int, optional): Rank for the low-rank decomposition. Defaults to 8.
        layer (Optional[nn.Conv2d], optional): Existing Conv2d layer to initialize from. If None, a new kernel is created. Defaults to None.
        dtype (optional): Data type of the parameters.
        device (optional): Device on which to allocate the parameters.

    Attributes:
        num_terms (int): Number of terms in the decomposition.
        low_rank (int): Rank of the decomposition.
        out_channels (int): Number of output channels.
        in_channels (int): Number of input channels.
        stride (Tuple[int, int]): Stride of the convolution.
        padding (Tuple[int, int]): Padding of the convolution.
        kernel_size (Tuple[int, int]): Size of the convolutional kernel.
        S1s (nn.Parameter): Learnable low-rank convolutional weights.
        U1s (torch.Tensor): Fixed sketch matrices.
        bias (nn.Parameter or None): Optional learnable bias.

    Methods:
        fromTorch(layer, num_terms, low_rank): Creates an SKConv2d from a torch.nn.Conv2d layer.
        forward(x): Forward pass of the layer.

    Notes:
        - The layer uses a randomized sketching technique for efficient approximation.
        - If an existing Conv2d layer is provided, its weights and bias are cloned.
        - The layer registers buffers and parameters for the low-rank decomposition.
    """

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
        bias: bool = True,
        num_terms: int = 6,
        low_rank: int = 8,
        layer: Optional[nn.Conv2d] = None,
        dtype=None,
        device=None,
    ):
        """
        Initializes the SKConv2d layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (Tuple, optional): Size of the convolutional kernel. Defaults to (3, 3).
            stride (Tuple, optional): Stride of the convolution. Defaults to (1, 1).
            padding (str | Tuple | int, optional): Padding type or size. Can be 'same', 'valid', a tuple, or an integer. Defaults to (1, 1).
            bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
            num_terms (int, optional): Number of terms for the low-rank decomposition. Defaults to 6.
            low_rank (int, optional): Rank for the low-rank decomposition. Defaults to 8.
            layer (Optional[nn.Conv2d], optional): Existing Conv2d layer to initialize from. If None, a new kernel is created. Defaults to None.
            dtype (optional): Data type of the parameters.
            device (optional): Device on which to allocate the parameters.

        Raises:
            ValueError: If an invalid padding type is provided.
            ValueError: If the provided layer is not a torch.nn.Conv2d instance or None.
            AssertionError: If the provided Conv2d layer has groups != 1 or dilation != (1, 1).

        Notes:
            - The SKConv2d layer uses a low-rank decomposition for the convolutional kernel.
            - If an existing Conv2d layer is provided, its weights and bias are cloned.
            - The layer registers buffers and parameters for the low-rank decomposition.
        """
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
            init.kaiming_uniform_(kernels, a=math.sqrt(5))

            if bias:
                self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
                fan_in, _ = init._calculate_fan_in_and_fan_out(kernels)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.bias, -bound, bound)
            else:
                self.register_parameter("bias", None)

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
        """
        Creates an SKConv2d instance from a given torch.nn.Conv2d layer.

        Args:
            layer (nn.Conv2d): The source PyTorch Conv2d layer to convert.
            num_terms (int): Number of terms for the SKConv2d decomposition.
            low_rank (int): Low-rank approximation parameter for SKConv2d.

        Returns:
            SKConv2d: An instance of SKConv2d initialized with parameters from the given Conv2d layer.

        Raises:
            AssertionError: If the provided layer is not an instance of torch.nn.Conv2d.
        """
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
        return sketched_conv2d_forward(
            x,
            self.S1s,
            self.U1s,
            self.stride,
            self.padding,
            self.kernel_size,
            self.bias,
        )
