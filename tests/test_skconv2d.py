import pytest
import torch
import torch.nn.functional as F

from panther.nn import SKConv2d
from pawX import sketched_conv2d_backward, sketched_conv2d_forward

torch.manual_seed(42)

BATCH_SIZE = 1
IN_CHANNELS = 1
OUT_CHANNELS = 4
HEIGHT = WIDTH = 10
KERNEL_SIZE = (3, 3)
STRIDE = (1, 1)
PADDING = (1, 1)
LOW_RANK_DIM = 2
NUM_TERMS = 3


class MockCtx:
    def __init__(self, *saved_tensors):
        self.saved_tensors = saved_tensors


def strid_tensor(x: torch.Tensor, kernel_size, stride, padding) -> torch.Tensor:
    B, C, H, W = x.shape
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
    return x_windows


@pytest.fixture
def test_tensors():
    layer = SKConv2d(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        low_rank=LOW_RANK_DIM,
        num_terms=NUM_TERMS,
        stride=STRIDE,
        padding=PADDING,
    )

    x = torch.randn(
        BATCH_SIZE,
        IN_CHANNELS,
        HEIGHT,
        WIDTH,
        requires_grad=True,
    )
    S1s = layer.S1s.clone().detach().requires_grad_(True)
    S2s = layer.S2s.clone().detach().requires_grad_(True)
    U1s = layer.U1s.clone().detach()
    U2s = layer.U2s.clone().detach()
    bias = layer.bias.clone().detach().requires_grad_(True)

    return (
        x,
        S1s,
        S2s,
        U1s,
        U2s,
        bias,
        STRIDE,
        PADDING,
        KERNEL_SIZE,
        (BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH),
    )


@pytest.mark.parametrize(
    "batch_size, in_channels, out_channels, height, width, kernel_size, stride, padding, low_rank_dim, num_terms",
    [
        (1, 1, 4, 10, 10, (3, 3), (1, 1), (1, 1), 2, 3),
        (2, 3, 8, 20, 20, (5, 5), (2, 2), (2, 2), 4, 5),
        (1, 1, 8, 10, 20, (3, 3), (1, 1), (1, 1), 2, 5),
        (2, 3, 4, 20, 10, (5, 5), (2, 2), (1, 1), 4, 3),
        (1, 3, 4, 10, 10, (3, 3), (1, 1), (2, 2), 2, 3),
    ],
)
def test_output_correctness(
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_size,
    stride,
    padding,
    low_rank_dim,
    num_terms,
):
    """Test the output correctness of the SKConv2d layer."""
    layer = SKConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        low_rank=low_rank_dim,
        num_terms=num_terms,
        stride=stride,
        padding=padding,
    )
    x = torch.randn(batch_size, in_channels, height, width, requires_grad=True)
    outputmodel = layer(x)
    S1s = (
        layer.S1s.clone()
        .detach()
        .reshape(num_terms, in_channels, kernel_size[0], kernel_size[1], low_rank_dim)
    )
    S2s = (
        layer.S2s.clone()
        .detach()
        .reshape(num_terms, low_rank_dim, kernel_size[0], kernel_size[1], out_channels)
    )
    U1s = layer.U1s.clone().detach()
    U2s = layer.U2s.clone().detach()
    bias = layer.bias.clone().detach()
    # S1s x4 U1s in code
    mat1 = torch.einsum("iabcd,ide->iabce", S1s, U1s)
    mat2 = []
    for i in range(num_terms):
        S2i_flat = S2s[i].view(-1, out_channels)
        proj = torch.matmul(U2s[i].T, S2i_flat)
        proj = proj.view(in_channels, kernel_size[0], kernel_size[1], out_channels)
        mat2.append(proj)
    mat2 = torch.stack(
        mat2, dim=0
    )  # Shape: (num_terms, in_channels, height, width, out_channels)
    mat1 = mat1.permute(
        0, 4, 1, 2, 3
    )  # Shape: (num_terms, out_channels, in_channels, height, width)
    mat2 = mat2.permute(
        0, 4, 1, 2, 3
    )  # Shape: (num_terms, out_channels, in_channels, height, width)
    outputtruth = torch.zeros_like(outputmodel)
    for i in range(num_terms):
        outputtruth += F.conv2d(
            x,
            mat1[i],
            bias=None,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
        )
    for i in range(num_terms):
        outputtruth += F.conv2d(
            x,
            mat2[i],
            bias=None,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
        )
    outputtruth /= num_terms * 2
    outputtruth += bias.view(1, out_channels, 1, 1)
    print(f"Output model: {outputmodel.shape}, Output truth: {outputtruth.shape}")
    assert torch.allclose(
        outputmodel, outputtruth, atol=1e-3, rtol=1e-3
    ), "Output mismatch."


def test_output_shape(test_tensors):
    input_tensor, S1s, S2s, U1s, U2s, bias, stride, padding, kernel_size, inshape = (
        test_tensors
    )
    output, _ = sketched_conv2d_forward(
        input_tensor, S1s, S2s, U1s, U2s, stride, padding, kernel_size, bias
    )
    H_out = (HEIGHT + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    W_out = (WIDTH + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    assert output.shape == (BATCH_SIZE, OUT_CHANNELS, H_out, W_out)


def test_forward_determinism(test_tensors):
    input_tensor, S1s, S2s, U1s, U2s, bias, stride, padding, kernel_size, inshape = (
        test_tensors
    )
    out1, _ = sketched_conv2d_forward(
        input_tensor, S1s, S2s, U1s, U2s, stride, padding, kernel_size, bias
    )
    out2, _ = sketched_conv2d_forward(
        input_tensor, S1s, S2s, U1s, U2s, stride, padding, kernel_size, bias
    )
    assert torch.allclose(out1, out2), "Non-deterministic forward pass."


def test_forward_gpu_vs_cpu(test_tensors):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")

    input_tensor, S1s, S2s, U1s, U2s, bias, stride, padding, kernel_size, inshape = (
        test_tensors
    )

    out_cpu, _ = sketched_conv2d_forward(
        input_tensor, S1s, S2s, U1s, U2s, stride, padding, kernel_size, bias
    )
    args_gpu = [x.cuda() for x in (input_tensor, S1s, S2s, U1s, U2s)]
    out_gpu, _ = sketched_conv2d_forward(
        *args_gpu, stride, padding, kernel_size, bias.cuda()
    )
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-3, rtol=1e-3)


def test_backward_vs_autograd(test_tensors):
    input_tensor, S1s, S2s, U1s, U2s, bias, stride, padding, kernel_size, inshape = (
        test_tensors
    )

    # Ensure that the input tensors require gradients
    input_tensor.requires_grad_()
    S1s.requires_grad_()
    S2s.requires_grad_()
    U1s.requires_grad_()
    U2s.requires_grad_()
    bias.requires_grad_()

    # Forward pass through the custom function
    out, x_windows = sketched_conv2d_forward(
        input_tensor, S1s, S2s, U1s, U2s, stride, padding, kernel_size, bias
    )

    grad_output = torch.randn_like(out)  # Gradient of the output (random for testing)

    # Autograd gradients
    autograd_grads = torch.autograd.grad(
        outputs=out,
        inputs=(
            input_tensor,
            S1s,
            S2s,
            bias,
        ),
        grad_outputs=grad_output,
        create_graph=True,
    )

    # Custom backward pass through the function
    custom_grads = sketched_conv2d_backward(
        x_windows,
        S1s,
        S2s,
        U1s,
        U2s,
        stride,
        padding,
        kernel_size,
        inshape[2:],
        grad_output,
    )
    # Compare autograd and custom gradients for correctness
    for g1, g2 in zip(custom_grads, autograd_grads):
        assert g1.shape == g2.shape, f"Shape mismatch: {g1.shape} vs {g2.shape}"
        assert torch.allclose(
            g1, g2, atol=1e-3, rtol=1e-3
        ), f"Gradients are not close: {g1} vs {g2}"
