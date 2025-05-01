import pytest
import torch
import torch.nn.functional as F

from panther.nn import SKConv2d
from panther.nn.conv2d import SketchedConv2dFunction

torch.manual_seed(42)

BATCH_SIZE = 10
IN_CHANNELS = 40
OUT_CHANNELS = 16
HEIGHT = WIDTH = 20
KERNEL_SIZE = (3, 3)
STRIDE = (1, 1)
PADDING = (1, 1)
LOW_RANK_DIM = 8
NUM_TERMS = 3


class MockCtx:
    def __init__(self, *saved_tensors):
        self.saved_tensors = saved_tensors


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
    if PADDING[0] > 0 or PADDING[1] > 0:
        x = F.pad(x, (PADDING[1], PADDING[1], PADDING[0], PADDING[0]))
    H_out = (x.shape[2] - KERNEL_SIZE[0]) // STRIDE[0] + 1
    W_out = (x.shape[3] - KERNEL_SIZE[1]) // STRIDE[1] + 1
    x_strided = x.as_strided(
        size=(
            x.shape[0],
            x.shape[1],
            H_out,
            W_out,
            KERNEL_SIZE[0],
            KERNEL_SIZE[1],
        ),
        stride=(
            x.stride(0),
            x.stride(1),
            x.stride(2) * STRIDE[0],
            x.stride(3) * STRIDE[1],
            x.stride(2),
            x.stride(3),
        ),
    )
    x_windows = x_strided.permute(0, 2, 3, 1, 4, 5)

    input_tensor = x_windows.reshape(-1, KERNEL_SIZE[0] * KERNEL_SIZE[1] * IN_CHANNELS)
    S1s = layer.S1s.clone().detach().requires_grad_(True)
    S2s = layer.S2s.clone().detach().requires_grad_(True)
    U1s = layer.U1s.clone().detach()
    U2s = layer.U2s.clone().detach()
    bias = layer.bias.clone().detach().requires_grad_(True)

    return (
        input_tensor,
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


def test_output_shape(test_tensors):
    input_tensor, S1s, S2s, U1s, U2s, bias, stride, padding, kernel_size, inshape = (
        test_tensors
    )
    output = SketchedConv2dFunction.forward(
        input_tensor, S1s, S2s, U1s, U2s, stride, padding, kernel_size, inshape, bias
    )
    H_out = (HEIGHT + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    W_out = (WIDTH + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    assert output.shape == (BATCH_SIZE, OUT_CHANNELS, H_out, W_out)


def test_forward_determinism(test_tensors):
    input_tensor, S1s, S2s, U1s, U2s, bias, stride, padding, kernel_size, inshape = (
        test_tensors
    )
    out1 = SketchedConv2dFunction.forward(
        input_tensor, S1s, S2s, U1s, U2s, stride, padding, kernel_size, inshape, bias
    )
    out2 = SketchedConv2dFunction.forward(
        input_tensor, S1s, S2s, U1s, U2s, stride, padding, kernel_size, inshape, bias
    )
    assert torch.allclose(out1, out2), "Non-deterministic forward pass."


def test_forward_gpu_vs_cpu(test_tensors):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available.")

    input_tensor, S1s, S2s, U1s, U2s, bias, stride, padding, kernel_size, inshape = (
        test_tensors
    )

    out_cpu = SketchedConv2dFunction.forward(
        input_tensor, S1s, S2s, U1s, U2s, stride, padding, kernel_size, inshape, bias
    )
    args_gpu = [x.cuda() for x in (input_tensor, S1s, S2s, U1s, U2s)]
    out_gpu = SketchedConv2dFunction.forward(
        *args_gpu, stride, padding, kernel_size, inshape, bias.cuda()
    ).cpu()
    assert torch.allclose(out_cpu, out_gpu, atol=1e-3, rtol=1e-3)


def test_backward_shapes(test_tensors):
    input_tensor, S1s, S2s, U1s, U2s, bias, stride, padding, kernel_size, inshape = (
        test_tensors
    )
    out = SketchedConv2dFunction.forward(
        input_tensor, S1s, S2s, U1s, U2s, stride, padding, kernel_size, inshape, bias
    )
    grad_output = torch.randn_like(out)
    ctx = MockCtx(
        input_tensor,
        S1s,
        S2s,
        U1s,
        U2s,
        torch.tensor(stride),
        torch.tensor(padding),
        torch.tensor(kernel_size),
        torch.tensor(inshape),
    )
    grads = SketchedConv2dFunction.backward(
        ctx,
        grad_output,
    )
    # this the input form is not the same as the grad that is correct
    assert grads[0].shape == (BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    assert grads[1].shape == S1s.shape
    assert grads[2].shape == S2s.shape
    assert grads[3].shape == bias.shape


def test_backward_vs_autograd(test_tensors):
    input_tensor, S1s, S2s, U1s, U2s, bias, stride, padding, kernel_size, inshape = (
        test_tensors
    )
    out = SketchedConv2dFunction.forward(
        input_tensor, S1s, S2s, U1s, U2s, stride, padding, kernel_size, inshape, bias
    )
    grad_output = (torch.randn_like(out),)

    autograd_grads = torch.autograd.grad(
        outputs=out,
        inputs=(input_tensor, S1s, S2s, bias),
        grad_outputs=grad_output[0],
        create_graph=True,
    )
    # named it correctly
    ctx = MockCtx(
        input_tensor,
        S1s,
        S2s,
        U1s,
        U2s,
        torch.tensor(stride),
        torch.tensor(padding),
        torch.tensor(kernel_size),
        torch.tensor(inshape),
    )
    custom_grads = SketchedConv2dFunction.backward(
        ctx,
        grad_output,
    )

    for g1, g2 in zip(custom_grads, autograd_grads):
        assert g1.shape == g2.shape
        assert torch.allclose(g1, g2, atol=1e-3, rtol=1e-3)
