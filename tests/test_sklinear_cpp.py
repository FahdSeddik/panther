from typing import Dict

import pytest
import torch

from panther.nn import SKLinear
from panther.utils.compatibility import has_tensor_core_support
from pawX import sketched_linear_backward, sketched_linear_forward

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Test configurations: dimensions are multiples of 16, with and without bias
CONFIGS = [
    # (batch_size, in_features, out_features, low_rank_dim, num_terms, bias_flag)
    (16, 1024, 768, 16, 2, True),
    (16, 768, 1024, 64, 2, False),
    (32, 1024, 2048, 128, 1, True),
    (64, 512, 512, 32, 3, False),
]

DTYPES = [torch.float32, torch.float16]

tc_comp = has_tensor_core_support()


def get_dtype_tols(dtype: torch.dtype) -> Dict[str, float]:
    if dtype == torch.float16:
        atol = 1 if tc_comp else 1e-1
        rtol = 1 if tc_comp else 1e-2
    elif dtype == torch.float32:
        atol = 1e-1 if tc_comp else 1e-4
        rtol = 1e-1 if tc_comp else 1e-3
    elif dtype == torch.float64:
        atol = 1e-5
        rtol = 1e-4
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return {"atol": atol, "rtol": rtol}


@pytest.fixture(params=DTYPES)
def dtype(request):
    return request.param


@pytest.fixture(params=CONFIGS)
def config(request):
    """Parametrized test configurations."""
    return request.param


@pytest.fixture
def test_tensors(config, dtype):
    """Fixture to provide test tensors based on configuration and dtype."""
    batch_size, in_features, out_features, low_rank_dim, num_terms, bias_flag = config
    linear = SKLinear(
        in_features=in_features,
        out_features=out_features,
        low_rank=low_rank_dim,
        num_terms=num_terms,
        bias=bias_flag,
        dtype=dtype,
    )
    # Create input and parameters with correct dtype
    input_tensor = torch.randn(batch_size, in_features, requires_grad=True, dtype=dtype)
    S1s = linear.S1s.clone().detach().to(dtype).requires_grad_(True)
    S2s = linear.S2s.clone().detach().to(dtype).requires_grad_(True)
    U1s = linear.U1s.clone().detach().to(dtype)
    U2s = linear.U2s.clone().detach().to(dtype)
    bias = None
    if bias_flag and linear.bias is not None:
        bias = linear.bias.clone().detach().to(dtype).requires_grad_(True)

    return input_tensor, S1s, S2s, U1s, U2s, bias


def test_forward_output_shape(test_tensors, config, dtype):
    """Ensure the output shape matches expectations."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported on this device.")
    input_tensor, S1s, S2s, U1s, U2s, bias = test_tensors
    batch_size, _, out_features, _, _, _ = config
    output = sketched_linear_forward(input_tensor, S1s, S2s, U1s, U2s, bias)
    assert output.shape == (
        batch_size,
        out_features,
    ), f"Expected output shape ({batch_size}, {out_features}), got {output.shape}."


def test_forward_deterministic(test_tensors, dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported on this device.")
    input_tensor, S1s, S2s, U1s, U2s, bias = test_tensors
    output1 = sketched_linear_forward(input_tensor, S1s, S2s, U1s, U2s, bias)
    output2 = sketched_linear_forward(input_tensor, S1s, S2s, U1s, U2s, bias)
    assert torch.allclose(output1, output2), "Forward pass is not deterministic."


def test_backward_output_shape(test_tensors, dtype):
    """Ensure the backward function returns gradients with expected shapes."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported on this device.")
    input_tensor, S1s, S2s, U1s, U2s, bias = test_tensors
    output = sketched_linear_forward(input_tensor, S1s, S2s, U1s, U2s, bias)
    grad_output = torch.randn_like(output)
    grads = sketched_linear_backward(
        grad_output,
        input_tensor,
        S1s,
        S2s,
        U1s,
        U2s,
        has_bias=bias is not None,
        use_gpu=False,
    )
    # Expect gradients: input, S1s, S2s, bias (if present)
    expected_grads = 4 if bias is not None else 3
    assert (
        len(grads) == expected_grads
    ), f"Expected {expected_grads} gradients, got {len(grads)}."
    # Check shapes for gradients that exist
    assert (
        grads[0].shape == input_tensor.shape
    ), "Gradient w.r.t input has incorrect shape."
    assert grads[1].shape == S1s.shape, "Gradient w.r.t S1s has incorrect shape."
    assert grads[2].shape == S2s.shape, "Gradient w.r.t S2s has incorrect shape."
    if bias is not None:
        assert grads[3].shape == bias.shape, "Gradient w.r.t bias has incorrect shape."


def test_backward_deterministic(test_tensors, dtype):
    """Ensure the backward function produces deterministic gradients."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported on this device.")
    input_tensor, S1s, S2s, U1s, U2s, bias = test_tensors
    output = sketched_linear_forward(input_tensor, S1s, S2s, U1s, U2s, bias)
    grad_output = torch.randn_like(output)
    grads1 = sketched_linear_backward(
        grad_output,
        input_tensor,
        S1s,
        S2s,
        U1s,
        U2s,
        has_bias=bias is not None,
        use_gpu=False,
    )
    grads2 = sketched_linear_backward(
        grad_output,
        input_tensor,
        S1s,
        S2s,
        U1s,
        U2s,
        has_bias=bias is not None,
        use_gpu=False,
    )
    assert all(
        torch.allclose(g1, g2) for g1, g2 in zip(grads1, grads2)
    ), "Backward pass is not deterministic."


@pytest.mark.parametrize("contiguous", [True, False])
def test_forward_gpu_vs_cpu(test_tensors, dtype, contiguous):
    """Ensure forward pass produces the same output on CPU and GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported on this device.")
    input_tensor, S1s, S2s, U1s, U2s, bias = test_tensors
    i_gpu = input_tensor.to("cuda")

    if not contiguous:
        B, in_feats = i_gpu.shape

        tmp = torch.zeros(B, 2 * in_feats, dtype=dtype, device=i_gpu.device)
        tmp[:, ::2] = i_gpu
        i_gpu = tmp[:, ::2]
        assert i_gpu.shape == (B, S1s.shape[1])
        assert not i_gpu.is_contiguous()

    S1s_gpu, S2s_gpu = S1s.to("cuda"), S2s.to("cuda")
    U1s_gpu, U2s_gpu = U1s.to("cuda"), U2s.to("cuda")
    bias_gpu = bias.to("cuda") if bias is not None else None
    out_gpu = sketched_linear_forward(
        i_gpu, S1s_gpu, S2s_gpu, U1s_gpu, U2s_gpu, bias_gpu, use_gpu=True
    ).cpu()
    out_cpu = sketched_linear_forward(
        input_tensor, S1s, S2s, U1s, U2s, bias, use_gpu=False
    )
    assert torch.allclose(
        out_cpu, out_gpu, **get_dtype_tols(dtype)
    ), "Forward pass differs between CPU and GPU."


@pytest.mark.parametrize("contiguous", [True, False])
def test_backward_gpu_vs_cpu(test_tensors, dtype, contiguous):
    """Ensure backward pass produces the same gradients on CPU and GPU."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported on this device.")
    input_tensor, S1s, S2s, U1s, U2s, bias = test_tensors

    S1s_gpu, S2s_gpu = S1s.to("cuda"), S2s.to("cuda")
    U1s_gpu, U2s_gpu = U1s.to("cuda"), U2s.to("cuda")
    grad_output_cpu = torch.randn(input_tensor.shape[0], U1s.shape[2], dtype=dtype)
    grads_cpu = sketched_linear_backward(
        grad_output_cpu,
        input_tensor,
        S1s,
        S2s,
        U1s,
        U2s,
        has_bias=bias is not None,
        use_gpu=False,
    )

    i_gpu = input_tensor.to("cuda")
    grad_output = grad_output_cpu.to("cuda")

    if not contiguous:
        B, in_feats = i_gpu.shape

        tmp = torch.zeros(B, 2 * in_feats, dtype=dtype, device=i_gpu.device)
        tmp[:, ::2] = i_gpu
        i_gpu = tmp[:, ::2]
        assert i_gpu.shape == (B, S1s.shape[1])
        assert not i_gpu.is_contiguous()
        # do same for grad_output
        B, G = grad_output.shape
        tmpg = torch.zeros(B, 2 * G, dtype=grad_output.dtype, device=grad_output.device)
        tmpg[:, ::2] = grad_output
        grad_output = tmpg[:, ::2]
        assert grad_output.shape == grad_output_cpu.shape
        assert not grad_output.is_contiguous()

    grads_gpu = sketched_linear_backward(
        grad_output,
        i_gpu,
        S1s_gpu,
        S2s_gpu,
        U1s_gpu,
        U2s_gpu,
        has_bias=bias is not None,
        use_gpu=True,
    )
    tols = get_dtype_tols(dtype)
    atol, rtol = tols["atol"], tols["rtol"]
    for i, (g_cpu, g_gpu) in enumerate(zip(grads_cpu, grads_gpu)):
        assert torch.allclose(
            g_cpu, g_gpu.cpu(), atol=atol, rtol=rtol
        ), f"Gradient mismatch at index {i}."


def test_backward_vs_autograd(test_tensors, dtype):
    """Ensure gradients computed by sketched_linear_backward match autograd."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 not supported on this device.")
    input_tensor, S1s, S2s, U1s, U2s, bias = test_tensors
    output = sketched_linear_forward(input_tensor, S1s, S2s, U1s, U2s, bias)
    grad_output = torch.randn_like(output)
    autograd_grads = torch.autograd.grad(
        outputs=output,
        inputs=(input_tensor, S1s, S2s, bias)
        if bias is not None
        else (input_tensor, S1s, S2s),
        grad_outputs=grad_output,
        create_graph=True,
    )
    custom_grads = sketched_linear_backward(
        grad_output,
        input_tensor,
        S1s,
        S2s,
        U1s,
        U2s,
        has_bias=bias is not None,
        use_gpu=False,
    )
    assert (
        len(custom_grads) == len(autograd_grads)
    ), f"Gradient length mismatch: custom={len(custom_grads)}, autograd={len(autograd_grads)}."
    for i, (c, a) in enumerate(zip(custom_grads, autograd_grads)):
        assert c.shape == a.shape, f"Gradient shape mismatch at index {i}."
        assert torch.allclose(
            a, c, **get_dtype_tols(dtype)
        ), f"Gradient values mismatch at index {i}."


def test_forward_noncontiguous_input(test_tensors, config, dtype):
    input, S1s, S2s, U1s, U2s, bias = test_tensors
    B, in_feats, out_feats, *_ = config

    # 1) allocate a buffer twice as wide
    tmp = torch.zeros(B, 2 * in_feats, dtype=dtype, device=input.device)
    # 2) scatter original into every-other column
    tmp[:, ::2] = input
    # 3) slice back to shape (B, in_feats) — this is non-contiguous
    input = tmp[:, ::2]
    assert input.shape == (B, in_feats)
    assert not input.is_contiguous()

    out = sketched_linear_forward(input, S1s, S2s, U1s, U2s, bias)
    assert out.shape == (B, out_feats)


def test_backward_noncontiguous_input(test_tensors, dtype):
    input_tensor, S1s, S2s, U1s, U2s, bias = test_tensors
    B, in_feats = input_tensor.shape

    tmp = torch.zeros(B, 2 * in_feats, dtype=dtype, device=input_tensor.device)
    tmp[:, ::2] = input_tensor
    input_tensor = tmp[:, ::2]
    assert input_tensor.shape == (B, S1s.shape[1])
    assert not input_tensor.is_contiguous()

    out = sketched_linear_forward(input_tensor, S1s, S2s, U1s, U2s, bias)
    grad_out = torch.randn_like(out)

    G = grad_out.shape[1]
    tmpg = torch.zeros(B, 2 * G, dtype=grad_out.dtype, device=grad_out.device)
    tmpg[:, ::2] = grad_out
    grad_out = tmpg[:, ::2]
    assert grad_out.shape == out.shape
    assert not grad_out.is_contiguous()

    grads = sketched_linear_backward(
        grad_out,
        input_tensor,
        S1s,
        S2s,
        U1s,
        U2s,
        has_bias=(bias is not None),
        use_gpu=False,
    )
    # Check shapes for gradients that exist
    assert grads[0].shape == input_tensor.shape
    assert grads[1].shape == S1s.shape
    assert grads[2].shape == S2s.shape
    if bias is not None:
        assert grads[3].shape == bias.shape


if __name__ == "__main__":
    pytest.main()
