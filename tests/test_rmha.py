import pytest
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from panther.nn.attention import RandMultiHeadAttention, verify_rmha_inputs
from pawX import causal_denominator_apply, causal_numerator_apply


# -----------------------------------------------------------------------------
# Test 1: Check output shape and a basic comparison against torch's MultiheadAttention.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "batch_size, seq_len, embed_dim, num_heads, num_random_features, bias",
    [
        (2, 10, 32, 4, 16, True),
        (4, 20, 64, 8, 32, False),
        (1, 5, 16, 2, 8, True),
        (3, 15, 48, 6, 24, False),
    ],
)
def test_forward_output_shape_and_mha_comparison(
    batch_size, seq_len, embed_dim, num_heads, num_random_features, bias
):
    torch.manual_seed(0)

    # Instantiate the RMHA model.
    rmha = RandMultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_random_features=num_random_features,
        dropout=0.0,
        bias=bias,
        kernel_fn="softmax",
        iscausal=False,
    )
    rmha.eval()

    # Create random input tensor (batch_size, seq_len, embed_dim)
    query = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32)
    value = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32)

    # Compute RMHA output.
    perf_out, _ = rmha(query, key, value)

    # Verify the output shape.
    assert perf_out.shape == (batch_size, seq_len, embed_dim)

    # --- Basic comparison with torch.nn.MultiheadAttention ---
    # Note:
    # 1. torch.nn.MultiheadAttention expects input as (seq_len, batch_size, embed_dim)
    # 2. The RMHA module applies its own linear layers.
    # To do a rough comparison of dimensions and interface compatibility we
    # copy the linear projection weights and biases from performer to a MultiheadAttention.
    mha = nn.MultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, dropout=0.0, bias=bias
    )
    mha.eval()

    # Copy weights from RMHA to the MHA module.
    # For MultiheadAttention, the in_proj_weight is a concatenation of query, key, and value weights.
    with torch.no_grad():
        in_proj_weight = torch.cat([rmha.Wq, rmha.Wk, rmha.Wv], dim=0)
        mha.in_proj_weight.copy_(in_proj_weight)
        in_proj_bias = torch.cat([rmha.bq, rmha.bk, rmha.bv], dim=0) if bias else None
        if bias:
            mha.in_proj_bias.copy_(in_proj_bias)
        mha.out_proj.weight.copy_(rmha.W0)
        if bias:
            mha.out_proj.bias.copy_(rmha.b0)

    # Transpose input shape to match MHA: (seq_len, batch_size, embed_dim)
    query_mha = query.transpose(0, 1)
    key_mha = key.transpose(0, 1)
    value_mha = value.transpose(0, 1)

    # Compute MultiheadAttention output.
    mha_out, _ = mha(query_mha, key_mha, value_mha)
    # Bring back to (batch, seq, embed_dim) for comparison.
    mha_out = mha_out.transpose(0, 1)

    # Check that the shapes match.
    assert mha_out.shape == (batch_size, seq_len, embed_dim)
    assert perf_out.shape == mha_out.shape

    # print norm diff
    norm_diff = torch.norm(perf_out - mha_out, p=2).item()
    print(f"Norm difference between RMHA and MHA: {norm_diff:.6f}")


# -----------------------------------------------------------------------------
# Test 2: gradcheck for the custom CausalNumeratorFunction.
# -----------------------------------------------------------------------------
def test_gradcheck_causal_numerator():
    torch.manual_seed(0)
    # Set dimensions:
    # Let: D = 2, I = 3, J = 4, C = 5, E = 6.
    D, I, J, C, E = 2, 3, 4, 5, 6  # noqa: E741

    # Create input tensors with requires_grad, in double precision.
    qs = torch.randn(D, I, J, C, dtype=torch.double, requires_grad=True)
    ks = torch.randn(D, I, J, C, dtype=torch.double, requires_grad=True)
    vs = torch.randn(D, I, J, E, dtype=torch.double, requires_grad=True)

    # Run gradcheck on the custom autograd function.
    # Note: gradcheck requires the function to be run with double precision.
    assert gradcheck(causal_numerator_apply, (qs, ks, vs), eps=1e-6, atol=1e-4)


# -----------------------------------------------------------------------------
# Test 3: gradcheck for the custom CausalDenominator function.
# -----------------------------------------------------------------------------
def test_gradcheck_causal_denominator():
    torch.manual_seed(0)
    # Use dimensions: D = 2, I = 3, J = 4, C = 5.
    D, I, J, C = 2, 3, 4, 5  # noqa: E741

    qs = torch.randn(D, I, J, C, dtype=torch.double, requires_grad=True)
    ks = torch.randn(D, I, J, C, dtype=torch.double, requires_grad=True)

    assert gradcheck(causal_denominator_apply, (qs, ks), eps=1e-6, atol=1e-4)


# -----------------------------------------------------------------------------
# Test 4: Verify input validations for RMHA.
# -----------------------------------------------------------------------------
def test_verify_rmha_inputs_invalid():
    # embed_dim not divisible by num_heads.
    with pytest.raises(ValueError):
        verify_rmha_inputs(
            embed_dim=30,
            num_heads=4,
            dropout=0.0,
            bias=True,
            kernel_fn="softmax",
            iscausal=False,
        )
    # embed_dim nonpositive.
    with pytest.raises(ValueError):
        verify_rmha_inputs(
            embed_dim=0,
            num_heads=4,
            dropout=0.0,
            bias=True,
            kernel_fn="softmax",
            iscausal=False,
        )
    # num_heads nonpositive.
    with pytest.raises(ValueError):
        verify_rmha_inputs(
            embed_dim=32,
            num_heads=0,
            dropout=0.0,
            bias=True,
            kernel_fn="softmax",
            iscausal=False,
        )
    # dropout out of range.
    with pytest.raises(ValueError):
        verify_rmha_inputs(
            embed_dim=32,
            num_heads=4,
            dropout=1.5,
            bias=True,
            kernel_fn="softmax",
            iscausal=False,
        )
    with pytest.raises(ValueError):
        verify_rmha_inputs(
            embed_dim=32,
            num_heads=4,
            dropout=-0.1,
            bias=True,
            kernel_fn="softmax",
            iscausal=False,
        )
    # bias not boolean.
    with pytest.raises(ValueError):
        verify_rmha_inputs(
            embed_dim=32,
            num_heads=4,
            dropout=0.0,
            bias="yes",
            kernel_fn="softmax",
            iscausal=False,
        )
    # kernel_fn invalid.
    with pytest.raises(ValueError):
        verify_rmha_inputs(
            embed_dim=32,
            num_heads=4,
            dropout=0.0,
            bias=True,
            kernel_fn="invalid",
            iscausal=False,
        )
    # iscausal not boolean.
    with pytest.raises(ValueError):
        verify_rmha_inputs(
            embed_dim=32,
            num_heads=4,
            dropout=0.0,
            bias=True,
            kernel_fn="softmax",
            iscausal="no",
        )


# -----------------------------------------------------------------------------
# Test 5: Check that different input dimensions propagate correctly.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "batch_size, seq_len, embed_dim, num_heads",
    [
        (1, 5, 16, 4),
        (2, 10, 32, 8),
        (4, 20, 64, 8),
    ],
)
def test_forward_varying_dimensions(batch_size, seq_len, embed_dim, num_heads):
    # For simplicity, choose a number of random features equal to half the head dimension.
    num_random_features = max(2, embed_dim // (2 * num_heads))

    performer = RandMultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_random_features=num_random_features,
        dropout=0.0,
        bias=True,
        kernel_fn="relu",
        iscausal=False,
    )
    performer.eval()

    query = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32)
    value = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32)

    out, _ = performer(query, key, value)
    assert out.shape == (batch_size, seq_len, embed_dim)


# -----------------------------------------------------------------------------
# Test 6: Verify GPU and CPU execution correctness and output similarity.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "batch_size, seq_len, embed_dim, num_heads",
    [
        (2, 10, 32, 4),
        (4, 20, 64, 8),
    ],
)
def test_gpu_cpu_execution(batch_size, seq_len, embed_dim, num_heads):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available. Skipping GPU test.")

    # For simplicity, choose a number of random features equal to half the head dimension.
    num_random_features = max(2, embed_dim // (2 * num_heads))

    rmha = RandMultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_random_features=num_random_features,
        dropout=0.0,
        bias=True,
        kernel_fn="softmax",
        iscausal=False,
    )
    rmha.eval()

    # Create random input tensors.
    query = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32)
    key = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32)
    value = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32)

    # Run on CPU.
    cpu_out, _ = rmha(query, key, value)
    assert cpu_out.device.type == "cpu"

    # Move model and inputs to GPU.
    rmha = rmha.to("cuda")
    query_gpu = query.to("cuda")
    key_gpu = key.to("cuda")
    value_gpu = value.to("cuda")

    # Run on GPU.
    gpu_out, _ = rmha(query_gpu, key_gpu, value_gpu)
    assert gpu_out.device.type == "cuda"

    # Move GPU output back to CPU for comparison.
    gpu_out_cpu = gpu_out.to("cpu")

    # Verify that the outputs are similar within a tolerance.
    assert torch.allclose(
        cpu_out, gpu_out_cpu, atol=1e-5
    ), "Outputs differ between CPU and GPU execution."


if __name__ == "__main__":
    # When running this file directly, invoke pytest.
    pytest.main([__file__])
