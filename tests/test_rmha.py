import pytest
import torch
import torch.nn as nn

from panther.nn.attention import RandMultiHeadAttention, verify_rmha_inputs
from pawX import (
    causal_denominator_backward,
    causal_denominator_forward,
    causal_numerator_backward,
    causal_numerator_forward,
)


# --------------------------------------------------------------------------
# Test 1: Check output shape and a basic comparison against torch's MultiheadAttention
# with independent query (L) and key/value (S) sequence lengths.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "batch_size, len_q, len_kv, embed_dim, num_heads, num_random_features, bias, iscausal, kernel_fn",
    [
        # (batch, L,  S,  dim, heads, m_feats, bias, causal,  kernel)
        (2, 10, 10, 32, 4, 16, True, True, "softmax"),
        (4, 20, 20, 64, 8, 32, False, True, "softmax"),
        (1, 5, 5, 16, 2, 8, True, True, "softmax"),
        (3, 15, 15, 48, 6, 24, False, True, "softmax"),
        (2, 10, 8, 32, 4, 16, True, False, "softmax"),
        (4, 20, 15, 64, 8, 32, False, False, "softmax"),
        (1, 5, 5, 16, 2, 8, True, False, "softmax"),
        (3, 15, 10, 48, 6, 24, False, False, "softmax"),
        # --- relu kernel variants ---
        (2, 10, 10, 32, 4, 16, True, True, "relu"),
        (4, 20, 20, 64, 8, 32, False, True, "relu"),
        (1, 5, 5, 16, 2, 8, True, True, "relu"),
        (3, 15, 15, 48, 6, 24, False, True, "relu"),
        (2, 10, 8, 32, 4, 16, True, False, "relu"),
        (4, 20, 15, 64, 8, 32, False, False, "relu"),
        (1, 5, 5, 16, 2, 8, True, False, "relu"),
        (3, 15, 10, 48, 6, 24, False, False, "relu"),
    ],
)
def test_forward_output_shape_and_mha_comparison(
    batch_size,
    len_q,
    len_kv,
    embed_dim,
    num_heads,
    num_random_features,
    bias,
    iscausal,
    kernel_fn,
):
    torch.manual_seed(0)

    # Instantiate the RMHA model.
    rmha = RandMultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_random_features=num_random_features,
        dropout=0.0,
        bias=bias,
        kernel_fn=kernel_fn,
        iscausal=iscausal,
    )
    rmha.eval()

    # Create random input tensor
    #   query: (batch_size, len_q,  embed_dim)
    #   key:   (batch_size, len_kv, embed_dim)
    #   value: (batch_size, len_kv, embed_dim)
    query = torch.randn(batch_size, len_q, embed_dim, dtype=torch.float32)
    key = torch.randn(batch_size, len_kv, embed_dim, dtype=torch.float32)
    value = torch.randn(batch_size, len_kv, embed_dim, dtype=torch.float32)

    # Compute RMHA output.
    perf_out, _ = rmha(query, key, value)

    # Verify the output shape is (batch_size, len_q, embed_dim)
    assert perf_out.shape == (batch_size, len_q, embed_dim)

    # --- Basic comparison with torch.nn.MultiheadAttention ---
    # torch.nn.MultiheadAttention expects (L, B, D) for query, key, value
    mha = nn.MultiheadAttention(
        embed_dim=embed_dim, num_heads=num_heads, dropout=0.0, bias=bias
    )
    mha.eval()

    # Copy Performer projections into MHA
    with torch.no_grad():
        in_proj_weight = torch.cat([rmha.Wq, rmha.Wk, rmha.Wv], dim=0)
        mha.in_proj_weight.copy_(in_proj_weight)
        if bias:
            in_proj_bias = torch.cat([rmha.bq, rmha.bk, rmha.bv], dim=0)
            mha.in_proj_bias.copy_(in_proj_bias)
        mha.out_proj.weight.copy_(rmha.W0)
        if bias:
            mha.out_proj.bias.copy_(rmha.b0)

    # Transpose shapes: (len, batch, dim)
    q_mha = query.transpose(0, 1)  # (len_q, batch, dim)
    k_mha = key.transpose(0, 1)  # (len_kv, batch, dim)
    v_mha = value.transpose(0, 1)  # (len_kv, batch, dim)

    # Compute MHA output
    mha_out, _ = mha(q_mha, k_mha, v_mha)
    mha_out = mha_out.transpose(0, 1)  # back to (batch, len_q, dim)

    # Check that the shapes match
    assert mha_out.shape == perf_out.shape == (batch_size, len_q, embed_dim)


@pytest.mark.parametrize(
    "D,I,J,K,L",
    [
        (1, 1, 1, 1, 1),
        (2, 2, 2, 2, 2),
        (3, 1, 4, 5, 6),
        (3, 4, 1, 5, 6),
        (3, 4, 5, 1, 6),
        (3, 4, 5, 6, 1),
        (2, 2, 2, 2, 2),
        (3, 2, 4, 1, 5),
        (4, 3, 1, 6, 2),
        (5, 6, 7, 8, 9),
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_causal_numerator_correctness(D, I, J, K, L, device, dtype):  # noqa: E741
    torch.manual_seed(0)
    qs = torch.randn(I, D, J, K, dtype=dtype, device=device, requires_grad=True)
    ks = torch.randn(I, D, J, K, dtype=dtype, device=device, requires_grad=True)
    vs = torch.randn(I, D, J, L, dtype=dtype, device=device, requires_grad=True)

    # Reference implementation
    KV = torch.einsum("idjk,idjl->idjkl", ks, vs)  # [D,I,J,K,L]
    sums2 = torch.cumsum(KV, dim=1)  # [D,I,J,K,L]
    expected = torch.einsum("idjkl,idjk->idjl", sums2, qs)

    # Your custom forward
    out, sums = causal_numerator_forward(qs, ks, vs)
    assert out.shape == (I, D, J, L), "Output shape mismatch"
    assert torch.allclose(
        out, expected, atol=1e-6, rtol=1e-5
    ), f"Output mismatch: max|diff|={(out-expected).abs().max():.3e}"

    # backward‐gradient check
    grad_out = torch.randn_like(out)
    # autograd‐computed grads
    # custom‐backend grads
    custom_q, custom_k, custom_v = causal_numerator_backward(
        res_grad=grad_out,
        sums=sums,
        qs=qs,
        ks=ks,
        vs=vs,
    )
    auto_q, auto_k, auto_v = torch.autograd.grad(
        outputs=out,
        inputs=(qs, ks, vs),
        grad_outputs=grad_out,
        create_graph=True,
    )

    for a, c, name in [
        (auto_q, custom_q, "qs"),
        (auto_k, custom_k, "ks"),
        (auto_v, custom_v, "vs"),
    ]:
        assert a.shape == c.shape, f"grad {name} has wrong shape"
        assert torch.allclose(a, c, atol=1e-6, rtol=1e-5), (
            f"Backward gradient mismatch for {name}: "
            f"max|diff|={(a-c).abs().max():.3e}"
        )


@pytest.mark.parametrize(
    "D,I,J,K",
    [
        (1, 1, 1, 1),
        (3, 1, 4, 5),
        (3, 4, 1, 5),
        (3, 4, 5, 1),
        (2, 2, 2, 2),
        (3, 2, 4, 1),
        (4, 3, 1, 6),
        (5, 6, 7, 8),
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_causal_denominator_correctness(D, I, J, K, device, dtype):  # noqa: E741
    torch.manual_seed(0)
    # Create inputs
    qs = torch.randn(I, D, J, K, dtype=dtype, device=device, requires_grad=True)
    ks = torch.randn(I, D, J, K, dtype=dtype, device=device, requires_grad=True)

    # Reference: sums over sequence axis then contract
    sums = torch.cumsum(ks, dim=1)  # [D,I,J,K]
    expected = torch.einsum("idjk,idjk->idj", sums, qs)  # [D,I,J]

    # Custom forward
    out, _ = causal_denominator_forward(qs, ks)
    assert out.shape == (I, D, J), "Output shape mismatch"
    assert torch.allclose(
        out, expected, atol=1e-6, rtol=1e-5
    ), f"Output mismatch: max|diff|={(out-expected).abs().max():.3e}"

    # Backward: compare autograd vs custom
    grad_out = torch.randn_like(out)
    auto_q, auto_k = torch.autograd.grad(
        outputs=out,
        inputs=(qs, ks),
        grad_outputs=grad_out,
        create_graph=True,
    )
    custom_q, custom_k = causal_denominator_backward(
        res_grad=grad_out.unsqueeze(-1),
        sums=sums,
        qs=qs,
    )

    # Check shapes and values
    for name, a, c in [("qs", auto_q, custom_q), ("ks", auto_k, custom_k)]:
        assert a.shape == c.shape, f"Gradient for {name} has wrong shape"
        diff = (a - c).abs().max()
        assert torch.allclose(
            a, c, atol=1e-6, rtol=1e-5
        ), f"Backward gradient mismatch for {name}, max|diff|={diff:.3e}"


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
