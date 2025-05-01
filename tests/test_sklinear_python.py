import warnings
from contextlib import nullcontext as does_not_raise

import pytest
import torch
from torch.nn import Linear

from panther.nn import SKLinear
from panther.utils.compatibility import has_tensor_core_support


def test_sklinear_vs_linear():
    linear = Linear(in_features=32, out_features=16)
    # Capture FC warning during SKLinear initialization
    with pytest.warns(UserWarning, match=SKLinear.WARNING_MSG_FC):
        sklinear = SKLinear(
            in_features=32,
            out_features=16,
            num_terms=1,
            low_rank=16,
            W_init=linear.weight,
        )

    sklinear_bias = sklinear.bias.detach().numpy()
    linear_bias = linear.bias.detach().numpy()

    assert sklinear_bias.shape == linear_bias.shape, "Bias shapes do not match"
    assert sklinear_bias.dtype == linear_bias.dtype, "Bias data types do not match"

    # call the forward method of the linear layer
    input = torch.randn(16, 32)
    linear_output = linear(input)
    sklinear_output = sklinear(input)

    assert (
        sklinear_output.reshape(-1, 1).shape == linear_output.reshape(-1, 1).shape
    ), "Output shapes do not match"
    assert (
        sklinear_output.dtype == linear_output.dtype
    ), "Output data types do not match"

    # check that the outputs are not equal
    assert not torch.allclose(sklinear_output, linear_output), "Outputs are equal"
    assert not torch.allclose(sklinear_output, linear_output), "Outputs are equal"


def test_backward_warnings_and_grad():
    input_data = torch.randn(1, 30)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", UserWarning)
        sklinear = SKLinear(
            in_features=30,
            out_features=5,
            num_terms=1,
            low_rank=3,
        )
        _ = sklinear(input_data)

    # Assert baseline FC warning
    msgs = [str(w.message) for w in record]
    assert any(SKLinear.WARNING_MSG_FC in m for m in msgs), "FC warning not raised"
    # Assert BATCH warning if supported and batch size not divisible by 16
    assert not any(
        SKLinear.WARNING_MSG_BATCH in m for m in msgs
    ), "Unexpected batch warning"

    # Grad check
    out = sklinear(torch.randn(16, 30))
    out.backward(torch.ones_like(out))


@pytest.mark.parametrize(
    "in_f,out_f,num_terms,low_rank,batch_size",
    [
        (30, 32, 1, 16, 16),  # TC in_f
        (32, 30, 1, 16, 16),  # TC out_f
        (32, 32, 1, 16, 16),  # no TC or BATCH
        (32, 32, 1, 16, 15),  # BATCH
        (32, 32, 1, 16, 16),  # no warnings
    ],
)
def test_sklinear_tc_and_batch(in_f, out_f, num_terms, low_rank, batch_size):
    can_tc = has_tensor_core_support()
    device = torch.device("cuda" if can_tc else "cpu")
    # Prepare input
    input_data = torch.randn(batch_size, in_f, device=device)

    # Capture warnings during init and forward
    with pytest.warns(UserWarning) as record:
        sk = SKLinear(
            in_features=in_f,
            out_features=out_f,
            num_terms=num_terms,
            low_rank=low_rank,
            device=device,
            dtype=torch.float32,
        ).to(device)
        _ = sk(input_data)

    # Base FC warning always
    expected = 1
    # TC warning if supported and any dim not divisible by 16 (excluding num_terms)
    tc_cond = ((in_f % 16 != 0) or (out_f % 16 != 0) or (low_rank % 16 != 0)) and can_tc
    # Batch warning if supported and batch size not divisible by 16
    batch_cond = (batch_size % 16 != 0) and can_tc
    expected += tc_cond + batch_cond

    assert len(record) == expected, f"Expected {expected} warnings, got {len(record)}"

    msgs = [str(w.message) for w in record]
    assert any(SKLinear.WARNING_MSG_FC in m for m in msgs)
    if tc_cond:
        assert any(SKLinear.WARNING_MSG_TC in m for m in msgs)
    if batch_cond:
        assert any(SKLinear.WARNING_MSG_BATCH in m for m in msgs)


@pytest.mark.parametrize(
    "input_dim, output_dim, num_terms, low_rank, samples, scale_rng",
    [
        (128, 64, 1, 16, 100, 0.1),
        (128, 256, 2, 16, 100, 0.1),
        (512, 768, 3, 32, 100, 0.1),
    ],
)
def test_network_output_variance(
    input_dim, output_dim, num_terms, low_rank, samples, scale_rng
):
    W = torch.randn(output_dim, input_dim) * scale_rng
    bias = torch.randn(output_dim) * scale_rng

    # determine context for TC warning
    tc_ctx = (
        pytest.warns(UserWarning, match=SKLinear.WARNING_MSG_TC)
        if has_tensor_core_support()
        else does_not_raise()
    )

    input = torch.randn(1, input_dim) * scale_rng
    ground_truth = input @ W.T + bias

    input = input.unsqueeze(0).expand(num_terms, input.shape[0], input.shape[1])
    output_term1s = []
    output_term2s = []
    for _ in range(samples):
        with tc_ctx:
            sklinear = SKLinear(
                in_features=input_dim,
                out_features=output_dim,
                num_terms=num_terms,
                low_rank=low_rank,
                W_init=W,
            )
        output_term1s.append(
            ((input.bmm(sklinear.S1s)).bmm(sklinear.U1s)).mean(0) + bias
        )
        output_term2s.append(
            ((input.bmm(sklinear.U2s)).bmm(sklinear.S2s)).mean(0) + bias
        )
    # E[(X - Y)^2] = E[X^2] + E[Y^2] - 2E[XY] = E[X^2] + Y^2 - 2Y^2 = E[X^2] - Y^2
    # outputterms^2 then take the mean
    variance_term1 = (
        (torch.stack(output_term1s) ** 2).mean(0) - ground_truth**2
    ).mean()
    variance_term2 = (
        (torch.stack(output_term2s) ** 2).mean(0) - ground_truth**2
    ).mean()

    variance_bound_term1 = 2 * output_dim * torch.norm(input @ W.T) ** 2 / low_rank
    variance_bound_term2 = (
        2 * torch.norm(W, p="fro") ** 2 * torch.norm(input) ** 2 / low_rank
    )

    assert variance_term1 <= variance_bound_term1
    assert variance_term2 <= variance_bound_term2
