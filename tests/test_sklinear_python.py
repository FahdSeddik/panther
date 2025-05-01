import pytest
import torch
from torch.nn import Linear

from panther.nn import SKLinear


def test_sklinear_vs_linear():
    linear = Linear(in_features=30, out_features=5)
    sklinear = SKLinear(
        in_features=30,
        out_features=5,
        num_terms=1,
        low_rank=1,
        W_init=linear.weight,
    )

    sklinear_bias = sklinear.bias.detach().numpy()
    linear_bias = linear.bias.detach().numpy()

    assert sklinear_bias.shape == linear_bias.shape, "Bias shapes do not match"
    assert sklinear_bias.dtype == linear_bias.dtype, "Bias data types do not match"

    # call the forward method of the linear layer
    input = torch.randn(1, 30)
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


def test_backward():
    with pytest.warns(
        UserWarning,
        match=SKLinear.WARNING_MSG_FC,
    ):
        sklinear = SKLinear(
            in_features=30,
            out_features=5,
            num_terms=1,
            low_rank=3,
        )

    # call the forward method of the linear layer
    input = torch.randn(1, 30)
    sklinear_output = sklinear(input)

    # call the backward method of the linear layer
    sklinear_output.backward(torch.ones_like(sklinear_output))


@pytest.mark.parametrize(
    "input_dim, output_dim, num_terms, low_rank, expectation",
    [
        (30, 10, 2, 3, pytest.warns(UserWarning, match=SKLinear.WARNING_MSG_FC)),
        (100, 200, 3, 15, pytest.warns(UserWarning, match=SKLinear.WARNING_MSG_FC)),
        (100, 200, 3, 30, pytest.warns(UserWarning, match=SKLinear.WARNING_MSG_FC)),
    ],
)
def test_network_output_variance(
    input_dim, output_dim, num_terms, low_rank, expectation, samples=100, scale_rng=0.1
):
    W = torch.randn(output_dim, input_dim) * scale_rng
    bias = torch.randn(output_dim) * scale_rng

    input = torch.randn(1, input_dim) * scale_rng
    ground_truth = input @ W.T + bias

    input = input.unsqueeze(0).expand(num_terms, input.shape[0], input.shape[1])
    output_term1s = []
    output_term2s = []
    for _ in range(samples):
        with expectation:
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
