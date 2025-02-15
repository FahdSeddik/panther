import torch
from torch.nn import Linear

from panther.nn import SKLinear


def test_sklinear_vs_linear():
    linear = Linear(in_features=30, out_features=5)
    sklinear = SKLinear(
        input_dim=30,
        output_dim=5,
        num_terms=1,
        low_rank=1,
        bias_init_std=0.01,
        W=linear.weight,
    )

    sklinear_bias = sklinear.bias.detach().numpy()
    linear_bias = linear.bias.detach().numpy()

    assert sklinear_bias.shape == linear_bias.shape, "Bias shapes do not match"
    assert sklinear_bias.dtype == linear_bias.dtype, "Bias data types do not match"

    # call the forward method of the linear layer
    input = torch.randn(1, 30)
    linear_output = linear(input)
    sklinear_output = sklinear(input)
    print("linear_output", linear_output.shape)
    print("sklinear_output", sklinear_output.shape)
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
    sklinear = SKLinear(
        input_dim=30,
        output_dim=5,
        num_terms=1,
        low_rank=1,
        bias_init_std=0.01,
    )

    # call the forward method of the linear layer
    input = torch.randn(1, 30)
    sklinear_output = sklinear(input)

    # call the backward method of the linear layer
    sklinear_output.backward(torch.ones_like(sklinear_output))
