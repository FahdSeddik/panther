import pytest
import torch

from pawX import sketched_linear_backward, sketched_linear_forward

# Set random seed for reproducibility
torch.manual_seed(42)

# Test parameters
BATCH_SIZE = 16
FEATURE_DIM = 32
NUM_TERMS = 4


@pytest.fixture
def test_tensors():
    """Fixture to provide test tensors with reproducible random values."""
    input_tensor = torch.randn(
        BATCH_SIZE, FEATURE_DIM, dtype=torch.double, requires_grad=True
    )
    S1s = torch.randn(
        NUM_TERMS, FEATURE_DIM, FEATURE_DIM, dtype=torch.double, requires_grad=True
    )
    S2s = torch.randn(
        NUM_TERMS, FEATURE_DIM, FEATURE_DIM, dtype=torch.double, requires_grad=True
    )
    U1s = torch.randn(
        NUM_TERMS, FEATURE_DIM, FEATURE_DIM, dtype=torch.double, requires_grad=False
    )
    U2s = torch.randn(
        NUM_TERMS, FEATURE_DIM, FEATURE_DIM, dtype=torch.double, requires_grad=False
    )
    bias = torch.randn(FEATURE_DIM, dtype=torch.double, requires_grad=True)

    return input_tensor, S1s, S2s, U1s, U2s, bias


def test_forward_output_shape(test_tensors):
    """Ensure the output shape matches expectations."""
    input_tensor, S1s, S2s, U1s, U2s, bias = test_tensors
    output = sketched_linear_forward(input_tensor, S1s, S2s, U1s, U2s, bias)

    assert output.shape == (BATCH_SIZE, FEATURE_DIM), "Output shape mismatch."


def test_forward_deterministic(test_tensors):
    """Ensure the forward pass produces deterministic results."""
    input_tensor, S1s, S2s, U1s, U2s, bias = test_tensors
    output1 = sketched_linear_forward(input_tensor, S1s, S2s, U1s, U2s, bias)
    output2 = sketched_linear_forward(input_tensor, S1s, S2s, U1s, U2s, bias)

    assert torch.allclose(output1, output2), "Forward pass is not deterministic."


def test_backward_output_shape(test_tensors):
    """Ensure the backward function returns gradients with expected shapes."""
    input_tensor, S1s, S2s, U1s, U2s, bias = test_tensors
    output = sketched_linear_forward(input_tensor, S1s, S2s, U1s, U2s, bias)

    # Compute gradient w.r.t output
    grad_output = torch.randn_like(output)

    # Call backward function
    grads = sketched_linear_backward(grad_output, input_tensor, S1s, S2s, U1s, U2s)

    assert len(grads) == 4, "Backward function must return 4 gradients."
    assert (
        grads[0].shape == input_tensor.shape
    ), "Gradient w.r.t input has incorrect shape."
    assert grads[1].shape == S1s.shape, "Gradient w.r.t S1s has incorrect shape."
    assert grads[2].shape == S2s.shape, "Gradient w.r.t S2s has incorrect shape."
    assert grads[3].shape == bias.shape, "Gradient w.r.t bias has incorrect shape."


def test_backward_deterministic(test_tensors):
    """Ensure the backward function produces deterministic gradients."""
    input_tensor, S1s, S2s, U1s, U2s, bias = test_tensors
    output = sketched_linear_forward(input_tensor, S1s, S2s, U1s, U2s, bias)

    grad_output = torch.randn_like(output)

    grads1 = sketched_linear_backward(grad_output, input_tensor, S1s, S2s, U1s, U2s)
    grads2 = sketched_linear_backward(grad_output, input_tensor, S1s, S2s, U1s, U2s)

    for g1, g2 in zip(grads1, grads2):
        assert torch.allclose(g1, g2), "Backward pass is not deterministic."


def test_backward_vs_autograd(test_tensors):
    """Ensure gradients computed by sketched_linear_backward match autograd."""
    input_tensor, S1s, S2s, U1s, U2s, bias = test_tensors

    # Forward pass
    output = sketched_linear_forward(input_tensor, S1s, S2s, U1s, U2s, bias)

    # Random gradient to backpropagate
    grad_output = torch.randn_like(output)

    # Compute gradients using autograd
    autograd_grads = torch.autograd.grad(
        outputs=output,
        inputs=(input_tensor, S1s, S2s, bias),
        grad_outputs=grad_output,
        create_graph=True,
    )

    # Compute gradients using custom backward
    custom_grads = sketched_linear_backward(
        grad_output, input_tensor, S1s, S2s, U1s, U2s
    )
    assert len(autograd_grads) == 4, "Autograd returned incorrect number of gradients"
    assert len(autograd_grads) == len(custom_grads), "Gradient length mismatch"

    # Compare gradients from both methods
    for i, (custom, reference) in enumerate(zip(custom_grads, autograd_grads)):
        assert custom.shape == reference.shape, f"Gradient shape mismatch at index {i}"
        assert torch.allclose(
            custom, reference, atol=1e-6, rtol=1e-4
        ), f"Gradient mismatch at index {i}"


if __name__ == "__main__":
    pytest.main()
