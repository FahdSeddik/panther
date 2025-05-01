import pytest
import torch

from panther.utils.compatibility import has_tensor_core_support
from pawX import sketched_linear_backward, sketched_linear_forward

# Set random seed for reproducibility
torch.manual_seed(42)

# Test parameters
BATCH_SIZE = 32
IN_FEATURES = 1024
OUT_FEATURES = 2048
LOW_RANK_DIM = 64
NUM_TERMS = 3


@pytest.fixture
def test_tensors():
    """Fixture to provide test tensors with reproducible random values."""
    input_tensor = torch.randn(
        BATCH_SIZE, IN_FEATURES, dtype=torch.float, requires_grad=True
    )
    S1s = torch.randn(
        NUM_TERMS, IN_FEATURES, LOW_RANK_DIM, dtype=torch.float, requires_grad=True
    )
    S2s = torch.randn(
        NUM_TERMS, LOW_RANK_DIM, OUT_FEATURES, dtype=torch.float, requires_grad=True
    )
    U1s = torch.randn(
        NUM_TERMS, LOW_RANK_DIM, OUT_FEATURES, dtype=torch.float, requires_grad=False
    )
    U2s = torch.randn(
        NUM_TERMS, IN_FEATURES, LOW_RANK_DIM, dtype=torch.float, requires_grad=False
    )
    bias = torch.randn(OUT_FEATURES, dtype=torch.float, requires_grad=True)

    return input_tensor, S1s, S2s, U1s, U2s, bias


def test_forward_output_shape(test_tensors):
    """Ensure the output shape matches expectations."""
    input_tensor, S1s, S2s, U1s, U2s, bias = test_tensors
    output = sketched_linear_forward(input_tensor, S1s, S2s, U1s, U2s, bias)

    assert output.shape == (BATCH_SIZE, OUT_FEATURES), "Output shape mismatch."


def test_tc(test_tensors):
    if not has_tensor_core_support():
        pytest.skip("Tensor Core support is not available.")
    # all test tensors to be on GPU
    input_tensor, S1s, S2s, U1s, U2s, bias = test_tensors
    i_gpu = input_tensor.to("cuda")
    S1s_gpu = S1s.to("cuda")
    S2s_gpu = S2s.to("cuda")
    U1s_gpu = U1s.to("cuda")
    U2s_gpu = U2s.to("cuda")
    bias_gpu = bias.to("cuda")
    # check that without and with gpu is same value or close
    output_tc = sketched_linear_forward(
        i_gpu, S1s_gpu, S2s_gpu, U1s_gpu, U2s_gpu, bias_gpu, use_tensor_core=True
    ).cpu()
    output_no_tc = sketched_linear_forward(
        input_tensor, S1s, S2s, U1s, U2s, bias, use_tensor_core=False
    )
    assert output_tc.shape == output_no_tc.shape, "Output shape mismatch."
    assert torch.allclose(
        output_tc, output_no_tc, atol=1
    ), "Output values mismatch between Tensor Core and non-Tensor Core."


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


def test_forward_gpu_vs_cpu(test_tensors):
    """Ensure forward pass produces the same output on CPU and GPU."""
    # skip test anyway
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")

    input_tensor, S1s, S2s, U1s, U2s, bias = test_tensors

    input_tensor_gpu = input_tensor.to("cuda")
    S1s_gpu = S1s.to("cuda")
    S2s_gpu = S2s.to("cuda")
    U1s_gpu = U1s.to("cuda")
    U2s_gpu = U2s.to("cuda")
    bias_gpu = bias.to("cuda")

    output_cpu = sketched_linear_forward(input_tensor, S1s, S2s, U1s, U2s, bias)
    output_gpu = sketched_linear_forward(
        input_tensor_gpu, S1s_gpu, S2s_gpu, U1s_gpu, U2s_gpu, bias_gpu
    ).cpu()

    assert torch.allclose(
        output_cpu, output_gpu, atol=1
    ), "Forward pass differs between CPU and GPU."


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
            custom, reference, atol=1e-3, rtol=1e-3
        ), f"Gradient mismatch at index {i}"


if __name__ == "__main__":
    pytest.main()
