import math

import numpy as np
import pytest
import torch

from pawX import DistributionFamily, dense_sketch_operator, sketch_tensor


# Helper: set a random seed for reproducibility of tests that depend on statistics.
@pytest.fixture(autouse=True)
def set_seed():
    torch.manual_seed(42)
    yield
    torch.manual_seed(torch.initial_seed())


#########################################
# Tests for dense_sketch_operator
#########################################


@pytest.mark.parametrize("m, n", [(32, 64), (64, 128), (128, 256)])
def test_dense_sketch_operator_gaussian_shape(m, n):
    # Test Gaussian: shape and basic statistical properties.
    S = dense_sketch_operator(m, n, DistributionFamily.Gaussian)
    assert S.shape == (m, n)
    S_np = S.cpu().numpy()
    # With Gaussian, expect mean ~0 and variance ~1.
    np.testing.assert_allclose(np.mean(S_np), 0, atol=0.3)
    np.testing.assert_allclose(np.var(S_np), 1, atol=0.3)


@pytest.mark.parametrize("m, n", [(32, 64), (64, 128), (128, 256)])
def test_dense_sketch_operator_uniform_scaling(m, n):
    # Test Uniform: after scaling, the variance should be approximately 1.
    S = dense_sketch_operator(m, n, DistributionFamily.Uniform)
    assert S.shape == (m, n)
    S_np = S.cpu().numpy()
    # Uniform U(-1,1) scaled by sqrt(3) gives values in [-sqrt(3), sqrt(3)].
    lower_bound, upper_bound = -math.sqrt(3), math.sqrt(3)
    assert np.all(S_np >= lower_bound)
    assert np.all(S_np <= upper_bound)
    np.testing.assert_allclose(np.var(S_np), 1, atol=0.3)


def test_dense_sketch_operator_invalid_distribution():
    # Passing an invalid distribution value should raise an exception.
    with pytest.raises(Exception):
        # Assuming the binding enforces the type for DistributionFamily.
        dense_sketch_operator(32, 32, 999)


#########################################
# Tests for sketch_tensor (operator-creating overload)
#########################################


@pytest.mark.parametrize(
    "input_shape, axis, new_size",
    [
        ((10, 20), 1, 10),
        ((10, 20, 30), 1, 15),
        ((10, 20, 30, 40), 2, 20),
    ],
)
def test_sketch_tensor_output_shape(input_shape, axis, new_size):
    # Create an input tensor and test the output shape.
    input_tensor = torch.randn(*input_shape)
    # Using the overload that returns (output, sketch_operator)
    output, operator = sketch_tensor(
        input_tensor, axis, new_size, DistributionFamily.Gaussian
    )
    # Expected shape: same as input except with dimension "axis" replaced by new_size.
    expected_shape = list(input_shape)
    expected_shape[axis] = new_size
    assert tuple(expected_shape) == output.shape
    # Operator shape should be (new_size x original_size along axis)
    assert operator.shape == (new_size, input_shape[axis])


@pytest.mark.parametrize(
    "input_shape, axis, new_size",
    [
        ((8, 16), 0, 4),
        ((8, 16), 1, 8),
        ((10, 20, 30), 0, 5),
        ((10, 20, 30), 2, 10),
    ],
)
def test_sketch_tensor_with_operator_explicit(input_shape, axis, new_size):
    # Test the overload that accepts an explicit sketch operator.
    input_tensor = torch.randn(*input_shape)
    original_size = input_tensor.size(axis)
    # Generate a sketch operator externally
    sketch_operator = dense_sketch_operator(
        new_size, original_size, DistributionFamily.Uniform
    )

    # Call the overload that takes the sketch_operator explicitly.
    output = sketch_tensor(input_tensor, axis, new_size, sketch_operator)
    expected_shape = list(input_shape)
    expected_shape[axis] = new_size
    assert tuple(expected_shape) == output.shape


def test_sketch_tensor_invalid_axis():
    # Invalid axis index (out of bounds) should raise an error.
    input_tensor = torch.randn(10, 20, 30)
    with pytest.raises(RuntimeError):
        sketch_tensor(
            input_tensor, axis=3, new_size=10, distribution=DistributionFamily.Gaussian
        )


def test_sketch_tensor_invalid_new_size():
    # new_size larger than the original dimension should raise an error.
    input_tensor = torch.randn(10, 20, 30)
    with pytest.raises(RuntimeError):
        sketch_tensor(
            input_tensor, axis=1, new_size=25, distribution=DistributionFamily.Gaussian
        )


#########################################
# Additional tests: multiple dimensions, statistical sanity, reproducibility etc.
#########################################


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_sketch_tensor_multidimensional(dim):
    # Create random tensors with increasing dimensions.
    shape = tuple(np.random.randint(5, 20) for _ in range(dim))
    input_tensor = torch.randn(*shape)
    # Randomly choose an axis to sketch and a valid new size.
    axis = np.random.randint(0, dim)
    original_size = input_tensor.size(axis)
    new_size = np.random.randint(1, original_size + 1)
    output, op = sketch_tensor(
        input_tensor, axis, new_size, DistributionFamily.Gaussian
    )
    expected_shape = list(shape)
    expected_shape[axis] = new_size
    assert tuple(expected_shape) == output.shape
    assert op.shape == (new_size, original_size)


def test_sketch_tensor_statistics_consistency():
    # Create an input tensor with shape (100, 200)
    input_tensor = torch.randn(100, 200)

    # We will sketch along axis=1 reducing its size to 50
    axis = 1
    new_size = 50

    # Perform the sketch operation
    output, _ = sketch_tensor(input_tensor, axis, new_size, DistributionFamily.Gaussian)
    var_input = input_tensor.var(dim=axis).mean().item() * input_tensor.shape[axis]
    var_output = output.var(dim=axis).mean().item()
    assert math.isclose(
        var_input, var_output, rel_tol=0.1, abs_tol=0.1
    ), f"Variance mismatch: {var_input} vs {var_output}"


@pytest.mark.parametrize(
    "device_str", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_sketch_tensor_device_dtype(device_str, dtype):
    # Validate that specifying device and dtype is honored.
    device = torch.device(device_str)
    input_tensor = torch.randn(10, 30, 50, device=device, dtype=dtype)
    new_size = 12
    output, sketch_operator = sketch_tensor(
        input_tensor,
        axis=2,
        new_size=new_size,
        distribution=DistributionFamily.Gaussian,
        device=device,
        dtype=dtype,
    )
    assert output.device == input_tensor.device
    assert output.dtype == input_tensor.dtype
    assert sketch_operator.device == input_tensor.device
    assert sketch_operator.dtype == input_tensor.dtype


def test_repeated_calls_reproducibility():
    # Although the underlying functions are random, setting a fixed seed before each call should yield
    # similar statistical behavior.
    torch.manual_seed(1234)
    out1, _ = sketch_tensor(
        torch.randn(50, 100),
        axis=1,
        new_size=40,
        distribution=DistributionFamily.Gaussian,
    )
    torch.manual_seed(1234)
    out2, _ = sketch_tensor(
        torch.randn(50, 100),
        axis=1,
        new_size=40,
        distribution=DistributionFamily.Gaussian,
    )
    # The outputs are random; if the seed resets they should be statistically similar.
    np.testing.assert_allclose(
        out1.cpu().numpy().mean(), out2.cpu().numpy().mean(), atol=0.3
    )
    np.testing.assert_allclose(
        out1.cpu().numpy().var(), out2.cpu().numpy().var(), atol=0.3
    )


#########################################
# Additional edge cases and stress tests.
#########################################


def test_large_tensor_sketch():
    # Stress test on a moderately large tensor.
    input_tensor = torch.randn(256, 512)
    axis = 1
    new_size = 128
    output, op = sketch_tensor(input_tensor, axis, new_size, DistributionFamily.Uniform)
    expected_shape = (input_tensor.size(0), new_size)
    assert output.shape == expected_shape
    assert op.shape == (new_size, input_tensor.size(axis))


def test_non_standard_dtype():
    # Test non-standard (but valid) dtype; e.g., torch.half.
    # Note: half precision might not be supported on CPU by all operations. Use cuda if available.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.half
        input_tensor = torch.randn(20, 40, device=device, dtype=dtype)
        new_size = 10
        output, op = sketch_tensor(
            input_tensor,
            axis=1,
            new_size=new_size,
            distribution=DistributionFamily.Gaussian,
            device=device,
            dtype=dtype,
        )
        assert output.dtype == dtype
        assert op.dtype == dtype
    else:
        pytest.skip("Half precision test skipped as CUDA is not available.")


if __name__ == "__main__":
    pytest.main()
