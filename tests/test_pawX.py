import pytest
import torch

import pawX


def test_import_pawX():
    """Test if pawX is importable and has the expected attributes."""
    assert hasattr(
        pawX, "scaled_sign_sketch"
    ), "❌ 'scaled_sign_sketch' not found in pawX"


def test_scaled_sign_sketch_output():
    """Test if scaled_sign_sketch returns a valid tensor."""
    m, n = 4, 4
    result = pawX.scaled_sign_sketch(m, n)

    assert isinstance(result, torch.Tensor), "❌ Output is not a torch.Tensor"
    assert result.shape == (
        m,
        n,
    ), f"❌ Unexpected shape: {result.shape}, expected ({m}, {n})"


def test_scaled_sign_sketch_values():
    """Test if scaled_sign_sketch produces values in {-1/sqrt(m), 1/sqrt(m)}."""
    m, n = 5, 5
    result = pawX.scaled_sign_sketch(m, n)
    unique_values = set(result.flatten().tolist())
    expected_values = {
        1 / torch.sqrt(torch.tensor(m, dtype=result.dtype)).item(),
        -1 / torch.sqrt(torch.tensor(m, dtype=result.dtype)).item(),
    }

    assert len(unique_values) == len(
        expected_values
    ), "❌ Unexpected number of unique values"
    assert all(
        any(
            torch.isclose(torch.tensor(val), torch.tensor(ev)) for ev in expected_values
        )
        for val in unique_values
    ), f"❌ Unexpected values: {unique_values - expected_values}"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scaled_sign_sketch_device(device):
    """Test if scaled_sign_sketch works on both CPU and CUDA (if available)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Skipping CUDA test: CUDA not available")

    m, n = 8, 8
    result = pawX.scaled_sign_sketch(m, n, device=torch.device(device))

    assert (
        result.device.type == device
    ), f"❌ Expected device: {device}, got {result.device}"


if __name__ == "__main__":
    pytest.main()
