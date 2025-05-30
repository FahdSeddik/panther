import math

import pytest
import torch

from panther.sketch import srht

# =============================================================================
# Parameter definitions
# =============================================================================
# Power-of-two sizes and m values
SIZE_PARAMS = [
    (2, 1),
    (2, 2),
    (4, 1),
    (4, 2),
    (4, 4),
    (8, 1),
    (8, 4),
    (8, 8),
    (16, 1),
    (16, 8),
    (16, 16),
    (32, 1),
    (32, 16),
    (32, 32),
    (64, 1),
    (64, 32),
    (64, 64),
    (128, 1),
    (128, 64),
    (128, 128),
    (256, 1),
    (256, 128),
    (256, 256),
    (512, 1),
    (512, 256),
    (512, 512),
    (1024, 1),
    (1024, 512),
    (1024, 1024),
]

# Additional mixed cases
for k in [2048, 4096]:
    SIZE_PARAMS.append((k, k // 2))
    SIZE_PARAMS.append((k, k))

# Keep only m <= n
SIZE_PARAMS = [(n, m) for n, m in SIZE_PARAMS if m <= n]

# Ensure at least 30 combinations
assert len(SIZE_PARAMS) >= 30, f"Expected >=30 size/m pairs, got {len(SIZE_PARAMS)}"

# =============================================================================
# Test: shape, dtype, device
# =============================================================================


@pytest.mark.parametrize("n,m", SIZE_PARAMS)
def test_shape_dtype_device(n, m):
    """Verify output shape, dtype, and device for srht."""
    x = torch.arange(n, dtype=torch.float64)  # some data
    # Place on CPU
    x_cpu = x.to(torch.device("cpu"))
    out = srht(x_cpu, m)
    assert isinstance(out, torch.Tensor), "Output is not a Tensor"
    assert out.shape == (m,), f"Expected output shape ({m},), got {out.shape}"
    assert (
        out.dtype == torch.float32
    ), f"Expected dtype float32 after transform, got {out.dtype}"
    assert out.device.type == "cpu", f"Expected output on CPU, got {out.device}"


# =============================================================================
# Test: error conditions
# =============================================================================


def test_error_non_power_of_two():
    """Input length not power of two should raise error."""
    x = torch.randn(3, dtype=torch.float32)
    with pytest.raises(Exception):
        srht(x, 2)


def test_error_m_greater_than_n():
    """Requesting m > n should raise error."""
    x = torch.randn(4, dtype=torch.float32)
    with pytest.raises(Exception):
        srht(x, 5)


def test_error_input_not_cpu():
    """Input tensor on CUDA should raise error (if CUDA available)."""
    if torch.cuda.is_available():
        x_cuda = torch.randn(8, device="cuda", dtype=torch.float32)
        with pytest.raises(Exception):
            srht(x_cuda, 4)
    else:
        pytest.skip("CUDA not available, skipping test_error_input_not_cpu")


# =============================================================================
# Test: reproducibility and randomness
# =============================================================================


@pytest.mark.parametrize("n,m", [(16, 4), (32, 8), (64, 16)])
def test_reproducibility_and_randomness(n, m):
    """With a fixed seed, two runs match; two runs without reseeding differ."""
    # Generate a random input once under a fixed seed
    torch.manual_seed(123)
    x = torch.randn(n)

    # Two calls with the same seed should match
    torch.manual_seed(42)
    out1 = srht(x.clone(), m)
    torch.manual_seed(42)
    out2 = srht(x.clone(), m)
    assert torch.allclose(out1, out2), "Same seed -> outputs should match"

    # Two calls *without* reseeding should differ
    out3 = srht(x.clone(), m)
    out4 = srht(x.clone(), m)
    assert not torch.allclose(out3, out4), "Without reseed -> outputs should differ"


# =============================================================================
# Test: zero vector input
# =============================================================================


@pytest.mark.parametrize("n,m", [(8, 4), (16, 8), (32, 16)])
def test_zero_input(n, m):
    """SRHT of zero vector should be zero vector."""
    x = torch.zeros(n, dtype=torch.float32)
    out = srht(x, m)
    assert torch.all(out == 0), f"Expected all zeros, got {out.tolist()}"


# =============================================================================
# Test: ones vector input
# =============================================================================


@pytest.mark.parametrize("n,m", [(8, 8), (16, 4), (32, 16)])
def test_ones_input(n, m):
    """SRHT of a constant vector is finite; no invalid values."""
    x = torch.ones(n, dtype=torch.float32)
    torch.manual_seed(0)
    out = srht(x, m)
    assert torch.isfinite(out).all(), "Output contains non-finite values"


# =============================================================================
# Test: basis unit vector input
# =============================================================================


@pytest.mark.parametrize("n,m", [(8, 4), (16, 8), (32, 16)])
def test_basis_vector_input(n, m):
    """SRHT of a basis vector yields values ±1/sqrt(n) or 0, up to floating‐point tol."""
    e0 = torch.zeros(n, dtype=torch.float32)
    e0[0] = 1.0
    torch.manual_seed(1)
    out = srht(e0, m).tolist()

    expected_mag = 1.0 / math.sqrt(n)
    for v in out:
        # allow zero, or ±expected_mag within tol
        if abs(v) < 1e-6:
            continue
        assert math.isclose(
            abs(v), expected_mag, rel_tol=1e-6, abs_tol=1e-6
        ), f"Value {v} not within tol of ±{expected_mag}"


# =============================================================================
# Test: performance for large n
# =============================================================================


@pytest.mark.parametrize("n,m", [(2048, 256), (4096, 512)])
def test_large_n_performance(n, m):
    """Ensure SRHT completes on large inputs."""
    x = torch.randn(n)
    out = srht(x, m)
    assert out.shape == (m,)
    out = srht(x, m)
    assert out.shape == (m,)
