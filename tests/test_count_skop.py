import pytest
import torch

from panther.sketch import count_skop

# =============================================================================
# Parameter definitions for parametric tests
# =============================================================================

# List of (m, d) dimension pairs to test shape and sparsity
SHAPE_PARAMS = [
    (1, 1),
    (1, 5),
    (5, 1),
    (2, 2),
    (3, 7),
    (7, 3),
    (8, 8),
    (16, 16),
    (32, 64),
    (64, 32),
    (128, 128),
    (256, 128),
    (128, 256),
    (512, 512),
]

# Data type options to test
DTYPE_PARAMS = [
    None,  # default dtype (float32)
    torch.float32,
    torch.float64,
    torch.float16,
]

# Device options to test; include CUDA if available
DEVICE_PARAMS = [None, torch.device("cpu")] + (
    [torch.device("cuda")] if torch.cuda.is_available() else []
)

# Full cross-product
FULL_PARAMS = [
    (m, d, dtype, device)
    for m, d in SHAPE_PARAMS
    for dtype in DTYPE_PARAMS
    for device in DEVICE_PARAMS
]
assert (
    len(FULL_PARAMS) >= 100
), "Expected at least 100 parameter combinations for thorough testing."

# =============================================================================
# Test: Shape and type correctness
# =============================================================================


@pytest.mark.parametrize("m,d,dtype,device", FULL_PARAMS)
def test_shape_and_dtype(m, d, dtype, device):
    """Test that count_skop returns a tensor of correct shape and dtype."""
    tensor = count_skop(m, d, device, dtype)
    assert tensor.shape == (m, d), f"Expected shape ({m}, {d}), got {tensor.shape}"
    expected_dtype = dtype if dtype is not None else torch.float32
    assert (
        tensor.dtype == expected_dtype
    ), f"Expected dtype {expected_dtype}, got {tensor.dtype}"


# =============================================================================
# Test: Device placement
# =============================================================================


@pytest.mark.parametrize("m,d,dtype,device", FULL_PARAMS)
def test_device(m, d, dtype, device):
    """Test that count_skop respects the device argument or uses default CPU."""
    tensor = count_skop(m, d, device, dtype)
    expected_device = (
        torch.device(device) if device is not None else torch.device("cpu")
    )
    # Normalize CUDA device to canonical form (e.g., cuda -> cuda:0)
    if expected_device.type == "cuda" and expected_device.index is None:
        expected_device = torch.device("cuda:0")
    assert (
        tensor.device == expected_device
    ), f"Expected device {expected_device}, got {tensor.device}"


# =============================================================================
# Test: Sparsity properties
# =============================================================================


@pytest.mark.parametrize("m,d", SHAPE_PARAMS)
def test_exact_one_nonzero_per_column(m, d):
    """Ensure exactly one non-zero entry per column."""
    tensor = count_skop(m, d, None, None)
    # For each column, count non-zeros
    col_counts = (tensor != 0).sum(dim=0)
    assert torch.all(
        col_counts == 1
    ), f"Expected exactly one non-zero per column, got counts {col_counts.tolist()}"


@pytest.mark.parametrize("m,d", SHAPE_PARAMS)
def test_nonzero_values_are_plus_minus_one(m, d):
    """Ensure non-zero entries are either +1 or -1."""
    tensor = count_skop(m, d, None, torch.float32)
    # Extract non-zero entries
    nz = tensor[tensor != 0]
    # Should be ±1
    assert torch.all((nz == 1) | (nz == -1)), f"Non-zero entries not ±1: {nz.tolist()}"


# =============================================================================
# Test: Distribution properties
# =============================================================================

TOLERANCE_DISTRIBUTION = 0.2  # 20% tolerance


@pytest.mark.parametrize("m,d", [(100, 50), (200, 100), (500, 200)])
def test_uniform_row_selection(m, d):
    """Test that row selection per column is (approximately) uniform."""
    torch.manual_seed(0)
    tensor = count_skop(m, d, None, None)
    # Count frequency of each row index across all columns
    indices = [torch.nonzero(tensor[:, j]).item() for j in range(d)]
    freq = torch.tensor([indices.count(i) for i in range(m)], dtype=torch.float32)
    expected = d / m
    rel_error = ((freq - expected).abs() / expected).mean().item()
    assert rel_error < 3, f"Row selection distribution rel_error {rel_error:.2f}"


@pytest.mark.parametrize("m,d", [(100, 100), (200, 200)])
def test_uniform_sign_selection(m, d):
    """Test that sign selection is approximately balanced between +1 and -1."""
    torch.manual_seed(1)
    tensor = count_skop(m, d, None, None)
    signs = torch.tensor([tensor[:, j][tensor[:, j] != 0].item() for j in range(d)])
    pos_frac = (signs == 1).float().mean().item()
    neg_frac = (signs == -1).float().mean().item()
    assert (
        abs(pos_frac - 0.5) < TOLERANCE_DISTRIBUTION
    ), f"Positive sign fraction {pos_frac:.2f}"
    assert (
        abs(neg_frac - 0.5) < TOLERANCE_DISTRIBUTION
    ), f"Negative sign fraction {neg_frac:.2f}"


# =============================================================================
# Test: Randomness and reproducibility
# =============================================================================


@pytest.mark.parametrize("m,d", [(10, 10), (50, 5), (5, 50)])
def test_randomness_and_reproducibility(m, d):
    """Test reproducibility with fixed seed and randomness by default."""
    torch.manual_seed(42)
    t1 = count_skop(m, d, None, None)
    torch.manual_seed(42)
    t2 = count_skop(m, d, None, None)
    assert torch.equal(t1, t2), "Outputs differ despite same seed"
    t3 = count_skop(m, d, None, None)
    assert not torch.equal(t2, t3), "Outputs identical without reseeding"


# =============================================================================
# Test: Edge and error cases
# =============================================================================


@pytest.mark.parametrize("m,d", [(1, 1)])
def test_non_integer_dimensions_error(m, d):
    """Passing non-integer types for m or d should raise a TypeError."""
    with pytest.raises(TypeError):
        count_skop(3.5, d, None, None)
    with pytest.raises(TypeError):
        count_skop(m, "4", None, None)


# =============================================================================
# Test: Performance stress tests
# =============================================================================


@pytest.mark.parametrize("m,d", [(10000, 100), (100, 10000)])
def test_large_matrix_performance(m, d):
    """Ensure generating large sparse matrices completes within time/memory limits."""
    tensor = count_skop(m, d, None, None)
    assert tensor.shape == (m, d)
