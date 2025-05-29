import pytest
import torch

from panther.sketch import sjlt_skop

# =============================================================================
# Parameter definitions
# =============================================================================

SHAPE_PARAMS = [
    (1, 1, 1),
    (1, 5, 1),
    (5, 1, 1),
    (2, 2, 1),
    (3, 7, 2),
    (7, 3, 2),
    (8, 8, 2),
    (16, 16, 3),
    (32, 64, 4),
    (64, 32, 5),
    (128, 128, 2),
    (256, 128, 4),
    (128, 256, 8),
    (512, 512, 16),
    (1024, 64, 2),
    (64, 1024, 4),
    (100, 200, 5),
    (200, 100, 10),
    (50, 50, 5),
    (75, 25, 3),
    (25, 75, 3),
    (123, 456, 2),
    (456, 123, 4),
    (512, 1024, 8),
    (1024, 512, 16),
    (999, 1, 1),
    (1, 999, 1),
]

DTYPE_PARAMS = [None, torch.float32, torch.float64, torch.float16]

DEVICE_PARAMS = [None, torch.device("cpu")] + (
    [torch.device("cuda")] if torch.cuda.is_available() else []
)

FULL_PARAMS = [
    (m, d, s, dtype, device)
    for (m, d, s) in SHAPE_PARAMS
    for dtype in DTYPE_PARAMS
    for device in DEVICE_PARAMS
]
assert (
    len(FULL_PARAMS) >= 200
), "Expected at least 200 parameter combinations for thorough testing."

# =============================================================================
# Test: shape, dtype, device
# =============================================================================


@pytest.mark.parametrize("m,d,sparsity,dtype,device", FULL_PARAMS)
def test_shape_dtype_device(m, d, sparsity, dtype, device):
    """Test correct shape, dtype, and device placement."""
    tensor = sjlt_skop(m, d, sparsity, device, dtype)
    assert tensor.shape == (
        m,
        d,
    ), f"Shape mismatch: expected ({m},{d}), got {tensor.shape}"
    expected_dtype = dtype if dtype is not None else torch.float32
    assert (
        tensor.dtype == expected_dtype
    ), f"Dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"
    expected_device = (
        torch.device(device) if device is not None else torch.device("cpu")
    )
    # Normalize CUDA device to canonical form (e.g., cuda -> cuda:0)
    if expected_device.type == "cuda" and expected_device.index is None:
        expected_device = torch.device("cuda:0")
    assert (
        tensor.device == expected_device
    ), f"Device mismatch: expected {expected_device}, got {tensor.device}"


# =============================================================================
# Test: column sparsity & normalization (with isclose)
# =============================================================================


@pytest.mark.parametrize("m,d,sparsity", [(10, 5, 2), (20, 10, 3), (50, 25, 5)])
def test_column_sparsity_and_normalization(m, d, sparsity):
    """Each column must have exactly `sparsity` non-zeros of ±1/sqrt(sparsity)."""
    tensor = sjlt_skop(m, d, sparsity, None, None)
    expected_val = 1.0 / (sparsity**0.5)
    for j in range(d):
        col = tensor[:, j]
        nonzeros = col[col != 0]
        assert (
            nonzeros.numel() == sparsity
        ), f"Column {j} nonzero count {nonzeros.numel()}, expected {sparsity}"
        # Check each nonzero is ±expected_val within tolerance
        for val in nonzeros:
            assert torch.isclose(
                val.abs(), torch.tensor(expected_val), atol=1e-6
            ), f"Column {j} contains value {val.item()}, expected ±{expected_val}"


# =============================================================================
# Test: uniformity of row selection via multiple trials
# =============================================================================

TOLERANCE_ROW = 0.2
TRIALS = 20  # average over multiple SJLT samples


@pytest.mark.parametrize("m,d,sparsity", [(100, 50, 2), (200, 100, 3), (500, 200, 5)])
def test_uniform_row_selection(m, d, sparsity):
    """Row indices should be roughly uniformly distributed across all nonzero entries."""
    counts = torch.zeros(m, dtype=torch.float32)

    for _ in range(TRIALS):
        torch.manual_seed(_)  # different seed each trial
        tensor = sjlt_skop(m, d, sparsity, None, None)
        # collect every nonzero row index
        for j in range(d):
            nz_pos = torch.nonzero(tensor[:, j], as_tuple=False).flatten().tolist()
            for idx in nz_pos:
                counts[idx] += 1

    # expected count per row: TRIALS * (d * sparsity / m)
    expected = TRIALS * (d * sparsity) / m
    rel_error = ((counts - expected).abs() / expected).mean().item()
    assert (
        rel_error < TOLERANCE_ROW
    ), f"Row selection distribution rel_error {rel_error:.2f} (> {TOLERANCE_ROW})"


# =============================================================================
# Test: balanced sign distribution
# =============================================================================

TOLERANCE_SIGN = 0.2


@pytest.mark.parametrize("m,d,sparsity", [(100, 100, 2), (200, 200, 3), (300, 150, 5)])
def test_balanced_signs(m, d, sparsity):
    """Signs should be roughly balanced between + and - across all nonzeros."""
    torch.manual_seed(1)
    tensor = sjlt_skop(m, d, sparsity, None, None)
    signs = []
    for j in range(d):
        col = tensor[:, j]
        vals = col[col != 0]
        signs.extend(
            [int(v.item() > 0) for v in vals]
        )  # 1 for positive, 0 for negative
    signs = torch.tensor(signs, dtype=torch.float32)
    pos_frac = signs.mean().item()
    assert abs(pos_frac - 0.5) < TOLERANCE_SIGN, f"Positive fraction {pos_frac:.2f}"


# =============================================================================
# Test: normalization variance property
# =============================================================================


@pytest.mark.parametrize("m,d,sparsity", [(100, 50, 2), (200, 100, 4), (400, 200, 8)])
def test_variance_normalization(m, d, sparsity):
    """Variance of entries should equal 1/sparsity across flattened nonzeros."""
    torch.manual_seed(2)
    tensor = sjlt_skop(m, d, sparsity, None, None)
    all_vals = tensor[tensor != 0]
    emp_var = all_vals.var(unbiased=False).item()
    expected_var = 1.0 / sparsity
    assert (
        abs(emp_var - expected_var) / expected_var < 0.1
    ), f"Variance {emp_var:.4f} vs expected {expected_var:.4f}"


# =============================================================================
# Test: randomness and reproducibility
# =============================================================================


@pytest.mark.parametrize("m,d,sparsity", [(10, 10, 2), (20, 5, 3), (5, 20, 1)])
def test_reproducibility_and_randomness(m, d, sparsity):
    """Fixed seed should reproduce, default should vary."""
    torch.manual_seed(42)
    t1 = sjlt_skop(m, d, sparsity, None, None)
    torch.manual_seed(42)
    t2 = sjlt_skop(m, d, sparsity, None, None)
    assert torch.equal(t1, t2), "Same seed yields different outputs"
    t3 = sjlt_skop(m, d, sparsity, None, None)
    assert not torch.equal(t2, t3), "Different calls without reseed match"


# =============================================================================
# Regression tests for small matrices (placeholders)
# =============================================================================


@pytest.mark.parametrize(
    "m,d,sparsity,seed,expected",
    [
        (2, 2, 1, 1, [[1.0, -1.0], [-1.0, 1.0]]),  # placeholder
        (
            3,
            3,
            2,
            7,
            [[0.707, -0.707, 0], [0, -0.707, 0.707], [0.707, 0, 0.707]],
        ),  # placeholder
    ],
)
def test_regression_small(m, d, sparsity, seed, expected):
    torch.manual_seed(seed)
    tensor = sjlt_skop(m, d, sparsity, None, None)
    assert tensor.shape == (m, d)
    assert tensor.dtype == torch.float32
    # Placeholder exact match check
    # assert tensor.tolist() == expected


# =============================================================================
# Edge and error cases
# =============================================================================


def test_invalid_dimensions_error():
    """Non-positive or non-integer dims error."""
    with pytest.raises(Exception):
        sjlt_skop(0, 5, 1, None, None)
    with pytest.raises(TypeError):
        sjlt_skop(5.5, 5, 1, None, None)
    with pytest.raises(TypeError):
        sjlt_skop(5, "5", 1, None, None)


# =============================================================================
# Performance tests
# =============================================================================


@pytest.mark.parametrize("m,d,sparsity", [(10000, 100, 2), (100, 10000, 3)])
def test_large_matrix_performance(m, d, sparsity):
    tensor = sjlt_skop(m, d, sparsity, None, None)
    assert tensor.shape == (m, d)
