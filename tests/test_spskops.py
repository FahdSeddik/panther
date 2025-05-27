import pytest
import torch

from panther.sketch import Axis, sparse_sketch_operator


def count_per_index(sparse_tensor, dim, size):
    """
    Returns a list `counts` of length `size` where
    counts[i] = number of nonzeros along `dim` == i.
    """
    coo = sparse_tensor.coalesce()
    idx = coo.indices().cpu()
    # dim=0 → count per row; dim=1 → count per column
    counts = [0] * size
    for i in range(idx.shape[1]):
        idx_i = int(idx[dim, i].item())
        counts[idx_i] += 1
    return counts


def unique_per_index(sparse_tensor, dim, size):
    """
    Returns a list `unique_counts` of length `size` where
    unique_counts[i] = number of unique indices along the other axis
    for which dim==i.
    """
    coo = sparse_tensor.coalesce()
    idx = coo.indices().cpu().numpy()
    other_dim = 1 - dim
    uniq = [set() for _ in range(size)]
    for j in range(idx.shape[1]):
        i = idx[dim, j]
        o = idx[other_dim, j]
        uniq[i].add(int(o))
    return [len(u) for u in uniq]


@pytest.fixture(params=[torch.device("cpu"), torch.device("cuda")])
def device(request):
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


@pytest.mark.parametrize(
    "m,n,vec_nnz",
    [
        (7, 20, 1),
        (7, 20, 2),
        (7, 20, 3),
        (7, 20, 7),
        (20, 7, 1),
        (20, 7, 2),
        (20, 7, 3),
        (20, 7, 7),
    ],
)
def test_saso_fixed_nnz_per_axis(m, n, vec_nnz, device, dtype):
    """
    For SASO (Axis.Short), if m < n, each column has exactly vec_nnz distinct rows.
    If m > n, each row has exactly vec_nnz distinct columns.
    """
    S = sparse_sketch_operator(m, n, vec_nnz, Axis.Short, device, dtype)

    # Ensure it's sparse COO
    assert S.layout == torch.sparse_coo

    coo = S.coalesce()
    nnz = coo.values().shape[0]
    # total nnz should equal vec_nnz * max(m,n)
    expected_nnz = vec_nnz * max(m, n)
    assert nnz == expected_nnz

    if m < n:
        # check per-column
        counts = count_per_index(coo, dim=1, size=n)
        unique = unique_per_index(coo, dim=1, size=n)
        for cnt, ucnt in zip(counts, unique):
            assert cnt == vec_nnz
            assert ucnt == vec_nnz
    else:
        # m > n: check per-row
        counts = count_per_index(coo, dim=0, size=m)
        unique = unique_per_index(coo, dim=0, size=m)
        for cnt, ucnt in zip(counts, unique):
            assert cnt == vec_nnz
            assert ucnt == vec_nnz


@pytest.mark.parametrize(
    "m,n",
    [
        (7, 20),
        (20, 7),
    ],
)
def test_laso_vec_nnz1_fixed_nnz(m, n, device, dtype):
    """
    For LASO with vec_nnz=1, if m < n, each row has exactly 1 nonzero.
    If m > n, each column has exactly 1 nonzero.
    """
    vec_nnz = 1
    S = sparse_sketch_operator(m, n, vec_nnz, Axis.Long, device, dtype)

    assert S.layout == torch.sparse_coo
    coo = S.coalesce()
    nnz = coo.values().shape[0]
    # For vec_nnz=1, total nnz == min(m,n)
    expected_nnz = min(m, n)
    assert nnz == expected_nnz

    if m < n:
        # each row must have exactly 1
        counts = count_per_index(coo, dim=0, size=m)
        unique = unique_per_index(coo, dim=0, size=m)
        for cnt, ucnt in zip(counts, unique):
            assert cnt == 1
            assert ucnt == 1
    else:
        # m > n: each column must have exactly 1
        counts = count_per_index(coo, dim=1, size=n)
        unique = unique_per_index(coo, dim=1, size=n)
        for cnt, ucnt in zip(counts, unique):
            assert cnt == 1
            assert ucnt == 1


@pytest.mark.parametrize(
    "m,n",
    [
        (7, 20),
        (20, 7),
    ],
)
def test_laso_randomness_values_sign(m, n, device, dtype):
    """
    For LASO with vec_nnz=1, value magnitude should be 1 (±1).
    We check that all nonzero values are ±1.
    """
    S = sparse_sketch_operator(m, n, 1, Axis.Long, device, dtype)
    coo = S.coalesce()
    vals = coo.values().cpu().numpy()
    for v in vals:
        assert pytest.approx(abs(v), rel=1e-6) == 1.0


@pytest.mark.parametrize(
    "m,n,vec_nnz",
    [
        (7, 20, 1),
        (7, 20, 2),
        (7, 20, 3),
        (7, 20, 7),
        (20, 7, 1),
        (20, 7, 2),
        (20, 7, 3),
        (20, 7, 7),
    ],
)
def test_saso_values_sign(m, n, vec_nnz, device, dtype):
    """
    For SASO, each nonzero value should be ±1.
    """
    S = sparse_sketch_operator(m, n, vec_nnz, Axis.Short, device, dtype)
    coo = S.coalesce()
    vals = coo.values().cpu().numpy()
    for v in vals:
        assert pytest.approx(abs(v), rel=1e-6) == 1.0
        assert pytest.approx(abs(v), rel=1e-6) == 1.0
