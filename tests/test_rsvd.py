import pytest
import torch

from panther.linalg import randomized_svd

torch.manual_seed(42)


def make_polynomial_matrix(m, n, k, cond=2.0, device="cpu", dtype=torch.float64):
    """
    Polynomial rank-k generator with cond number ~= cond:
        A = U[:, :k] @ diag(s) @ V[:, :k].T
    with s_0=1, s_{k-1}=1/cond, geometric decay.
    """
    U = torch.linalg.qr(torch.randn(m, k, dtype=dtype, device=device))[0]
    V = torch.linalg.qr(torch.randn(n, k, dtype=dtype, device=device))[0]
    exponents = torch.linspace(0, 1, steps=k, device=device, dtype=dtype)
    s = cond ** (-exponents)
    return U @ torch.diag(s) @ V.T, U, s, V


# Parameterize over shapes, data types, and devices
shapes = [
    (10, 10, 5),
    (50, 20, 10),
    (30, 60, 20),
    (100, 10, 5),
]
dtypes = [torch.float32, torch.float64]
devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("m,n,k", shapes)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("device", devices)
def test_randlapack_equivalence_and_orthogonality(m, n, k, dtype, device):
    # tolerance settings
    eps = torch.finfo(dtype).eps
    tol = eps**0.5625
    tol_strict = eps**0.625

    # Generate test matrix on the specified device & dtype
    A, U_true, s_true, V_true = make_polynomial_matrix(
        m, n, k, cond=2.0, device=device, dtype=dtype
    )

    # Exact truncated SVD
    U_full, S_full, Vt_full = torch.linalg.svd(A, full_matrices=False)
    U0, S0, V0 = U_full[:, :k], S_full[:k], Vt_full[:k, :].T

    # Randomized SVD
    U1, S1, V1 = randomized_svd(A, k, tol)

    # Check shapes
    assert U1.shape == (m, k)
    assert S1.shape == (k,)
    assert V1.shape == (n, k)

    # Reconstruct low-rank approximations
    A_k0 = U0 @ torch.diag(S0) @ V0.T
    A_approx = U1 @ torch.diag(S1) @ V1.T

    # Compare approximations
    diff = torch.norm(A_k0 - A_approx, p="fro").item()
    assert diff < tol_strict, (
        f"Shape {(m,n,k)} dtype {dtype} device {device}: "
        f"‖A_k_exact - A_k_rand‖_F = {diff:.3e} > {tol_strict:.1e}"
    )

    # Check orthonormality of U1 and V1
    I_k = torch.eye(k, dtype=dtype, device=device)
    errU = torch.norm(U1.T @ U1 - I_k, p="fro").item()
    errV = torch.norm(V1.T @ V1 - I_k, p="fro").item()
    assert errU < 1e-5, f"U not orthonormal (err={errU:.1e}) on {device},{dtype}"
    assert errV < 2e-5, f"V not orthonormal (err={errV:.1e}) on {device},{dtype}"


if __name__ == "__main__":
    pytest.main([__file__])
