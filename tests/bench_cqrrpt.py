"""
benchmark_cqrrpt.py

Benchmark CQRRPT vs. GEQRF, GEQR, GEQP3, GEQPT, and sCholQR3 on tall-skinny matrices.

For each m in {8192, 16768} and n in {64, 128, 256, 512, 1024}, we:
  - Generate M ∈ R^{mxn} with i.i.d. N(0,1) entries (dtype chosen by DTYPE).
  - Perform NUM_RUNS independent runs of each algorithm (after a few warmups).
  - Record wall-clock time, RSS memory usage, factorization error, orthogonality error,
    and now also the absolute reconstruction error.
  - Report the **best** (minimum) runtime among those NUM_RUNS runs—per the paper's specification.

"""

import math
import time

import numpy as np
import pandas as pd
import psutil
import torch
from scipy.linalg import qr as scipy_qr  # type: ignore  # for pivoted and unpivoted QR

from panther.linalg import cqrrpt

# ------------------------------------------------------------
# User-Configurable Parameters
# ------------------------------------------------------------
M_SIZES = [8192, 16768]  # m values
N_SIZES = [64, 128, 256, 512, 1024]
GAMMA = 1.25
DTYPE = torch.float64  # torch.float32 or torch.float64
NUM_WARMUP = 2  # Warmup runs before timing
NUM_RUNS = 5  # As specified: "best of 20 runs"
OUTPUT_CSV = "cqrrpt_benchmark_results2.csv"
SEED = 1234  # For reproducibility


# ------------------------------------------------------------
# Helper: Measure current process resident-set size (RSS) in bytes
# ------------------------------------------------------------
def measure_memory():
    proc = psutil.Process()
    return proc.memory_info().rss


# ------------------------------------------------------------
# Helper: Orthogonality error ‖I − QᵀQ‖_F
# ------------------------------------------------------------
def orthogonality_error(Q: torch.Tensor) -> float:
    """
    Q: (mxk). Compute ‖I_k − QᵀQ‖_F in Python float.
    """
    k = Q.shape[1]
    QtQ = Q.transpose(0, 1).mm(Q)  # (kxk)
    I_k = torch.eye(k, dtype=Q.dtype, device=Q.device)
    diff = QtQ - I_k
    return diff.norm(p="fro").item()


# ------------------------------------------------------------
# Helper: Factorization error for pivoted QR: ‖M[:, J] − Q R‖_F / ‖M‖_F
# ------------------------------------------------------------
def factorization_error_pivoted(
    M: torch.Tensor, Q: torch.Tensor, R: torch.Tensor, J: torch.Tensor
) -> float:
    """
    J: 1D int64 Tensor of length n, 0-based column indices, so that
       M[:, J] = Q @ R ideally.
    Returns: relative Frobenius-norm error.
    """
    M_perm = M.index_select(dim=1, index=J)
    M_est = Q.mm(R)
    num = (M_perm - M_est).norm(p="fro").item()
    den = M.norm(p="fro").item()
    return num / (den + 1e-16)


# ------------------------------------------------------------
# Helper: Factorization error for unpivoted QR: ‖M − Q R‖_F / ‖M‖_F
# ------------------------------------------------------------
def factorization_error_unpivoted(
    M: torch.Tensor, Q: torch.Tensor, R: torch.Tensor
) -> float:
    """
    Returns: relative Frobenius-norm error.
    """
    M_est = Q.mm(R)
    num = (M - M_est).norm(p="fro").item()
    den = M.norm(p="fro").item()
    return num / (den + 1e-16)


# ------------------------------------------------------------
# Implementation of shifted-Cholesky-QR-3 (sCholQR3)
# (Algorithm 4.2 of FKN+20, using the Frobenius-norm shift).
# ------------------------------------------------------------
def shifted_cholqr3(M: torch.Tensor):
    """
    Input:
      M: (mxn) Tensor, dtype either float32 or float64, on CPU.
    Output:
      Q: (mxn) Tensor, R: (nxn) Tensor, J: None (no pivoting here).
    This loops adding a shift delta until cond(R)^2 · eps < 1 (approx).
    """
    m, n = M.shape
    dtype = M.dtype
    device = M.device
    eps = torch.finfo(dtype).eps

    # Compute ‖M‖_F²
    frob2 = (M.norm(p="fro") ** 2).item()
    # Initial shift δ = 10 · eps · ‖M‖_F²
    delta = 10.0 * eps * frob2

    # Identity_n
    I_n = torch.eye(n, dtype=dtype, device=device)

    # Solve X @ R = B when R is upper-triangular.
    def solve_right_upper(B: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        R_t = R.transpose(0, 1)  # lower-triangular
        B_t = B.transpose(0, 1)
        X_t = torch.linalg.solve_triangular(
            R_t,
            B_t,
            upper=False,  # R_t is lower-triangular
            left=True,  # solve R_t · X_t = B_t
            unitriangular=False,
        )
        return X_t.transpose(0, 1)

    while True:
        # G = MᵀM + δ I_n
        G = M.transpose(0, 1).mm(M) + delta * I_n  # (nxn), SPD
        # Compute Cholesky: G = L Lᵀ, L lower-triangular
        L = torch.linalg.cholesky(G, upper=False)  # (nxn)
        R = L.transpose(0, 1)  # (nxn), upper-triangular

        # Q = M @ inv(R)
        Q = solve_right_upper(M, R)  # (mxn)

        # Check cond(R)² · eps < 1?
        diagR = torch.diagonal(R, 0)
        abs_diag = diagR.abs()
        cond_est_sq = ((abs_diag.max() / abs_diag.min()) ** 2).item()
        if cond_est_sq * eps < 1.0:
            return Q, R, None

        # Otherwise, bump δ by a factor of 10 and repeat
        delta *= 10.0


# ------------------------------------------------------------
# GEQPT: "GEQR + GEQP3 on R1" per the description.
# ------------------------------------------------------------
def geqpt_torch(M: torch.Tensor):
    """
    Input:
      M: (mxn) Tensor, dtype either float32 or float64, on CPU.
    Output:
      Q: (mxn) Tensor, R: (nxn) Tensor, J: (n,) int64 Tensor, 0-based.
    Steps:
      1. [Q1, R1] = unpivoted QR of M  (torch.linalg.qr without pivot)
      2. [Q2, R2, piv] = pivoted QR of R1  (SciPy's qr(..., pivoting=True))
      3. Q = Q1 @ Q2
         R = R2
         J = piv  (0-based SciPy indices).
    """
    m, n = M.shape
    dtype = M.dtype
    device = M.device

    # 1) Unpivoted QR on M
    Q1, R1 = torch.linalg.qr(M, mode="reduced")  # (mxn), (nxn)

    # 2) Pivoted QR on R1 using SciPy
    R1_np = R1.cpu().numpy()
    Q2_np, R2_np, piv = scipy_qr(R1_np, mode="economic", pivoting=True)
    Q2 = torch.from_numpy(Q2_np).to(device=device, dtype=dtype)  # (nxn)
    R2 = torch.from_numpy(R2_np).to(device=device, dtype=dtype)  # (nxn)
    J = torch.from_numpy(np.array(piv, dtype=np.int64)).to(device=device)  # (n,)

    # 3) Form Q = Q1 @ Q2, R = R2
    Q = Q1.mm(Q2)  # (mxn)
    return Q, R2, J


# ------------------------------------------------------------
# GEQP3 using SciPy directly on M
# ------------------------------------------------------------
def gepq3_scipy(M: torch.Tensor):
    """
    Input:
      M: (mxn) Tensor, dtype either float32 or float64, on CPU.
    Output:
      Q: (mxn) Tensor, R: (nxn) Tensor, J: (n,) int64 Tensor, 0-based.
    Steps:
      Use SciPy's qr(…, pivoting=True) on M.numpy().
    """
    m, n = M.shape
    dtype = M.dtype
    device = M.device

    M_np = M.cpu().numpy()
    Q_np, R_np, piv = scipy_qr(M_np, mode="economic", pivoting=True)
    Q = torch.from_numpy(Q_np).to(device=device, dtype=dtype)
    R = torch.from_numpy(R_np).to(device=device, dtype=dtype)
    J = torch.from_numpy(np.array(piv, dtype=np.int64)).to(device=device)
    return Q, R, J


# ------------------------------------------------------------
# GEQRF (standard unpivoted Householder QR) via PyTorch
# ------------------------------------------------------------
def geqrf_torch(M: torch.Tensor):
    """
    Input:
      M: (mxn) Tensor, dtype either float32 or float64, on CPU.
    Output:
      Q: (mxn) Tensor, R: (nxn) Tensor, J=None (no pivoting).
    Uses torch.linalg.qr(M, mode="reduced") to produce (Q,R).
    """
    Q, R = torch.linalg.qr(M, mode="reduced")
    return Q, R, None


# ------------------------------------------------------------
# GEQR (unpivoted, dispatching logic)
#
# In PyTorch, torch.linalg.qr(...) automatically chooses the best internal routine
# for tall-skinny vs. general matrices. We label it as "GEQR," but the code is identical
# to geqrf_torch because torch.linalg.qr internally calls GEQRF+orgqr or a tall-skinny-special.
# ------------------------------------------------------------
def geqr_torch(M: torch.Tensor):
    """
    Identical to geqrf_torch, but labeled "GEQR" for clarity.
    """
    Q, R = torch.linalg.qr(M, mode="reduced")
    return Q, R, None


# ------------------------------------------------------------
# Main Benchmark Logic
# ------------------------------------------------------------
def run_benchmark():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Added "recon_err" column to record absolute reconstruction error.
    columns = [
        "m",
        "n",
        "algorithm",
        "best_time_sec",
        "memory_bytes",  # memory usage from the fastest run
        "factor_err",  # relative factorization error
        "orth_err",  # orthogonality error
        "recon_err",  # <<< added: absolute reconstruction error (Frobenius norm)
    ]
    records = []

    for m in M_SIZES:
        for n in N_SIZES:
            print(f"\n=== Benchmarking (m={m}, n={n}, dtype={DTYPE}) ===")
            # 1) Generate random matrix M ∈ ℝ^{mxn}, dtype DTYPE, on CPU:
            M = torch.randn((m, n), dtype=DTYPE, device="cpu")

            # 2) Warmup: run each algorithm twice (ignore output & timing)
            for _ in range(NUM_WARMUP):
                try:
                    _ = cqrrpt(M, GAMMA)
                except NotImplementedError:
                    pass
            for _ in range(NUM_WARMUP):
                _ = geqrf_torch(M)
            for _ in range(NUM_WARMUP):
                _ = geqr_torch(M)
            for _ in range(NUM_WARMUP):
                _ = gepq3_scipy(M)
            for _ in range(NUM_WARMUP):
                _ = geqpt_torch(M)
            for _ in range(NUM_WARMUP):
                _ = shifted_cholqr3(M)

            # ------------------------------------------------------------
            # For each algorithm, run NUM_RUNS times and record the *best* runtime
            # ------------------------------------------------------------
            def benchmark_algorithm(name, func, is_pivoted: bool):
                """
                Runs `func(M)` NUM_RUNS times, records:
                  • best runtime among runs,
                  • memory usage at that best run,
                  • factorization error (relative),
                  • orthogonality error,
                  • reconstruction error (absolute).
                Appends one dictionary to `records`.
                """
                best_time = math.inf
                best_mem = None
                best_Q, best_R, best_J = None, None, None

                for run_idx in range(NUM_RUNS):
                    t0 = time.perf_counter()
                    mem0 = measure_memory()
                    result = func(M)
                    t1 = time.perf_counter()
                    mem1 = measure_memory()

                    elapsed = t1 - t0
                    used_mem = mem1 - mem0

                    if elapsed < best_time:
                        best_time = elapsed
                        best_mem = used_mem
                        if isinstance(result, tuple) and len(result) == 3:
                            best_Q, best_R, best_J = result
                        else:
                            # In case someone returns a different shape, be safe:
                            best_Q, best_R, best_J = result[0], result[1], None

                # Compute errors from the best-time run
                if best_Q is None or best_R is None:
                    fac_err = None
                    ortho_err = None
                    recon_err = None
                else:
                    # Orthogonality error
                    ortho_err = orthogonality_error(best_Q)

                    # Factorization error (relative)
                    if is_pivoted:
                        assert (
                            best_J is not None
                        ), "best_J should not be None for pivoted QR"
                        fac_err = factorization_error_pivoted(M, best_Q, best_R, best_J)
                    else:
                        fac_err = factorization_error_unpivoted(M, best_Q, best_R)

                    # Reconstruction error (absolute Frobenius norm)
                    if is_pivoted:
                        # M[:, J] − Q R
                        assert (
                            best_J is not None
                        ), "best_J should not be None for pivoted QR"
                        M_perm = M.index_select(dim=1, index=best_J)
                        recon_err = (M_perm - best_Q.mm(best_R)).norm(p="fro").item()
                    else:
                        # M − Q R
                        recon_err = (M - best_Q.mm(best_R)).norm(p="fro").item()

                records.append(
                    {
                        "m": m,
                        "n": n,
                        "algorithm": name,
                        "best_time_sec": best_time,
                        "memory_bytes": best_mem,
                        "factor_err": fac_err,
                        "orth_err": ortho_err,
                        "recon_err": recon_err,  # <<< added
                    }
                )

                # Print a summary line including reconstruction error
                fac_err_str = f"{fac_err:.2e}" if fac_err is not None else "N/A"
                ortho_err_str = f"{ortho_err:.2e}" if ortho_err is not None else "N/A"
                recon_err_str = f"{recon_err:.2e}" if recon_err is not None else "N/A"
                best_mem_mb = best_mem / 1e6 if best_mem is not None else 0.0
                print(
                    f"    {name:10s}  →  time: {best_time:.4f}s,  "
                    f"mem: {best_mem_mb:.2f} MB,  "
                    f"fact_err: {fac_err_str},  "
                    f"orth_err: {ortho_err_str},  "
                    f"recon_err: {recon_err_str}"
                )

            # (A) CQRRPT (pivoted, random)
            try:
                benchmark_algorithm(
                    "CQRRPT", lambda A: cqrrpt(A, GAMMA), is_pivoted=True
                )
            except NotImplementedError:
                print("    CQRRPT: [skipped - binding not found]")

            # (B) GEQRF (unpivoted Householder)
            benchmark_algorithm("GEQRF", geqrf_torch, is_pivoted=False)

            # (C) GEQR (unpivoted, PyTorch dispatch)
            benchmark_algorithm("GEQR", geqr_torch, is_pivoted=False)

            # (D) GEQP3 (pivoted QR via SciPy)
            benchmark_algorithm("GEQP3", gepq3_scipy, is_pivoted=True)

            # (E) GEQPT (two-stage: GEQR + GEQP3 on R1)
            benchmark_algorithm("GEQPT", geqpt_torch, is_pivoted=True)

            # (F) sCholQR3 (shifted Cholesky-QR3)
            benchmark_algorithm("sCholQR3", shifted_cholqr3, is_pivoted=False)

            print(f"  → Completed all algorithms for (m={m}, n={n}).\n")

    # ------------------------------------------------------------
    # Build DataFrame from `records` and save to CSV
    # ------------------------------------------------------------
    df = pd.DataFrame.from_records(records, columns=columns)

    pd.set_option("display.expand_frame_repr", False)
    print("\n=== Raw Benchmark Results (including reconstruction error) ===")
    print(df)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run_benchmark()
