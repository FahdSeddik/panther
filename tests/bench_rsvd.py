import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from memory_profiler import memory_usage  # type: ignore
from sklearn.utils.extmath import (  # type: ignore[import-untyped]
    randomized_svd as sklearn_randomized_svd,
)
from torch.linalg import svd as torch_svd

# Import your custom RSVD implementation
from panther.linalg import randomized_svd as panther_randomized_svd

# -------------------------
# Configuration Parameters
# -------------------------
CONFIG = {
    "matrix_shapes": [
        (100, 100),
        (500, 300),
        (1000, 1000),
        (2000, 500),
    ],
    "target_ranks": [10, 50, 100],
    "n_runs": 5,
    "tolerance": 1e-6,
    "random_seed": 42,
    "results_csv": "rsvd_benchmark_results.csv",
    "timing_plot": "rsvd_timing.png",
    "error_plot": "rsvd_error.png",
    "memory_plot": "rsvd_memory.png",
}


# -------------------------
# Helper Functions
# -------------------------
def compute_reconstruction_error(
    A: np.ndarray, U: np.ndarray, S: np.ndarray, V: np.ndarray
) -> Tuple[float, float]:
    """
    Returns absolute Frobenius error and relative error = ||A - U S V^T|| / ||A||.
    """
    A_approx = U @ np.diag(S) @ V.T
    abs_err = np.linalg.norm(A - A_approx, ord="fro")
    rel_err = abs_err / np.linalg.norm(A, ord="fro")
    return abs_err.item(), rel_err.item()


def run_panther(A: np.ndarray, k: int, tol: float):
    A_t = torch.from_numpy(A)
    U_t, S_t, V_t = panther_randomized_svd(A_t, k, tol)
    return U_t.cpu().numpy(), S_t.cpu().numpy(), V_t.cpu().numpy()


def run_sklearn(A: np.ndarray, k: int, tol: float):
    U, S, Vt = sklearn_randomized_svd(A, n_components=k)
    return U, S, Vt.T


def run_torch_svd(A: np.ndarray, k: int, tol: float):
    A_t = torch.from_numpy(A)
    U_t, S_t, Vt_t = torch_svd(A_t, full_matrices=False)
    U = U_t[:, :k].cpu().numpy()
    S = S_t[:k].cpu().numpy()
    V = Vt_t[:k, :].T.cpu().numpy()
    return U, S, V


def benchmark_single_run(func, A: np.ndarray, k: int, tol: float):
    """
    Returns (time_sec, peak_mem_mb, abs_error, rel_error).
    We measure timing first with pure perf_counter,
    then separately peak memory via memory_usage().
    """
    # Warm‐up
    _ = func(A, k, tol)

    # 1) Timing
    t0 = time.perf_counter()
    U, S, V = func(A, k, tol)
    elapsed = time.perf_counter() - t0

    # 2) Memory
    # Only measure memory footprint of the call, not timing
    mem_usage, _ = memory_usage(
        (func, (A, k, tol)),
        retval=True,
        max_iterations=1,
        interval=0.01,
    )
    peak_mem = max(mem_usage) - mem_usage[0]

    # 3) Reconstruction error
    abs_err, rel_err = compute_reconstruction_error(A, U, S, V)
    return elapsed, peak_mem, abs_err, rel_err


# -------------------------
# Benchmark Loop
# -------------------------
def run_benchmarks(config):
    np.random.seed(config["random_seed"])
    results = []

    implementations = {
        "Panther_RSVD": run_panther,
        "ScikitLearn_RSVD": run_sklearn,
        "Torch_SVD": run_torch_svd,
    }

    for m, n in config["matrix_shapes"]:
        for k in config["target_ranks"]:
            if k > min(m, n):
                continue
            A = np.random.randn(m, n)
            for name, func in implementations.items():
                times, mems, abs_errs, rel_errs = [], [], [], []
                for _ in range(config["n_runs"]):
                    t, mem, ae, re = benchmark_single_run(
                        func, A, k, config["tolerance"]
                    )
                    times.append(t)
                    mems.append(mem)
                    abs_errs.append(ae)
                    rel_errs.append(re)
                results.append(
                    {
                        "implementation": name,
                        "m": m,
                        "n": n,
                        "k": k,
                        "time_mean": np.mean(times),
                        "time_std": np.std(times),
                        "memory_mean": np.mean(mems),
                        "memory_std": np.std(mems),
                        "abs_error_mean": np.mean(abs_errs),
                        "abs_error_std": np.std(abs_errs),
                        "rel_error_mean": np.mean(rel_errs),
                        "rel_error_std": np.std(rel_errs),
                    }
                )
                print(f"Done {name} on {m}×{n}, k={k}")

    df = pd.DataFrame(results)
    df.to_csv(config["results_csv"], index=False)
    return df


# -------------------------
# Plotting
# -------------------------
def plot_metric(df, metric_column, ylabel, output_file):
    """
    Plot only the mean value of metric_column for each implementation.
    All implementations share the same x-position per (m,n,k) scenario.
    """
    # Unique scenarios in order
    scen_df = df[["m", "n", "k"]].drop_duplicates().reset_index(drop=True)
    scen_df["label"] = scen_df.apply(lambda r: f"{r.m}×{r.n},k={r.k}", axis=1)
    x = np.arange(len(scen_df))

    plt.figure(figsize=(12, 6))
    for impl in sorted(df["implementation"].unique()):
        sub = df[df["implementation"] == impl]
        y = []
        for _, scen in scen_df.iterrows():
            row = sub[(sub.m == scen.m) & (sub.n == scen.n) & (sub.k == scen.k)]
            y.append(row[metric_column].values[0])
        plt.plot(x, y, marker="o", linestyle="-", label=impl)

    plt.xticks(x, scen_df["label"], rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    if os.path.exists(str(CONFIG["results_csv"])):
        df = pd.read_csv(str(CONFIG["results_csv"]))
        print(f"Loaded existing results from {CONFIG['results_csv']}")
    else:
        df = run_benchmarks(CONFIG)

    # Timing curve
    plot_metric(df, "time_mean", "Time (sec)", CONFIG["timing_plot"])
    # Absolute error curve
    plot_metric(df, "abs_error_mean", "Frobenius Error", CONFIG["error_plot"])
    # Relative error curve
    plot_metric(
        df,
        "rel_error_mean",
        "Relative Error",
        str(CONFIG["error_plot"]).replace(".png", "_rel.png"),
    )
    # Memory curve
    plot_metric(df, "memory_mean", "Peak Memory (MB)", CONFIG["memory_plot"])

    print("Done: CSV + plots generated.")
