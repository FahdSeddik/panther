"""
plot_benchmark_results.py

Load the CSV produced by benchmark_cqrrpt.py and generate:
  1. Runtime vs. n (for each m, with one curve per algorithm)
  2. Factorization error vs. n (same layout)
  3. Orthogonality error vs. n (same layout)

Usage:
    python plot_benchmark_results.py cqrrpt_benchmark_results.csv

"""

import sys

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(df, metric_col, ylabel, log_scale=False, output_filename=None):
    """
    For each fixed m, plot metric_col vs. n for each algorithm.
    If log_scale=True, y-axis is log-scaled.
    """
    ms = sorted(df["m"].unique())
    algorithms = sorted(df["algorithm"].unique())

    # Create one figure per m
    for m in ms:
        sub = df[df["m"] == m]
        plt.figure(figsize=(8, 5))
        for alg in algorithms:
            sub_alg = sub[sub["algorithm"] == alg]
            # sort by n
            sub_alg = sub_alg.sort_values("n")
            n_values = sub_alg["n"].values
            y_values = sub_alg[metric_col].values
            plt.plot(n_values, y_values, marker="o", label=alg)

        plt.title(f"{ylabel} vs. n  (m = {m})")
        plt.xlabel("n")
        plt.ylabel(ylabel)
        plt.xticks(n_values, n_values, rotation=45)
        if log_scale:
            plt.yscale("log")
        plt.grid(True, which="both", ls="--", lw=0.5)
        plt.legend()
        plt.tight_layout()
        if output_filename:
            plt.savefig(f"{output_filename}_m{m}.png")
        else:
            plt.show()


def main(csv_path):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Ensure correct dtypes
    df["m"] = df["m"].astype(int)
    df["n"] = df["n"].astype(int)
    df["best_time_sec"] = df["best_time_sec"].astype(float)
    df["memory_bytes"] = df["memory_bytes"].astype(float)
    df["factor_err"] = df["factor_err"].astype(float)
    df["orth_err"] = df["orth_err"].astype(float)
    df["recon_err"] = df["recon_err"].astype(float)

    # Plot runtime (best_time_sec). Use log scale on y for readability.
    plot_metric(
        df,
        metric_col="best_time_sec",
        ylabel="Log Runtime (sec)",
        log_scale=True,
        output_filename="runtime",
    )

    # Plot factorization error. Use log scale on y.
    plot_metric(
        df,
        metric_col="factor_err",
        ylabel="Log Relative Factorization Error",
        log_scale=True,
        output_filename="factor_error",
    )

    # Plot orthogonality error. Use log scale on y.
    plot_metric(
        df,
        metric_col="orth_err",
        ylabel="Log Orthogonality Error (‖I - QᵀQ‖_F)",
        log_scale=True,
        output_filename="orthogonality_error",
    )

    plot_metric(
        df,
        metric_col="recon_err",
        ylabel="Log Reconstruction Error",
        log_scale=True,
        output_filename="recon_error",
    )

    print(
        "Plots saved as PNG files: runtime_m<...>.png, factor_error_m<...>.png, orthogonality_error_m<...>.png"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_benchmark_results.py <csv_file>")
        sys.exit(1)
    csv_path = sys.argv[1]
    main(csv_path)
