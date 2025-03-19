import time
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Enable error suppression to fall back to eager mode if compilation fails
import torch._dynamo
import torch.nn as nn

from panther.nn import SKLinear

torch._dynamo.config.suppress_errors = True


class BenchmarkModel(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, num_terms: int, low_rank: int
    ):
        super().__init__()
        self.layer = SKLinear(
            in_features=in_features,
            out_features=out_features,
            num_terms=num_terms,
            low_rank=low_rank,
        )

    def forward(self, x):
        return self.layer(x)


def benchmark_compile(
    model: nn.Module,
    input_tensor: torch.Tensor,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    fullgraph: bool = True,
    dynamic: bool = False,
    options: Optional[Dict[str, Any]] = None,
    num_runs: int = 50,  # Reduced for faster benchmarking
    warmup: int = 5,  # Reduced for faster benchmarking
    compile_name: str = "Unknown",
    device: str = "cpu",
) -> Dict[str, Union[float, str]]:
    """
    Benchmark a model with torch.compile using different backends and parameters.

    Args:
        model: The model to benchmark
        input_tensor: Input tensor for the model
        backend: The backend to use for torch.compile
        mode: The mode to use for torch.compile
        fullgraph: Whether to use fullgraph mode
        dynamic: Whether to use dynamic shapes
        options: Additional options for torch.compile
        num_runs: Number of runs for benchmarking
        warmup: Number of warmup runs
        compile_name: Name for this compilation configuration

    Returns:
        Dictionary with benchmark results
    """
    print(f"Compiling model with {compile_name} configuration...")

    # Create a copy of the model for compilation
    try:
        compiled_model = torch.compile(
            model,
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            options=options,
        )
        print(f"Successfully compiled model with {compile_name}")
    except Exception as e:
        print(f"Compilation failed for {compile_name}, falling back to eager mode: {e}")
        compiled_model = model  # Fall back to the original model

    # Warmup
    print(f"Running warmup for {compile_name}...")
    for _ in range(warmup):
        try:
            _ = compiled_model(input_tensor)
        except Exception as e:
            print(f"Error during warmup for {compile_name}: {e}")
            print("Falling back to original model")
            compiled_model = model
            # Re-run warmup with original model
            for _ in range(warmup):
                _ = compiled_model(input_tensor)
            break

    # Benchmark forward pass
    print(f"Benchmarking forward pass for {compile_name}...")
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = compiled_model(input_tensor)
    torch.cuda.synchronize() if device == "cuda" else None
    forward_time = (time.perf_counter() - start_time) / num_runs

    # Benchmark backward pass
    print(f"Benchmarking backward pass for {compile_name}...")
    criterion = nn.MSELoss()
    target = torch.randn_like(compiled_model(input_tensor))

    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.perf_counter()
    for _ in range(num_runs):
        output = compiled_model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
    torch.cuda.synchronize() if device == "cuda" else None
    backward_time = (time.perf_counter() - start_time) / num_runs

    print(f"Completed benchmarking for {compile_name}")
    return {
        "name": compile_name,
        "backend": str(backend) if backend else "default",
        "mode": str(mode) if mode else "default",
        "fullgraph": str(fullgraph),
        "dynamic": str(dynamic),
        "forward_time": forward_time,
        "backward_time": backward_time,
        "total_time": forward_time + backward_time,
    }


def benchmark_baseline(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 50,
    warmup: int = 5,
    device: str = "cpu",
) -> Dict[str, Union[float, str]]:
    """Benchmark the baseline model without compilation"""
    print("Running baseline benchmark (no compilation)...")

    # Warmup
    for _ in range(warmup):
        _ = model(input_tensor)

    # Benchmark forward pass
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = model(input_tensor)
    torch.cuda.synchronize() if device == "cuda" else None
    forward_time = (time.perf_counter() - start_time) / num_runs

    # Benchmark backward pass
    criterion = nn.MSELoss()
    target = torch.randn_like(model(input_tensor))

    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.perf_counter()
    for _ in range(num_runs):
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
    torch.cuda.synchronize() if device == "cuda" else None
    backward_time = (time.perf_counter() - start_time) / num_runs

    print("Completed baseline benchmark")
    return {
        "name": "Baseline (No Compile)",
        "backend": "None",
        "mode": "None",
        "fullgraph": "N/A",
        "dynamic": "N/A",
        "forward_time": forward_time,
        "backward_time": backward_time,
        "total_time": forward_time + backward_time,
    }


def plot_results(results: List[Dict[str, Union[float, str]]]):
    """Plot the benchmark results for all backends together"""
    try:
        # Sort results by total time
        results = sorted(
            results,
            key=lambda x: float(x["total_time"])
            if isinstance(x["total_time"], (int, float))
            else 0,
        )

        # Extract data for plotting
        names = [result["name"] for result in results]
        forward_times = [
            float(result["forward_time"]) * 1000 for result in results
        ]  # Convert to ms
        backward_times = [
            float(result["backward_time"]) * 1000 for result in results
        ]  # Convert to ms

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create stacked bar chart
        bar_width = 0.6
        x = np.arange(len(names))

        ax.bar(x, forward_times, bar_width, label="Forward Pass", color="skyblue")
        ax.bar(
            x,
            backward_times,
            bar_width,
            bottom=forward_times,
            label="Backward Pass",
            color="salmon",
        )

        # Add labels and title
        ax.set_xlabel("Compilation Configuration")
        ax.set_ylabel("Time (ms)")
        ax.set_title("SKLinear Performance with Different torch.compile Configurations")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()

        # Add speedup annotations
        baseline_idx = next(
            (
                i
                for i, result in enumerate(results)
                if result["name"] == "Baseline (No Compile)"
            ),
            0,
        )
        baseline_time = float(results[baseline_idx]["total_time"])

        for i, result in enumerate(results):
            # Fix the sum() issue - we're just adding two floats
            total_time_ms = forward_times[i] + backward_times[i]
            speedup = baseline_time / float(result["total_time"])
            ax.text(
                i,
                total_time_ms + 0.5,
                f"{speedup:.2f}x",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig("sklinear_compile_benchmark_all.png")
        print("Saved plot to sklinear_compile_benchmark_all.png")
        plt.show()
    except Exception as e:
        print(f"Error plotting results: {e}")
        print("Skipping plot generation")


def plot_results_by_backend(
    results: List[Dict[str, Union[float, str]]],
    baseline_result: Dict[str, Union[float, str]],
    device: str = "cpu",
):
    """Plot the benchmark results separately for each backend"""
    try:
        # Group results by backend
        backends = set(
            result["backend"] for result in results if result["backend"] != "None"
        )

        # Get baseline time for speedup calculations
        baseline_time = float(baseline_result["total_time"])

        for backend in backends:
            # Filter results for this backend and add baseline
            backend_results = [baseline_result] + [
                r for r in results if r["backend"] == backend
            ]

            # Sort results by total time
            backend_results = sorted(
                backend_results,
                key=lambda x: float(x["total_time"])
                if isinstance(x["total_time"], (int, float))
                else 0,
            )

            # Extract data for plotting
            names = [result["name"] for result in backend_results]
            forward_times = [
                float(result["forward_time"]) * 1000 for result in backend_results
            ]  # Convert to ms
            backward_times = [
                float(result["backward_time"]) * 1000 for result in backend_results
            ]  # Convert to ms

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Create stacked bar chart
            bar_width = 0.6
            x = np.arange(len(names))

            ax.bar(x, forward_times, bar_width, label="Forward Pass", color="skyblue")
            ax.bar(
                x,
                backward_times,
                bar_width,
                bottom=forward_times,
                label="Backward Pass",
                color="salmon",
            )

            # Add labels and title
            ax.set_xlabel("Compilation Configuration")
            ax.set_ylabel("Time (ms)")
            ax.set_title(
                f"SKLinear Performance with {backend.capitalize()} Backend on {device}"
            )
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.legend()

            # Add speedup annotations
            for i, result in enumerate(backend_results):
                total_time_ms = forward_times[i] + backward_times[i]
                speedup = baseline_time / float(result["total_time"])
                ax.text(
                    i,
                    total_time_ms + 0.5,
                    f"{speedup:.2f}x",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()
            filename = f"sklinear_compile_benchmark_{backend}.png"
            plt.savefig(filename)
            print(f"Saved plot to {filename}")
            plt.show()

    except Exception as e:
        print(f"Error plotting results by backend: {e}")
        print("Skipping plot generation by backend")


def check_triton_compatibility():
    """
    Check if Triton is available and compatible with PyTorch Inductor.

    Returns:
        tuple: (is_available, message)
            is_available (bool): True if Triton is available and compatible
            message (str): Descriptive message about Triton status
    """
    try:
        import triton

        triton_version = triton.__version__
        print(f"Triton version: {triton_version}")

        # Simple test to verify Triton works with PyTorch
        if torch.cuda.is_available():
            try:
                # Try a minimal test with inductor
                x = torch.randn(10, 10, device="cuda")

                def simple_fn(x):
                    return x + x

                test_model = torch.compile(simple_fn, backend="inductor")
                test_model(x)  # Run once to trigger compilation
                return (
                    True,
                    f"Triton {triton_version} is available and compatible with PyTorch Inductor",
                )
            except Exception as e:
                return (
                    False,
                    f"Triton {triton_version} is installed but not compatible with PyTorch Inductor: {str(e)}",
                )
        else:
            return False, "CUDA is not available, Inductor backend requires CUDA"
    except ImportError:
        return False, "Triton is not installed"
    except Exception as e:
        return False, f"Error checking Triton: {str(e)}"


def run_benchmarks(device: str = "cpu"):
    """Run all benchmarks"""
    # Set up parameters
    in_features = 4096
    out_features = 2048
    num_terms = 4
    low_rank = 64
    batch_size = 32

    # Reduce number of runs for faster benchmarking
    num_runs = 50
    warmup = 5

    # Create model and input tensor
    print(f"Running benchmarks on device: {device}")
    model = BenchmarkModel(in_features, out_features, num_terms, low_rank).to(device)
    input_tensor = torch.randn(batch_size, in_features, device=device)

    inductor_available, triton_message = True, None
    # inductor_available, triton_message = False, None
    # if device == "cuda":
    #     # Check if Triton is available and compatible with PyTorch Inductor
    #     inductor_available, triton_message = check_triton_compatibility()
    #     print(triton_message)
    # else:
    #     inductor_available = True
    #     triton_message = "Triton is not available on CPU"

    # Define compilation configurations to test
    configs = [
        # Baseline (no compilation)
        {"name": "Baseline (No Compile)"},
    ]

    # Define the backends and modes
    backends = []

    # Add the working backends
    # backends.extend(["eager", "aot_eager"])

    # Only add inductor if compatible
    if inductor_available:
        backends.append("inductor")

    # if device == "cuda":
        # backends.extend(["cudagraphs"])

    # Add other backends that don't depend on Triton
    # backends.extend(["onnxrt", "openxla", "tvm"])

    # modes = ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]
    modes = ["default"]

    # Add all combinations of backends and modes
    for backend in backends:
        if backend == "cudagraphs" or backend == "tvm" or backend == "eager" or backend == "aot_eager":
            configs.append(
                {
                    "backend": backend,
                    "fullgraph": True,
                    "dynamic": False,
                    "name": f"{backend.capitalize()}",
                }
            )
            continue
        # Add all modes for each backend
        for mode in modes:
            configs.append(
                {
                    "backend": backend,
                    "mode": mode,
                    "fullgraph": True,
                    "dynamic": True,
                    "name": f"{backend.capitalize()} ({mode})",
                }
            )

    # Run benchmarks
    results = []

    # First run baseline without compilation
    print("Running baseline benchmark...")
    baseline_result = benchmark_baseline(
        model, input_tensor, num_runs=num_runs, warmup=warmup, device=device
    )
    results.append(baseline_result)

    # Then run all compilation configurations
    for config in configs:
        if config.get("name") == "Baseline (No Compile)":
            continue  # Skip baseline as we already ran it

        # Skip inductor configurations if not available
        backend = config.get("backend")
        if backend == "inductor" and not inductor_available:
            print(f"Skipping {config.get('name')} - Inductor not compatible")
            continue

        name = config.pop("name")
        print(f"Running benchmark for {name}...")
        try:
            result = benchmark_compile(
                model=model,
                input_tensor=input_tensor,
                compile_name=name,
                num_runs=num_runs,
                warmup=warmup,
                device=device,
                **config,
            )
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {name}: {e}")
            # Add a placeholder result for failed benchmarks
            results.append(
                {
                    "name": f"{name} (Failed)",
                    "backend": str(config.get("backend", "unknown")),
                    "mode": str(config.get("mode", "default")),
                    "fullgraph": str(config.get("fullgraph", True)),
                    "dynamic": str(config.get("dynamic", False)),
                    "forward_time": 0.0,
                    "backward_time": 0.0,
                    "total_time": 0.0,
                }
            )

    # Convert results to DataFrame for easy viewing
    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df)

    # Plot all results together
    try:
        plot_results(results)
    except Exception as e:
        print(f"Error plotting all results: {e}")
        print("Results DataFrame:")
        print(df)

    # Plot results by backend
    try:
        plot_results_by_backend(results, baseline_result, device=device)
    except Exception as e:
        print(f"Error plotting results by backend: {e}")
        print("Results DataFrame:")
        print(df)

    # Save results to CSV
    try:
        df.to_csv("sklinear_compile_benchmark_results.csv", index=False)
        print("Saved results to sklinear_compile_benchmark_results.csv")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

    # Save results to CSV by backend
    try:
        for backend in set(
            result["backend"] for result in results if result["backend"] != "None"
        ):
            backend_df = pd.DataFrame(
                [baseline_result] + [r for r in results if r["backend"] == backend]
            )
            backend_df.to_csv(
                f"sklinear_compile_benchmark_results_{backend}.csv", index=False
            )
            print(f"Saved results to sklinear_compile_benchmark_results_{backend}.csv")
    except Exception as e:
        print(f"Error saving results to CSV by backend: {e}")

    return df

def test_correctness(device="cuda"):
    # Set up parameters
    in_features = 4096
    out_features = 2048
    num_terms = 4
    low_rank = 64
    batch_size = 32

    # Reduce number of runs for faster benchmarking
    num_runs = 50
    warmup = 5

    input_tensor = torch.randn(batch_size, in_features, device=device)
    baseline_input = input_tensor.clone().detach()
    compile_input = input_tensor.clone().detach()
    
    # forward and backward pass of the baseline
    model = BenchmarkModel(in_features, out_features, num_terms, low_rank).to(device)

    criterion_baseline = nn.MSELoss()
    target_baseline = torch.randn_like(model(baseline_input))

    forward_out_baseline = model(baseline_input)
    loss = criterion_baseline(forward_out_baseline, target_baseline)
    loss.backward()

    # forward and backward pass of the compiled (ignore warmup since performance is not important here)
    compiled_model = torch.compile(
            model,
            backend="inductor",
            fullgraph=True,
            dynamic=True,
        )

    criterion_compiled = nn.MSELoss()
    target_compiled = torch.randn_like(model(compile_input))

    forward_out_compiled = model(compile_input)
    loss = criterion_compiled(forward_out_compiled, target_compiled)
    loss.backward()

    # start asserting on the outputs and grads
    assert torch.allclose(forward_out_baseline, forward_out_compiled)
    assert 

if __name__ == "__main__":
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    device = "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available, please set device to 'cuda'")
    results_df = run_benchmarks(device=device)