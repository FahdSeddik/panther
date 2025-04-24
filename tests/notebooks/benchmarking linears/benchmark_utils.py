import time
import numpy as np
import torch
import torch._dynamo
import torch._inductor.config as config
import itertools
import pandas as pd

# Configure torch
config.max_autotune_gemm = False
torch._dynamo.config.cache_size_limit = 128

def is_valid_params(in_features, out_features, num_terms, low_rank):
    """
    Check if parameter combination is valid:
    A combination is invalid if 2 * num_terms * low_rank * (out_features + in_features) >= out_features * in_features
    """
    return 2 * num_terms * low_rank * (out_features + in_features) < out_features * in_features

class BenchmarkParams:
    def __init__(self, 
                 in_features=256, 
                 out_features=256,
                 num_terms=3,
                 low_rank=8,
                 batch_size=64, 
                 num_runs=100, 
                 warmup=15, 
                 device='cuda',
                 dtype=torch.float16):
        self.in_features = in_features
        self.out_features = out_features
        self.num_terms = num_terms
        self.low_rank = low_rank
        self.batch_size = batch_size
        self.num_runs = num_runs
        self.warmup = warmup
        self.device = device
        self.dtype = dtype

def benchmark_model(model, x, model_name, params):
    """
    Generic benchmarking function for any PyTorch model.
    
    Args:
        model: The PyTorch model to benchmark
        x: Input tensor
        model_name: Name of the model for logging
        params: Benchmark parameters
    
    Returns:
        Dictionary with benchmark results
    """
    # Compile the model
    model_compiled = torch.compile(
        model,
        backend="inductor",
        fullgraph=True,
        dynamic=False
    )
    
    # Benchmark forward pass
    print(f"\n=== {model_name} FORWARD PASS BENCHMARK ===")
    
    # Warmup runs for forward pass
    model_compiled.eval()
    with torch.no_grad():
        for _ in range(params.warmup):
            _ = model_compiled(x)
    
    torch.cuda.synchronize()
    
    # Actual timed runs for forward
    forward_times = []
    with torch.no_grad():
        for _ in range(params.num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model_compiled(x)
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            forward_times.append((end - start) * 1000)  # Convert to ms
    
    mean_forward = np.mean(forward_times)
    std_forward = np.std(forward_times)
    print(f"{model_name} forward: {mean_forward:.3f} ± {std_forward:.3f} ms")
    
    # Benchmark backward pass
    print(f"\n=== {model_name} BACKWARD PASS BENCHMARK ===")
    
    # Warmup runs for backward pass
    model_compiled.train()
    for _ in range(params.warmup):
        out = model_compiled(x)
        loss = out.sum()
        loss.backward()
        x.grad.zero_()
    
    torch.cuda.synchronize()
    
    # Actual timed runs for backward
    backward_times = []
    for _ in range(params.num_runs):
        out = model_compiled(x)
        loss = out.sum()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        backward_times.append((end - start) * 1000)  # Convert to ms
        x.grad.zero_()
    
    mean_backward = np.mean(backward_times)
    std_backward = np.std(backward_times)
    print(f"{model_name} backward: {mean_backward:.3f} ± {std_backward:.3f} ms")
    
    return {
        "forward": {
            "mean": mean_forward,
            "std": std_forward,
            "times": forward_times
        },
        "backward": {
            "mean": mean_backward,
            "std": std_backward,
            "times": backward_times
        }
    }

def benchmark_model_factory(model_factory, model_name, params):
    """
    Benchmark a model using a factory function.
    
    Args:
        model_factory: Function that creates the model
        model_name: Name of the model for logging
        params: Benchmark parameters
    
    Returns:
        Dictionary with benchmark results
    """
    # Create the model
    torch.manual_seed(42)
    model = model_factory(params)
    
    # Create input tensor for benchmarking
    x = torch.randn(params.batch_size, params.in_features, 
                  dtype=params.dtype, device=params.device, requires_grad=True)
    
    return benchmark_model(model, x, model_name, params)

if __name__ == "__main__":
    import torch.nn as nn
    from panther.nn import SKLinear, SKLinear_triton
    
    # Parameter combinations to test
    ratios = [(1, 128), (128, 1), (1, 1), (2, 1), (1, 2)]
    base_sizes = [256, 512, 1024, 8192, 16384]
    num_terms_options = [1, 2, 3]
    low_rank_options = [10, 15, 20, 50, 100, 150]
    
    # Define model factories
    def create_torch_linear(p):
        return nn.Linear(p.in_features, p.out_features, 
                        bias=True, device=p.device, dtype=p.dtype)
    
    def create_sklinear(p):
        return SKLinear(p.in_features, p.out_features, 
                       p.num_terms, p.low_rank, 
                       dtype=p.dtype, device=p.device)
    
    def create_sklinear_triton(p):
        return SKLinear_triton(p.in_features, p.out_features, 
                             p.num_terms, p.low_rank, 
                             dtype=p.dtype, device=p.device)
    
    models_to_benchmark = [
        (create_torch_linear, "torch.Linear"),
        (create_sklinear, "SKLinear"),
        (create_sklinear_triton, "SKLinear_triton")
    ]
    
    # Prepare data structure to store all results
    results_data = []
    
    # Iterate through all parameter combinations
    total_combinations = len(ratios) * len(base_sizes) * len(num_terms_options) * len(low_rank_options)
    current_combo = 0
    
    for ratio, base_size in itertools.product(ratios, base_sizes):
        ratio_in, ratio_out = ratio
        
        # Calculate actual dimensions based on ratio and base size
        if ratio_in == 1:
            in_features = base_size
            out_features = base_size * ratio_out
        else:
            out_features = base_size
            in_features = base_size * ratio_in
        
        for num_terms, low_rank in itertools.product(num_terms_options, low_rank_options):
            current_combo += 1
            print(f"\n\n{'='*20} COMBINATION {current_combo}/{total_combinations} {'='*20}")
            print(f"In features: {in_features}, Out features: {out_features}, Ratio: {ratio_in}:{ratio_out}")
            print(f"Base size: {base_size}, Num terms: {num_terms}, Low rank: {low_rank}")
            
            # Check if parameters are valid
            is_valid = is_valid_params(in_features, out_features, num_terms, low_rank)
            
            if not is_valid:
                print(f"INVALID COMBINATION: 2 * {num_terms} * {low_rank} * ({out_features} + {in_features}) >= {out_features} * {in_features}")
                print("Skipping benchmarks for this invalid combination")
                
                # Add invalid entry to results data
                for model_name in [m[1] for m in models_to_benchmark]:
                    results_data.append({
                        'model': model_name,
                        'in_features': in_features,
                        'out_features': out_features,
                        'ratio': f"{ratio_in}:{ratio_out}",
                        'base_size': base_size,
                        'num_terms': num_terms,
                        'low_rank': low_rank,
                        'forward_mean_ms': float('nan'),
                        'forward_std_ms': float('nan'),
                        'backward_mean_ms': float('nan'),
                        'backward_std_ms': float('nan'),
                        'is_valid': False,
                        'error': "Invalid parameter combination"
                    })
                continue
            
            # Create parameter object for this combination
            params = BenchmarkParams(
                in_features=in_features,
                out_features=out_features,
                num_terms=num_terms,
                low_rank=low_rank
            )
            
            all_results = {}
            for model_factory, model_name in models_to_benchmark:
                print(f"\n{'='*20} Benchmarking {model_name} {'='*20}")
                try:
                    results = benchmark_model_factory(model_factory, model_name, params)
                    all_results[model_name] = results
                    
                    # Add result to our data collection
                    results_data.append({
                        'model': model_name,
                        'in_features': in_features,
                        'out_features': out_features,
                        'ratio': f"{ratio_in}:{ratio_out}",
                        'base_size': base_size,
                        'num_terms': num_terms,
                        'low_rank': low_rank,
                        'forward_mean_ms': results['forward']['mean'],
                        'forward_std_ms': results['forward']['std'],
                        'backward_mean_ms': results['backward']['mean'],
                        'backward_std_ms': results['backward']['std'],
                        'is_valid': True
                    })
                except Exception as e:
                    print(f"Error benchmarking {model_name}: {e}")
                    # Add error entry to data
                    results_data.append({
                        'model': model_name,
                        'in_features': in_features,
                        'out_features': out_features,
                        'ratio': f"{ratio_in}:{ratio_out}",
                        'base_size': base_size,
                        'num_terms': num_terms,
                        'low_rank': low_rank,
                        'forward_mean_ms': float('nan'),
                        'forward_std_ms': float('nan'),
                        'backward_mean_ms': float('nan'),
                        'backward_std_ms': float('nan'),
                        'is_valid': True,
                        'error': str(e)
                    })
            
            # Print comparative summary for this combination
            if all_results:
                print("\n" + "="*60)
                print(f"{'='*20} SUMMARY FOR CURRENT COMBINATION {'='*20}")
                print("="*60)
                print(f"{'Model':<20} {'Forward (ms)':<25} {'Backward (ms)':<25}")
                print("-"*60)
                
                for model_name, results in all_results.items():
                    fwd = f"{results['forward']['mean']:.3f} ± {results['forward']['std']:.3f}"
                    bwd = f"{results['backward']['mean']:.3f} ± {results['backward']['std']:.3f}"
                    print(f"{model_name:<20} {fwd:<25} {bwd:<25}")
    
    # Create a DataFrame with all results
    df = pd.DataFrame(results_data)
    
    # Save results to CSV
    results_file = "benchmark_results.csv"
    df.to_csv(results_file, index=False)
    print(f"\nAll benchmark results saved to {results_file}")