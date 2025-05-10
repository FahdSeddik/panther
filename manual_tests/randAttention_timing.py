import time
import numpy as np
import torch
import torch._dynamo
import torch._inductor.config as config
import itertools
import pandas as pd

# Configure torch
config.max_autotune_gemm = False
torch._dynamo.config.cache_size_limit = 2**16
torch._dynamo.config.accumulated_cache_size_limit = 2**16

def is_valid_params(embed_dim, num_heads, num_random_features):
    """
    Check if parameter combination is valid:
    embed_dim must be divisible by num_heads
    """
    return embed_dim % num_heads == 0

class BenchmarkParams:
    def __init__(self, 
                 embed_dim=256,
                 num_heads=8,
                 num_random_features=128,
                 batch_size=64, 
                 seq_length=32,
                 num_runs=200, 
                 warmup=15, 
                 device=torch.device("cuda"),
                 dtype=torch.float32):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_random_features = num_random_features
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_runs = num_runs
        self.warmup = warmup
        self.device = device
        self.dtype = dtype

def benchmark_model(model, inputs, model_name, params):
    """
    Generic benchmarking function for any PyTorch model.
    
    Args:
        model: The PyTorch model to benchmark
        inputs: Dictionary of input tensors
        model_name: Name of the model for logging
        params: Benchmark parameters
    
    Returns:
        Dictionary with benchmark results
    """
    # Compile the model
    # model_compiled = torch.compile(
    #     model,
    #     backend="inductor",
    #     fullgraph=True,
    #     dynamic=False
    # )
    model_compiled = model
    
    # Benchmark forward pass
    print(f"\n=== {model_name} FORWARD PASS BENCHMARK ===")
    
    # Warmup runs for forward pass
    model_compiled.eval()
    with torch.no_grad():
        for _ in range(params.warmup):
            _ = model_compiled(**inputs)
    
    torch.cuda.synchronize()
    
    # Actual timed runs for forward
    forward_times = []
    forward_memories = []
    with torch.no_grad():
        for _ in range(params.num_runs):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model_compiled(**inputs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            forward_times.append((end - start) * 1000)  # Convert to ms
            forward_memories.append(torch.cuda.max_memory_allocated() / (1024 * 1024))  # Convert to MB
    
    mean_forward = np.mean(forward_times)
    std_forward = np.std(forward_times)
    mean_forward_memory = np.mean(forward_memories)
    std_forward_memory = np.std(forward_memories)
    print(f"{model_name} forward: {mean_forward:.3f} ± {std_forward:.3f} ms, Memory: {mean_forward_memory:.2f} ± {std_forward_memory:.2f} MB")
    
    # Benchmark backward pass
    print(f"\n=== {model_name} BACKWARD PASS BENCHMARK ===")
    
    # Get query for backward
    query = inputs['query']
    
    # Warmup runs for backward pass
    model_compiled.train()
    for _ in range(params.warmup):
        out = model_compiled(**inputs)[0]
        loss = out.sum()
        loss.backward()
        query.grad.zero_()
    
    torch.cuda.synchronize()
    
    # Actual timed runs for backward
    backward_times = []
    backward_memories = []
    for _ in range(params.num_runs):
        out = model_compiled(**inputs)[0]
        loss = out.sum()
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        backward_times.append((end - start) * 1000)  # Convert to ms
        backward_memories.append(torch.cuda.max_memory_allocated() / (1024 * 1024))  # Convert to MB
        query.grad.zero_()
    
    mean_backward = np.mean(backward_times)
    std_backward = np.std(backward_times)
    mean_backward_memory = np.mean(backward_memories)
    std_backward_memory = np.std(backward_memories)
    print(f"{model_name} backward: {mean_backward:.3f} ± {std_backward:.3f} ms, Memory: {mean_backward_memory:.2f} ± {std_backward_memory:.2f} MB")
    
    return {
        "forward": {
            "mean": mean_forward,
            "std": std_forward,
            "times": forward_times,
            "memory_mb": mean_forward_memory,
            "memory_std": std_forward_memory,
            "memories": forward_memories
        },
        "backward": {
            "mean": mean_backward,
            "std": std_backward,
            "times": backward_times,
            "memory_mb": mean_backward_memory,
            "memory_std": std_backward_memory,
            "memories": backward_memories
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
    
    # Create input tensors for benchmarking
    query = torch.randn(params.batch_size, params.seq_length, params.embed_dim, 
                      dtype=params.dtype, device=params.device, requires_grad=True)
    key = torch.randn(params.batch_size, params.seq_length, params.embed_dim, 
                     dtype=params.dtype, device=params.device)
    value = torch.randn(params.batch_size, params.seq_length, params.embed_dim, 
                       dtype=params.dtype, device=params.device)
    
    inputs = {
        'query': query,
        'key': key,
        'value': value,
        # 'attention_mask': None
    }
    
    return benchmark_model(model, inputs, model_name, params)

if __name__ == "__main__":
    import torch.nn as nn
    from panther.nn.attention import RandMultiHeadAttention
    
    # Parameter combinations to test
    embed_dims = [128, 256, 512, 1024]
    num_heads_options = [4, 8, 16]
    # num_random_features_options = [64, 128, 256]
    num_random_features_options = [64]
    kernel_fn_options = ["softmax", "relu"]
    causal_options = [False]
    # causal_options = [False, True]
    seq_lens = [512, 1024, 2048, 4096, 8192]
    
    # Define model factories
    def create_attention(p):
        return RandMultiHeadAttention(
            embed_dim=p.embed_dim,
            num_heads=p.num_heads,
            num_random_features=p.num_random_features,
            dropout=0.0,
            kernel_fn=p.kernel_fn if hasattr(p, 'kernel_fn') else "softmax",
            iscausal=p.iscausal if hasattr(p, 'iscausal') else False,
            device=p.device,
            dtype=p.dtype
        )
    
    def create_torch_attention(p):
        # Create a wrapper class that applies the appropriate activation function
        class ActivatedMultiheadAttention(nn.Module):
            def __init__(self, mha, activation_type="softmax"):
                super().__init__()
                self.mha = mha
                self.activation_type = activation_type
            
            def forward(self, query, key, value, attention_mask=None):
                output, attn_weights = self.mha(query, key, value)
                
                # Apply activation function to match kernel_fn in RandMultiHeadAttention
                if self.activation_type == "relu":
                    return torch.relu(output), attn_weights
                # Default is softmax, which is already applied in the standard implementation
                return output, attn_weights
        
        mha = torch.nn.MultiheadAttention(
            embed_dim=p.embed_dim,
            num_heads=p.num_heads,
            dropout=0.0,
            batch_first=True,  # Since your inputs are [batch, seq, dim]
            device=p.device,
            dtype=p.dtype
        )
        
        # Wrap with the appropriate activation
        kernel_fn = p.kernel_fn if hasattr(p, 'kernel_fn') else "softmax"
        return ActivatedMultiheadAttention(mha, activation_type=kernel_fn)
    
    models_to_benchmark = [
        (create_torch_attention, "attention")
    ]
    
    # Prepare data structure to store all results
    results_data = []
    
    # Iterate through all parameter combinations
    total_combinations = len(embed_dims) * len(num_heads_options) * len(num_random_features_options) * len(kernel_fn_options) * len(causal_options) * len(seq_lens)
    current_combo = 0
    
    for embed_dim, num_heads, num_random_features, kernel_fn, iscausal, seq_length in itertools.product(
        embed_dims, num_heads_options, num_random_features_options, kernel_fn_options, causal_options, seq_lens
    ):
        current_combo += 1
        print(f"\n\n{'='*20} COMBINATION {current_combo}/{total_combinations} {'='*20}")
        print(f"Embed dimension: {embed_dim}, Num heads: {num_heads}, Num random features: {num_random_features}")
        print(f"Kernel function: {kernel_fn}, Causal: {iscausal}, Sequence length: {seq_length}")
        
        # Check if parameters are valid
        is_valid = is_valid_params(embed_dim, num_heads, num_random_features)
        
        if not is_valid:
            print(f"INVALID COMBINATION: {embed_dim} is not divisible by {num_heads}")
            print("Skipping benchmarks for this invalid combination")
            
            # Add invalid entry to results data
            for model_name in [m[1] for m in models_to_benchmark]:
                results_data.append({
                    'model': model_name,
                    'embed_dim': embed_dim,
                    'num_heads': num_heads,
                    'num_random_features': num_random_features,
                    'kernel_fn': kernel_fn,
                    'iscausal': iscausal,
                    'seq_length': seq_length,
                    'forward_mean_ms': float('nan'),
                    'forward_std_ms': float('nan'),
                    'backward_mean_ms': float('nan'),
                    'backward_std_ms': float('nan'),
                    'forward_memory_mb': float('nan'),
                    'backward_memory_mb': float('nan'),
                    'is_valid': False,
                    'error': "Invalid parameter combination"
                })
            continue
        
        # Create parameter object for this combination
        params = BenchmarkParams(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_random_features=num_random_features,
            seq_length=seq_length
        )
        # Add the new parameters
        params.kernel_fn = kernel_fn
        params.iscausal = iscausal
        
        all_results = {}
        for model_factory, model_name in models_to_benchmark:
            print(f"\n{'='*20} Benchmarking {model_name} {'='*20}")
            try:
                results = benchmark_model_factory(model_factory, model_name, params)
                all_results[model_name] = results
                
                # Add result to our data collection
                results_data.append({
                    'model': model_name,
                    'embed_dim': embed_dim,
                    'num_heads': num_heads,
                    'num_random_features': num_random_features,
                    'kernel_fn': kernel_fn,
                    'iscausal': iscausal,
                    'seq_length': seq_length,
                    'forward_mean_ms': results['forward']['mean'],
                    'forward_std_ms': results['forward']['std'],
                    'backward_mean_ms': results['backward']['mean'],
                    'backward_std_ms': results['backward']['std'],
                    'forward_memory_mb': results['forward']['memory_mb'],
                    'backward_memory_mb': results['backward']['memory_mb'],
                    'is_valid': True
                })
            except Exception as e:
                print(f"Error benchmarking {model_name}: {e}")
                # Add error entry to data
                results_data.append({
                    'model': model_name,
                    'embed_dim': embed_dim,
                    'num_heads': num_heads,
                    'num_random_features': num_random_features,
                    'kernel_fn': kernel_fn, 
                    'iscausal': iscausal,
                    'seq_length': seq_length,
                    'forward_mean_ms': float('nan'),
                    'forward_std_ms': float('nan'),
                    'backward_mean_ms': float('nan'),
                    'backward_std_ms': float('nan'),
                    'forward_memory_mb': float('nan'),
                    'backward_memory_mb': float('nan'),
                    'is_valid': True,
                    'error': str(e)
                })
        
        # Print comparative summary for this combination
        if all_results:
            print("\n" + "="*60)
            print(f"{'='*20} SUMMARY FOR CURRENT COMBINATION {'='*20}")
            print("="*60)
            print(f"{'Model':<30} {'Forward (ms)':<25} {'Backward (ms)':<25} {'Forward Memory (MB)':<25} {'Backward Memory (MB)':<25}")
            print("-"*60)
            
            for model_name, results in all_results.items():
                fwd = f"{results['forward']['mean']:.3f} ± {results['forward']['std']:.3f}"
                bwd = f"{results['backward']['mean']:.3f} ± {results['backward']['std']:.3f}"
                fwd_mem = f"{results['forward']['memory_mb']:.2f}"
                bwd_mem = f"{results['backward']['memory_mb']:.2f}"
                print(f"{model_name:<30} {fwd:<25} {bwd:<25} {fwd_mem:<25} {bwd_mem:<25}")
    
    # Create a DataFrame with all results
    df = pd.DataFrame(results_data)
    
    # Save results to CSV
    results_file = "attention_benchmark_results.csv"
    df.to_csv(results_file, index=False)
    print(f"\nAll benchmark results saved to {results_file}")