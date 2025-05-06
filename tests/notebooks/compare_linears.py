def benchmark_forward(models, 
                      in_features, out_features, num_terms, low_rank, 
                      batch_size, num_runs, warmup, device='cuda'):
    """Benchmark forward pass for multiple compiled models with proper warmup."""
    # Create input tensor with appropriate size
    x = torch.randn(batch_size, in_features, dtype=dtype, device=device)
    
    # Store times for each model
    model_times = {}
    
    # Benchmark each model
    for model_name, model_compiled in models.items():
        # Set model to eval mode for forward pass
        model_compiled.eval()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup):
                _ = model_compiled(x)
        
        torch.cuda.synchronize()
        
        # Actual timed runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model_compiled(x)
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                times.append((end - start) * 1000)  # Convert to ms
        
        model_times[model_name] = times
    
    # Calculate and print times for each model
    print("\nForward Pass Times:")
    for model_name, times in model_times.items():
        mean_time = np.mean(times)
        print(f"{model_name} forward: {mean_time:.3f} ms")
    
    return model_times

# Function to vary input features and benchmark forward pass
def benchmark_forward_varying_in_features(in_feature_sizes, out_features, num_terms, low_rank, 
                                          batch_size, num_runs, warmup, device='cuda'):
    results = []
    
    for in_feat in in_feature_sizes:
        print(f"\nBenchmarking forward with in_features={in_feat}")
        
        # Create models for this iteration
        torch.manual_seed(42)
        torch_linear = nn.Linear(in_feat, out_features, bias=True, device=device, dtype=dtype)
        torch.manual_seed(42)
        sklinear_triton = SKLinear_triton(in_feat, out_features, num_terms, low_rank, dtype=dtype, device=device)
        torch.manual_seed(42)
        sklinear_normal = SKLinear(in_feat, out_features, num_terms, low_rank, dtype=dtype, device=device)
        
        # Compile the models
        models = {
            'torch.Linear': torch.compile(
                torch_linear,
                backend="inductor",
                fullgraph=True,
                dynamic=False
            ),
            'SKLinear (Triton)': torch.compile(
                sklinear_triton,
                backend="inductor",
                fullgraph=True,
                dynamic=False
            ),
            'SKLinear (Normal)': torch.compile(
                sklinear_normal,
                backend="inductor",
                fullgraph=True,
                dynamic=False
            )
        }
        
        # Run forward benchmark
        times = benchmark_forward(models, in_feat, out_features, num_terms, low_rank,
                                  batch_size, num_runs, warmup, device)
        
        # Calculate speedups relative to torch.Linear
        speedups = {
            model_name: np.mean(times['torch.Linear']) / np.mean(times[model_name]) 
            for model_name in times.keys() if model_name != 'torch.Linear'
        }
        
        results.append({
            'in_features': in_feat,
            'times': times,
            'speedups': speedups
        })
    
    return results

# Function to vary output features and benchmark forward pass
def benchmark_forward_varying_out_features(in_features, out_feature_sizes, num_terms, low_rank, 
                                           batch_size, num_runs, warmup, device='cuda'):
    results = []
    
    for out_feat in out_feature_sizes:
        print(f"\nBenchmarking forward with out_features={out_feat}")
        
        # Create models for this iteration
        torch.manual_seed(42)
        torch_linear = nn.Linear(in_features, out_feat, bias=True, device=device, dtype=dtype)
        torch.manual_seed(42)
        sklinear_triton = SKLinear_triton(in_features, out_feat, num_terms, low_rank, dtype=dtype, device=device)
        torch.manual_seed(42)
        sklinear_normal = SKLinear(in_features, out_feat, num_terms, low_rank, dtype=dtype, device=device)
        
        # Compile the models
        models = {
            'torch.Linear': torch.compile(
                torch_linear,
                backend="inductor",
                fullgraph=True,
                dynamic=False
            ),
            'SKLinear (Triton)': torch.compile(
                sklinear_triton,
                backend="inductor",
                fullgraph=True,
                dynamic=False
            ),
            'SKLinear (Normal)': torch.compile(
                sklinear_normal,
                backend="inductor",
                fullgraph=True,
                dynamic=False
            )
        }
        
        # Run forward benchmark
        times = benchmark_forward(models, in_features, out_feat, num_terms, low_rank,
                                  batch_size, num_runs, warmup, device)
        
        # Calculate speedups relative to torch.Linear
        speedups = {
            model_name: np.mean(times['torch.Linear']) / np.mean(times[model_name]) 
            for model_name in times.keys() if model_name != 'torch.Linear'
        }
        
        results.append({
            'out_features': out_feat,
            'times': times,
            'speedups': speedups
        })
    
    return results

# Function to plot the forward benchmark results
def plot_forward_benchmark_results(in_feature_results, out_feature_results):
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Forward Execution times for input features
    plt.subplot(2, 2, 1)
    in_sizes = [r['in_features'] for r in in_feature_results]
    
    for model_name in ['torch.Linear', 'SKLinear (Triton)', 'SKLinear (Normal)']:
        times = [np.mean(r['times'][model_name]) for r in in_feature_results]
        plt.plot(in_sizes, times, 'o-', label=model_name)
    
    plt.xlabel('Input Features')
    plt.ylabel('Forward Time (ms)')
    plt.title('Forward Pass Time vs Input Size')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Forward Speedup for input features (relative to torch.Linear)
    plt.subplot(2, 2, 2)
    
    for model_name in ['SKLinear (Triton)', 'SKLinear (Normal)']:
        speedups = [r['speedups'][model_name] for r in in_feature_results]
        plt.plot(in_sizes, speedups, 'D-', label=f'{model_name} Speedup')
    
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.xlabel('Input Features')
    plt.ylabel('Speedup (relative to torch.Linear)')
    plt.title('Forward Speedup vs Input Size')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Forward Execution times for output features
    plt.subplot(2, 2, 3)
    out_sizes = [r['out_features'] for r in out_feature_results]
    
    for model_name in ['torch.Linear', 'SKLinear (Triton)', 'SKLinear (Normal)']:
        times = [np.mean(r['times'][model_name]) for r in out_feature_results]
        plt.plot(out_sizes, times, 'o-', label=model_name)
    
    plt.xlabel('Output Features')
    plt.ylabel('Forward Time (ms)')
    plt.title('Forward Pass Time vs Output Size')
    plt.grid(True)
    plt.legend()
    
    # Plot 4: Forward Speedup for output features (relative to torch.Linear)
    plt.subplot(2, 2, 4)
    
    for model_name in ['SKLinear (Triton)', 'SKLinear (Normal)']:
        speedups = [r['speedups'][model_name] for r in out_feature_results]
        plt.plot(out_sizes, speedups, 'D-', label=f'{model_name} Speedup')
    
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.xlabel('Output Features')
    plt.ylabel('Speedup (relative to torch.Linear)')
    plt.title('Forward Speedup vs Output Size')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    return plt

# Define input and output feature sizes
input_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
output_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Run the forward benchmarks
print("Benchmarking forward with varying input features...")
forward_in_feature_results = benchmark_forward_varying_in_features(
    input_sizes, out_features, num_terms, low_rank, 
    batch_size, num_runs, warmup, device
)

print("\nBenchmarking forward with varying output features...")
forward_out_feature_results = benchmark_forward_varying_out_features(
    in_features, output_sizes, num_terms, low_rank, 
    batch_size, num_runs, warmup, device
)

# Plot the forward results
plt = plot_forward_benchmark_results(forward_in_feature_results, forward_out_feature_results)
plt.savefig('sklinear_forward_benchmark_results.png')
plt.show()

# Print summary
print("\n=== FORWARD SUMMARY ===")
print("Input Feature Scaling (Forward):")
for result in forward_in_feature_results:
    print(f"\nIn Features: {result['in_features']}")
    for model_name, speedup in result['speedups'].items():
        print(f"{model_name} Speedup: {speedup:.2f}x")

print("\nOutput Feature Scaling (Forward):")
for result in forward_out_feature_results:
    print(f"\nOut Features: {result['out_features']}")
    for model_name, speedup in result['speedups'].items():
        print(f"{model_name} Speedup: {speedup:.2f}x")