Performance Optimization
========================

This tutorial covers techniques to maximize performance when using Panther's sketched layers, including GPU acceleration, memory optimization, and Tensor Core usage.

GPU Acceleration Fundamentals
------------------------------

**Basic GPU Usage**

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   import time
   
   # Check GPU availability
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"Using device: {device}")
   
   if torch.cuda.is_available():
       print(f"GPU: {torch.cuda.get_device_name()}")
       print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
   
   # Create models on GPU
   input_dim, hidden_dim, output_dim = 2048, 1024, 512
   
   standard_model = nn.Sequential(
       nn.Linear(input_dim, hidden_dim),
       nn.ReLU(),
       nn.Linear(hidden_dim, output_dim)
   ).to(device)
   
   sketched_model = nn.Sequential(
       pr.nn.SKLinear(input_dim, hidden_dim, num_terms=8, low_rank=128),
       nn.ReLU(),
       pr.nn.SKLinear(hidden_dim, output_dim, num_terms=6, low_rank=96)
   ).to(device)
   
   # Benchmark forward pass
   batch_size = 256
   x = torch.randn(batch_size, input_dim, device=device)
   
   def benchmark_model(model, input_tensor, num_runs=100, warmup=10):
       \"\"\"Benchmark model inference time.\"\"\""
       model.eval()
       
       # Warmup
       with torch.no_grad():
           for _ in range(warmup):
               _ = model(input_tensor)
       
       if device.type == 'cuda':
           torch.cuda.synchronize()
       
       # Timing
       start_time = time.time()
       with torch.no_grad():
           for _ in range(num_runs):
               _ = model(input_tensor)
               if device.type == 'cuda':
                   torch.cuda.synchronize()
       
       elapsed = time.time() - start_time
       return elapsed / num_runs
   
   standard_time = benchmark_model(standard_model, x)
   sketched_time = benchmark_model(sketched_model, x)
   
   print(f"\\nInference Benchmarks (batch_size={batch_size}):")
   print(f"Standard model: {standard_time*1000:.2f} ms")
   print(f"Sketched model: {sketched_time*1000:.2f} ms")
   print(f"Speedup: {standard_time/sketched_time:.2f}x")

**Memory Usage Optimization**

.. code-block:: python

   def analyze_memory_usage(model, input_shape, batch_sizes=[32, 64, 128, 256]):
       \"\"\"Analyze memory usage across different batch sizes.\"\"\""
       
       model = model.to(device)
       results = {}
       
       for batch_size in batch_sizes:
           if device.type == 'cuda':
               torch.cuda.empty_cache()
               torch.cuda.reset_peak_memory_stats()
           
           x = torch.randn(batch_size, *input_shape, device=device)
           
           # Forward pass
           with torch.no_grad():
               y = model(x)
           
           if device.type == 'cuda':
               memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
               results[batch_size] = {
                   'memory_mb': memory_used,
                   'memory_per_sample': memory_used / batch_size
               }
           else:
               results[batch_size] = {'memory_mb': 0, 'memory_per_sample': 0}
           
           del x, y
       
       return results
   
   # Compare memory usage
   input_shape = (2048,)
   
   print("Memory Usage Analysis:")
   print("\\nStandard Model:")
   standard_memory = analyze_memory_usage(standard_model, input_shape)
   for bs, stats in standard_memory.items():
       print(f"  Batch {bs}: {stats['memory_mb']:.1f} MB ({stats['memory_per_sample']:.2f} MB/sample)")
   
   print("\\nSketched Model:")
   sketched_memory = analyze_memory_usage(sketched_model, input_shape)
   for bs, stats in sketched_memory.items():
       print(f"  Batch {bs}: {stats['memory_mb']:.1f} MB ({stats['memory_per_sample']:.2f} MB/sample)")
   
   # Calculate memory savings
   if device.type == 'cuda':
       memory_savings = {}
       for bs in standard_memory:
           standard_mem = standard_memory[bs]['memory_mb']
           sketched_mem = sketched_memory[bs]['memory_mb']
           savings = (1 - sketched_mem / standard_mem) * 100 if standard_mem > 0 else 0
           memory_savings[bs] = savings
       
       print("\\nMemory Savings:")
       for bs, savings in memory_savings.items():
           print(f"  Batch {bs}: {savings:.1f}%")

Tensor Core Optimization
-------------------------

**Understanding Tensor Cores**

Modern NVIDIA GPUs (V100, A100, RTX series) include Tensor Cores that can dramatically accelerate matrix operations when certain conditions are met:

* Input dimensions are multiples of 8 (FP16) or 16 (INT8)
* Data types are FP16, BF16, or INT8
* Matrix operations are sufficiently large

.. code-block:: python

   def optimize_for_tensor_cores(in_features, out_features, num_terms, low_rank):
       \"\"\"Optimize dimensions for Tensor Core usage.\"\"\""
       
       # Round dimensions to multiples of 8 for FP16 Tensor Cores
       def round_to_multiple(x, multiple=8):
           return ((x + multiple - 1) // multiple) * multiple
       
       optimized_in = round_to_multiple(in_features)
       optimized_out = round_to_multiple(out_features)
       optimized_rank = round_to_multiple(low_rank)
       
       # Ensure we don't exceed original dimensions significantly
       if optimized_in > in_features * 1.2:
           optimized_in = in_features
       if optimized_out > out_features * 1.2:
           optimized_out = out_features
       
       return optimized_in, optimized_out, num_terms, optimized_rank
   
   # Example: Create Tensor Core optimized layers
   class TensorCoreOptimizedModel(nn.Module):
       def __init__(self, layer_configs):
           super().__init__()
           
           layers = []
           for config in layer_configs:
               in_feat, out_feat, terms, rank = optimize_for_tensor_cores(
                   config['in_features'], config['out_features'],
                   config['num_terms'], config['low_rank']
               )
               
               layer = pr.nn.SKLinear(in_feat, out_feat, num_terms=terms, low_rank=rank)
               layers.extend([layer, nn.ReLU()])
           
           self.network = nn.Sequential(*layers)
       
       def forward(self, x):
           return self.network(x)
   
   # Create optimized model
   configs = [
       {'in_features': 1000, 'out_features': 500, 'num_terms': 8, 'low_rank': 64},
       {'in_features': 500, 'out_features': 250, 'num_terms': 6, 'low_rank': 48},
       {'in_features': 250, 'out_features': 10, 'num_terms': 4, 'low_rank': 32}
   ]
   
   optimized_model = TensorCoreOptimizedModel(configs).to(device)

**Mixed Precision Training with Tensor Cores**

.. code-block:: python

   def train_with_tensor_cores(model, train_loader, num_epochs=5):
       \"\"\"Train model using mixed precision for Tensor Core acceleration.\"\"\""
       
       model = model.to(device)
       criterion = nn.CrossEntropyLoss()
       optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
       
       # Enable mixed precision training
       scaler = torch.cuda.amp.GradScaler()
       
       model.train()
       
       for epoch in range(num_epochs):
           epoch_loss = 0
           epoch_time = time.time()
           
           for batch_idx, (data, target) in enumerate(train_loader):
               data, target = data.to(device), target.to(device)
               
               optimizer.zero_grad()
               
               # Forward pass with autocast for FP16
               with torch.cuda.amp.autocast():
                   output = model(data)
                   loss = criterion(output, target)
               
               # Backward pass with gradient scaling
               scaler.scale(loss).backward()
               scaler.step(optimizer)
               scaler.update()
               
               epoch_loss += loss.item()
               
               if batch_idx % 50 == 0:
                   print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
           
           epoch_time = time.time() - epoch_time
           avg_loss = epoch_loss / len(train_loader)
           
           print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")

Advanced Memory Management
--------------------------

**Gradient Accumulation for Large Models**

.. code-block:: python

   class MemoryEfficientTrainer:
       \"\"\"Trainer with advanced memory management.\"\"\""
       
       def __init__(self, model, accumulation_steps=4, max_memory_mb=8000):
           self.model = model
           self.accumulation_steps = accumulation_steps
           self.max_memory_mb = max_memory_mb
           
       def train_epoch(self, train_loader, optimizer, criterion):
           \"\"\"Train one epoch with gradient accumulation and memory monitoring.\"\"\""
           
           self.model.train()
           optimizer.zero_grad()
           
           accumulated_loss = 0
           step_count = 0
           
           for batch_idx, (data, target) in enumerate(train_loader):
               # Check memory usage
               if device.type == 'cuda':
                   current_memory = torch.cuda.memory_allocated() / 1024**2
                   if current_memory > self.max_memory_mb:
                       print(f"Warning: High memory usage ({current_memory:.1f} MB)")
                       torch.cuda.empty_cache()
               
               data, target = data.to(device), target.to(device)
               
               # Forward pass
               with torch.cuda.amp.autocast():
                   output = self.model(data)
                   loss = criterion(output, target) / self.accumulation_steps
               
               # Backward pass
               scaler.scale(loss).backward()
               accumulated_loss += loss.item()
               
               # Update weights every accumulation_steps
               if (batch_idx + 1) % self.accumulation_steps == 0:
                   scaler.step(optimizer)
                   scaler.update()
                   optimizer.zero_grad()
                   step_count += 1
               
               # Memory cleanup
               del data, target, output, loss
               
               # Periodic garbage collection
               if batch_idx % 100 == 0:
                   import gc
                   gc.collect()
                   if device.type == 'cuda':
                       torch.cuda.empty_cache()
           
           return accumulated_loss / len(train_loader)

**Dynamic Batch Size Adjustment**

.. code-block:: python

   class AdaptiveBatchSizeTrainer:
       \"\"\"Trainer that adapts batch size based on available memory.\"\"\""
       
       def __init__(self, model, initial_batch_size=64, memory_threshold=0.9):
           self.model = model
           self.current_batch_size = initial_batch_size
           self.memory_threshold = memory_threshold
           
           if device.type == 'cuda':
               self.total_memory = torch.cuda.get_device_properties(0).total_memory
           else:
               self.total_memory = None
       
       def find_optimal_batch_size(self, sample_data):
           \"\"\"Find the largest batch size that fits in memory.\"\"\""
           
           if self.total_memory is None:
               return self.current_batch_size
           
           test_batch_sizes = [32, 64, 128, 256, 512, 1024]
           optimal_batch_size = 32
           
           for batch_size in test_batch_sizes:
               try:
                   if device.type == 'cuda':
                       torch.cuda.empty_cache()
                       torch.cuda.reset_peak_memory_stats()
                   
                   # Create test batch
                   batch_data = sample_data[:batch_size].to(device)
                   
                   # Forward pass
                   with torch.no_grad():
                       _ = self.model(batch_data)
                   
                   if device.type == 'cuda':
                       peak_memory = torch.cuda.max_memory_allocated()
                       memory_ratio = peak_memory / self.total_memory
                       
                       if memory_ratio < self.memory_threshold:
                           optimal_batch_size = batch_size
                       else:
                           break
                   
                   del batch_data
                   
               except torch.cuda.OutOfMemoryError:
                   break
           
           self.current_batch_size = optimal_batch_size
           print(f"Optimal batch size: {optimal_batch_size}")
           return optimal_batch_size

Algorithmic Optimizations
-------------------------

**Efficient Parameter Updates**

.. code-block:: python

   class OptimizedSKLinear(pr.nn.SKLinear):
       \"\"\"SKLinear with optimized parameter updates.\"\"\""
       
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           
           # Pre-compute frequently used values
           self.register_buffer('scale_factor', torch.tensor(1.0 / (2 * self.num_terms)))
           
           # Cache for intermediate computations
           self._cached_u1_input = None
           self._cached_u2_input = None
           
       def forward(self, x):
           batch_size = x.size(0)
           
           # Optimized computation with caching
           if self.training:
               # Cache intermediate results for backward pass
               u1_input = x @ self.U1s.view(-1, x.size(1)).t()
               u2_input = x @ self.U2s.view(-1, x.size(1)).t()
               
               self._cached_u1_input = u1_input
               self._cached_u2_input = u2_input
           else:
               # Standard computation for inference
               u1_input = x @ self.U1s.view(-1, x.size(1)).t()
               u2_input = x @ self.U2s.view(-1, x.size(1)).t()
           
           # Reshape for batch processing
           u1_reshaped = u1_input.view(batch_size, self.num_terms, self.low_rank)
           u2_reshaped = u2_input.view(batch_size, self.num_terms, self.low_rank)
           
           # Efficient batch matrix multiply
           s1_output = torch.bmm(u1_reshaped, self.S1s.transpose(1, 2))
           s2_output = torch.bmm(u2_reshaped, self.S2s.transpose(1, 2))
           
           # Combine terms
           combined = (s1_output + s2_output) * self.scale_factor
           output = combined.sum(dim=1)
           
           if self.bias is not None:
               output = output + self.bias
           
           return output

**Kernel Fusion for CUDA**

.. code-block:: python

   # Note: This is conceptual - actual kernel fusion requires C++/CUDA implementation
   
   class FusedSketchedOps:
       \"\"\"Conceptual fused operations for sketched layers.\"\"\""
       
       @staticmethod
       def fused_sketch_linear_relu(x, s1s, s2s, u1s, u2s, bias=None):
           \"\"\"Fused sketched linear + ReLU operation.\"\"\""
           
           # This would be implemented as a custom CUDA kernel
           # that combines the sketched linear computation with ReLU
           # to reduce memory bandwidth and improve performance
           
           # Placeholder implementation using standard operations
           output = OptimizedSKLinear.forward_impl(x, s1s, s2s, u1s, u2s, bias)
           return torch.relu(output)
       
       @staticmethod
       def fused_sketch_linear_dropout(x, s1s, s2s, u1s, u2s, bias=None, dropout_p=0.1, training=True):
           \"\"\"Fused sketched linear + dropout operation.\"\"\""
           
           output = OptimizedSKLinear.forward_impl(x, s1s, s2s, u1s, u2s, bias)
           
           if training and dropout_p > 0:
               return torch.dropout(output, dropout_p, training)
           return output

Profiling and Benchmarking
---------------------------

**Comprehensive Performance Profiling**

.. code-block:: python

   import torch.profiler
   import matplotlib.pyplot as plt
   
   def profile_model_performance(model, input_tensor, num_steps=100):
       \"\"\"Profile model performance using PyTorch profiler.\"\"\""
       
       model = model.to(device)
       input_tensor = input_tensor.to(device)
       
       # Warmup
       for _ in range(10):
           with torch.no_grad():
               _ = model(input_tensor)
       
       if device.type == 'cuda':
           torch.cuda.synchronize()
       
       # Profiling with PyTorch profiler
       with torch.profiler.profile(
           activities=[
               torch.profiler.ProfilerActivity.CPU,
               torch.profiler.ProfilerActivity.CUDA,
           ],
           schedule=torch.profiler.schedule(
               wait=10, warmup=10, active=20, repeat=1
           ),
           on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
           record_shapes=True,
           profile_memory=True,
           with_stack=True
       ) as prof:
           
           for step in range(num_steps):
               with torch.no_grad():
                   _ = model(input_tensor)
               prof.step()
       
       # Print summary
       print("\\nProfiler Summary:")
       print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
       
       return prof
   
   # Profile both models
   x = torch.randn(128, 2048, device=device)
   
   print("Profiling Standard Model:")
   standard_prof = profile_model_performance(standard_model, x)
   
   print("\\nProfiling Sketched Model:")
   sketched_prof = profile_model_performance(sketched_model, x)

**Automated Benchmarking Suite**

.. code-block:: python

   class BenchmarkSuite:
       \"\"\"Comprehensive benchmarking for sketched models.\"\"\""
       
       def __init__(self, device=None):
           self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           self.results = {}
       
       def benchmark_layer_configs(self, base_config, param_ranges):
           \"\"\"Benchmark different layer configurations.\"\"\""
           
           results = []
           
           for num_terms in param_ranges['num_terms']:
               for low_rank in param_ranges['low_rank']:
                   config = base_config.copy()
                   config.update({'num_terms': num_terms, 'low_rank': low_rank})
                   
                   # Create layer
                   layer = pr.nn.SKLinear(**config).to(self.device)
                   
                   # Benchmark
                   input_tensor = torch.randn(128, config['in_features'], device=self.device)
                   
                   # Parameter count
                   param_count = sum(p.numel() for p in layer.parameters())
                   
                   # Timing
                   times = []
                   for _ in range(50):
                       start = time.time()
                       with torch.no_grad():
                           _ = layer(input_tensor)
                       if self.device.type == 'cuda':
                           torch.cuda.synchronize()
                       times.append(time.time() - start)
                   
                   avg_time = sum(times) / len(times)
                   
                   # Memory usage
                   if self.device.type == 'cuda':
                       torch.cuda.reset_peak_memory_stats()
                       with torch.no_grad():
                           _ = layer(input_tensor)
                       peak_memory = torch.cuda.max_memory_allocated() / 1024**2
                   else:
                       peak_memory = 0
                   
                   results.append({
                       'num_terms': num_terms,
                       'low_rank': low_rank,
                       'param_count': param_count,
                       'avg_time_ms': avg_time * 1000,
                       'peak_memory_mb': peak_memory
                   })
           
           return results
       
       def plot_performance_landscape(self, results):
           \"\"\"Plot performance vs parameter tradeoffs.\"\"\""
           
           import pandas as pd
           
           df = pd.DataFrame(results)
           
           fig, axes = plt.subplots(2, 2, figsize=(12, 10))
           
           # Time vs Parameters
           axes[0, 0].scatter(df['param_count'], df['avg_time_ms'])
           axes[0, 0].set_xlabel('Parameter Count')
           axes[0, 0].set_ylabel('Time (ms)')
           axes[0, 0].set_title('Time vs Parameters')
           
           # Memory vs Parameters
           axes[0, 1].scatter(df['param_count'], df['peak_memory_mb'])
           axes[0, 1].set_xlabel('Parameter Count')
           axes[0, 1].set_ylabel('Peak Memory (MB)')
           axes[0, 1].set_title('Memory vs Parameters')
           
           # Time vs num_terms
           for lr in df['low_rank'].unique():
               subset = df[df['low_rank'] == lr]
               axes[1, 0].plot(subset['num_terms'], subset['avg_time_ms'], 
                              marker='o', label=f'low_rank={lr}')
           axes[1, 0].set_xlabel('num_terms')
           axes[1, 0].set_ylabel('Time (ms)')
           axes[1, 0].set_title('Time vs num_terms')
           axes[1, 0].legend()
           
           # Time vs low_rank
           for nt in df['num_terms'].unique():
               subset = df[df['num_terms'] == nt]
               axes[1, 1].plot(subset['low_rank'], subset['avg_time_ms'], 
                              marker='o', label=f'num_terms={nt}')
           axes[1, 1].set_xlabel('low_rank')
           axes[1, 1].set_ylabel('Time (ms)')
           axes[1, 1].set_title('Time vs low_rank')
           axes[1, 1].legend()
           
           plt.tight_layout()
           plt.show()
   
   # Example usage
   benchmark = BenchmarkSuite()
   
   base_config = {
       'in_features': 1024,
       'out_features': 512
   }
   
   param_ranges = {
       'num_terms': [2, 4, 6, 8, 12, 16],
       'low_rank': [16, 32, 48, 64, 96, 128]
   }
   
   results = benchmark.benchmark_layer_configs(base_config, param_ranges)
   benchmark.plot_performance_landscape(results)

Production Optimization Strategies
-----------------------------------

**Model Compilation and Optimization**

.. code-block:: python

   def optimize_model_for_production(model, sample_input, optimization_level='standard'):
       \"\"\"Optimize model for production deployment.\"\"\""
       
       model = model.to(device)
       model.eval()
       
       # TorchScript compilation
       print("Compiling model with TorchScript...")
       try:
           scripted_model = torch.jit.script(model)
           
           # Optimization passes
           if optimization_level == 'aggressive':
               scripted_model = torch.jit.optimize_for_inference(scripted_model)
           
           print("TorchScript compilation successful")
           
       except Exception as e:
           print(f"TorchScript compilation failed: {e}")
           scripted_model = model
       
       # Quantization (if supported)
       if optimization_level in ['standard', 'aggressive']:
           try:
               print("Applying dynamic quantization...")
               quantized_model = torch.quantization.quantize_dynamic(
                   scripted_model, {nn.Linear}, dtype=torch.qint8
               )
               print("Quantization successful")
               return quantized_model
               
           except Exception as e:
               print(f"Quantization failed: {e}")
       
       return scripted_model
   
   # Model serving with optimization
   class OptimizedModelServer:
       \"\"\"Production model server with optimizations.\"\"\""
       
       def __init__(self, model, batch_size=32, max_latency_ms=100):
           self.model = model
           self.batch_size = batch_size
           self.max_latency_ms = max_latency_ms
           
           # Request batching
           self.request_queue = []
           self.batch_timer = None
           
       def predict_batch(self, inputs):
           \"\"\"Process a batch of inputs efficiently.\"\"\""
           
           start_time = time.time()
           
           # Move to device
           if isinstance(inputs, list):
               inputs = torch.stack(inputs).to(self.device)
           else:
               inputs = inputs.to(self.device)
           
           # Inference
           with torch.no_grad():
               outputs = self.model(inputs)
           
           # Move back to CPU if needed
           if self.device.type == 'cuda':
               outputs = outputs.cpu()
           
           latency = (time.time() - start_time) * 1000
           
           return outputs, latency

**Memory-Mapped Model Loading**

.. code-block:: python

   class MemoryMappedModel:
       \"\"\"Model with memory-mapped parameter loading for large models.\"\"\""
       
       def __init__(self, model_path, device=None):
           self.device = device or torch.device('cpu')
           self.model_path = model_path
           
       def load_model_lazy(self):
           \"\"\"Load model with lazy parameter loading.\"\"\""
           
           # Load model structure without parameters
           checkpoint = torch.load(self.model_path, map_location='cpu')
           
           model = create_model_from_config(checkpoint['config'])
           
           # Memory-map large parameter tensors
           for name, param in model.named_parameters():
               if param.numel() > 1000000:  # Large parameters
                   # In practice, this would use memory mapping
                   # Here we simulate lazy loading
                   param.data = checkpoint['state_dict'][name]
               else:
                   param.data = checkpoint['state_dict'][name].to(self.device)
           
           return model.to(self.device)

Best Practices Summary
----------------------

**1. Memory Optimization Checklist**

.. code-block:: python

   def memory_optimization_checklist():
       \"\"\"Best practices for memory optimization.\"\"\""
       
       optimizations = [
           "✓ Use gradient accumulation for large effective batch sizes",
           "✓ Enable mixed precision training (FP16)",
           "✓ Clear cache periodically with torch.cuda.empty_cache()",
           "✓ Use gradient checkpointing for very deep models",
           "✓ Optimize dimensions for Tensor Core usage (multiples of 8)",
           "✓ Monitor memory usage during training",
           "✓ Use dynamic batch sizing based on available memory",
           "✓ Implement memory-efficient data loading",
       ]
       
       for item in optimizations:
           print(item)

**2. Performance Optimization Checklist**

.. code-block:: python

   def performance_optimization_checklist():
       \"\"\"Best practices for performance optimization.\"\"\""
       
       optimizations = [
           "✓ Profile code to identify bottlenecks",
           "✓ Use TorchScript compilation for production",
           "✓ Enable CUDA graphs for repeated computations",
           "✓ Optimize layer dimensions for hardware",
           "✓ Use efficient data types (FP16, BF16)",
           "✓ Implement kernel fusion where possible",
           "✓ Minimize data transfers between CPU/GPU",
           "✓ Use asynchronous operations when possible",
           "✓ Optimize learning rate schedules",
           "✓ Use efficient optimizers (AdamW, Lion)",
       ]
       
       for item in optimizations:
           print(item)

This tutorial provides comprehensive guidance on optimizing Panther models for maximum performance. The next tutorial will cover custom sketching implementations.
