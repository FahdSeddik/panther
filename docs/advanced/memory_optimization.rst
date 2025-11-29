Memory Optimization
===================

This guide covers advanced memory optimization techniques when using Panther for large-scale applications.

.. note::
   Many techniques in this guide (like gradient checkpointing and mixed precision training) 
   are standard PyTorch practices that work with any model. The focus here is on how to apply 
   them effectively with Panther's sketched layers.

Memory Profiling and Analysis
-----------------------------

**Understanding Memory Usage Patterns**

.. code-block:: python

   import torch
   import psutil
   import panther as pr
   
   def get_memory_usage():
       """Get current memory usage."""
       if torch.cuda.is_available():
           gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
           gpu_cached = torch.cuda.memory_reserved() / 1024**2   # MB
       else:
           gpu_memory, gpu_cached = 0, 0
           
       cpu_memory = psutil.Process().memory_info().rss / 1024**2  # MB
       
       return {
           'cpu': cpu_memory,
           'gpu_allocated': gpu_memory,
           'gpu_cached': gpu_cached
       }
   
   def profile_layer_memory():
       """Profile memory usage of a specific layer."""
       
       # Clear GPU cache
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
       
       # Measure before
       memory_before = get_memory_usage()
       
       # Create large SKLinear layer
       layer = pr.nn.SKLinear(4096, 2048, num_terms=8, low_rank=128).cuda()
       x = torch.randn(512, 4096, device='cuda')
       output = layer(x)
       
       # Measure after
       memory_after = get_memory_usage()
       
       # Calculate differences
       print(f"\\nMemory Usage:")
       print(f"  CPU: {memory_after['cpu'] - memory_before['cpu']:+.2f} MB")
       print(f"  GPU Allocated: {memory_after['gpu_allocated'] - memory_before['gpu_allocated']:+.2f} MB")
       print(f"  GPU Cached: {memory_after['gpu_cached'] - memory_before['gpu_cached']:+.2f} MB")
   
   profile_layer_memory()

**Detailed Memory Breakdown**

.. code-block:: python

   def analyze_layer_memory_components():
       """Analyze memory usage of different layer components."""
       
       layer_configs = [
           {'in_features': 1024, 'out_features': 512, 'num_terms': 4, 'low_rank': 64},
           {'in_features': 2048, 'out_features': 1024, 'num_terms': 4, 'low_rank': 128},
           {'in_features': 4096, 'out_features': 2048, 'num_terms': 8, 'low_rank': 128},
       ]
       
       for config in layer_configs:
           print(f"\\nAnalyzing layer: {config}")
           
           # Standard linear layer
           std_layer = torch.nn.Linear(
               config['in_features'], config['out_features']
           ).cuda()
           
           # SKLinear layer
           sk_layer = pr.nn.SKLinear(
               config['in_features'], 
               config['out_features'],
               config['num_terms'],
               config['low_rank']
           ).cuda()
           
           # Calculate memory for parameters
           std_params = sum(p.numel() for p in std_layer.parameters())
           sk_params = sum(p.numel() for p in sk_layer.parameters())
           
           # Memory in MB (assuming float32)
           std_memory = std_params * 4 / 1024**2
           sk_memory = sk_params * 4 / 1024**2
           
           print(f"  Standard Linear:")
           print(f"    Parameters: {std_params:,}")
           print(f"    Memory: {std_memory:.2f} MB")
           
           print(f"  SKLinear:")
           print(f"    Parameters: {sk_params:,}")
           print(f"    Memory: {sk_memory:.2f} MB")
           print(f"    Reduction: {(1 - sk_memory/std_memory)*100:.1f}%")
           
           # Activation memory for forward pass
           batch_size = 128
           
           # Input activation
           input_mem = batch_size * config['in_features'] * 4 / 1024**2
           
           # Output activation
           output_mem = batch_size * config['out_features'] * 4 / 1024**2
           
           print(f"  Activation Memory (batch_size={batch_size}):")
           print(f"    Input: {input_mem:.2f} MB")
           print(f"    Output: {output_mem:.2f} MB")
   
   analyze_layer_memory_components()

Memory-Efficient Training Strategies
------------------------------------

**Gradient Checkpointing with Sketched Layers**

.. code-block:: python

   import torch.utils.checkpoint as checkpoint
   
   class MemoryEfficientSketchedModel(torch.nn.Module):
       """Model with gradient checkpointing for memory efficiency."""
       
       def __init__(self, layer_sizes, num_terms=4, low_rank=32, checkpoint_segments=4):
           super().__init__()
           
           self.checkpoint_segments = checkpoint_segments
           
           # Create sketched layers
           layers = []
           for i in range(len(layer_sizes) - 1):
               in_size = layer_sizes[i]
               out_size = layer_sizes[i + 1]
               
               layers.append(pr.nn.SKLinear(
                   in_size, out_size,
                   num_terms=num_terms,
                   low_rank=low_rank
               ))
               if i < len(layer_sizes) - 2:  # No activation after last layer
                   layers.append(torch.nn.ReLU())
           
           self.layers = torch.nn.ModuleList(layers)
           
           # Group layers into checkpointing segments
           layers_per_segment = len(self.layers) // self.checkpoint_segments
           self.segments = []
           
           for i in range(self.checkpoint_segments):
               start_idx = i * layers_per_segment
               end_idx = start_idx + layers_per_segment
               if i == self.checkpoint_segments - 1:  # Last segment gets remaining layers
                   end_idx = len(self.layers)
               
               segment = torch.nn.Sequential(*self.layers[start_idx:end_idx])
               self.segments.append(segment)
       
       def forward(self, x):
           """Forward pass with gradient checkpointing."""
           
           for segment in self.segments:
               # Use gradient checkpointing for each segment
               if self.training:
                   x = checkpoint.checkpoint(segment, x, use_reentrant=False)
               else:
                   x = segment(x)
           
           return x
   
   # Example usage
   layer_sizes = [1024, 512, 256, 128, 64]
   
   model = MemoryEfficientSketchedModel(
       layer_sizes,
       num_terms=4,
       low_rank=32,
       checkpoint_segments=2
   ).cuda()
   
   # Training with reduced memory footprint
   x = torch.randn(64, 1024, device='cuda', requires_grad=True)
   y = torch.randn(64, 64, device='cuda')
   
   optimizer = torch.optim.Adam(model.parameters())
   criterion = torch.nn.MSELoss()
   
   # Forward and backward pass
   optimizer.zero_grad()
   output = model(x)
   loss = criterion(output, y)
   loss.backward()
   optimizer.step()

**Mixed Precision Training**

.. code-block:: python

   from torch.cuda.amp import autocast, GradScaler
   
   def train_with_mixed_precision():
       """Training loop with automatic mixed precision."""
       
       # Model setup
       model = torch.nn.Sequential(
           pr.nn.SKLinear(2048, 1024, num_terms=4, low_rank=128),
           torch.nn.ReLU(),
           pr.nn.SKLinear(1024, 512, num_terms=4, low_rank=64),
           torch.nn.ReLU(),
           pr.nn.SKLinear(512, 10, num_terms=2, low_rank=32)
       ).cuda()
       
       optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
       scaler = GradScaler()
       
       # Training data (synthetic)
       def get_batch():
           x = torch.randn(128, 2048, device='cuda')
           y = torch.randint(0, 10, (128,), device='cuda')
           return x, y
       
       model.train()
       for epoch in range(5):
           for batch_idx in range(100):
               x, y = get_batch()
               
               optimizer.zero_grad()
               
               # Forward pass with autocast
               with autocast():
                   output = model(x)
                   loss = torch.nn.functional.cross_entropy(output, y)
               
               # Backward pass with gradient scaling
               scaler.scale(loss).backward()
               scaler.step(optimizer)
               scaler.update()
               
               if batch_idx % 20 == 0:
                   print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
   
   train_with_mixed_precision()

Memory Monitoring Best Practices
---------------------------------

**Monitoring GPU Memory**

.. code-block:: python

   def monitor_memory_usage(model, input_shape, batch_size=32):
       """Monitor memory usage during training."""
       
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       model = model.to(device)
       
       # Clear cache
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
           torch.cuda.reset_peak_memory_stats()
       
       # Create sample input
       x = torch.randn(batch_size, *input_shape, device=device)
       y = torch.randint(0, 10, (batch_size,), device=device)
       
       # Forward pass
       optimizer = torch.optim.Adam(model.parameters())
       criterion = torch.nn.CrossEntropyLoss()
       
       optimizer.zero_grad()
       output = model(x)
       loss = criterion(output, y)
       loss.backward()
       optimizer.step()
       
       if torch.cuda.is_available():
           allocated = torch.cuda.memory_allocated() / 1024**3  # GB
           reserved = torch.cuda.memory_reserved() / 1024**3  # GB
           peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
           
           print(f"\\nMemory Usage:")
           print(f"  Allocated: {allocated:.2f} GB")
           print(f"  Reserved: {reserved:.2f} GB")
           print(f"  Peak: {peak:.2f} GB")

**Comparing Memory Usage**

.. code-block:: python

   def compare_layer_memory():
       """Compare memory usage between standard and sketched layers."""
       
       configs = [
           (1024, 512, 4, 64),
           (2048, 1024, 8, 128),
           (4096, 2048, 16, 256)
       ]
       
       print("Memory Comparison:")
       print("-" * 60)
       
       for in_feat, out_feat, num_terms, low_rank in configs:
           # Standard layer
           std = torch.nn.Linear(in_feat, out_feat)
           std_params = sum(p.numel() for p in std.parameters())
           std_mb = std_params * 4 / 1024**2
           
           # Sketched layer
           sk = pr.nn.SKLinear(in_feat, out_feat, num_terms, low_rank)
           sk_params = sum(p.numel() for p in sk.parameters())
           sk_mb = sk_params * 4 / 1024**2
           
           reduction = (1 - sk_mb / std_mb) * 100
           
           print(f"\\n{in_feat} → {out_feat}:")
           print(f"  Standard: {std_params:,} params ({std_mb:.2f} MB)")
           print(f"  Sketched: {sk_params:,} params ({sk_mb:.2f} MB)")
           print(f"  Reduction: {reduction:.1f}%")
   
   compare_layer_memory()

Tips for Memory-Efficient Training
-----------------------------------

1. **Use gradient accumulation for large models**

.. code-block:: python

   accumulation_steps = 4
   
   for i, (x, y) in enumerate(dataloader):
       output = model(x)
       loss = criterion(output, y) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()

2. **Clear cache periodically**

.. code-block:: python

   if torch.cuda.is_available():
       torch.cuda.empty_cache()

3. **Use smaller data types when possible**

.. code-block:: python

   model = pr.nn.SKLinear(1024, 512, num_terms=4, low_rank=64, 
                         dtype=torch.float16)
               
               # Add activation function between layers
               if i < len(self.layers) - 1:
                   x = torch.relu(x)
           
           # Move final output back to original device
           if x.device != current_device:
               x = x.to(current_device)
           
           return x
   
   # Example with 4 GPUs
   if torch.cuda.device_count() >= 4:
       devices = [f'cuda:{i}' for i in range(4)]
       
       layer_configs = [
           {'in_features': 4096, 'out_features': 2048, 'num_terms': 8, 'low_rank': 128},
           {'in_features': 2048, 'out_features': 1024, 'num_terms': 4, 'low_rank': 128},
           {'in_features': 1024, 'out_features': 512, 'num_terms': 4, 'low_rank': 64},
           {'in_features': 512, 'out_features': 256, 'num_terms': 2, 'low_rank': 64},
       ]
       
       model = DistributedSketchedModel(layer_configs, devices)
       
       # Test forward pass
       x = torch.randn(32, 4096, device='cuda:0')
       output = model(x)
       print(f"Output shape: {output.shape}, Device: {output.device}")

**Memory-Efficient Inference**

.. code-block:: python

   class StreamingInference:
       """Memory-efficient inference for large inputs."""
       
       def __init__(self, model, chunk_size=1000):
           self.model = model
           self.chunk_size = chunk_size
           
       def predict_large_batch(self, X):
           """Process large batch in chunks to save memory."""
           
           n_samples = X.shape[0]
           results = []
           
           self.model.eval()
           with torch.no_grad():
               for i in range(0, n_samples, self.chunk_size):
                   end_idx = min(i + self.chunk_size, n_samples)
                   chunk = X[i:end_idx]
                   
                   # Move chunk to GPU, process, move back to CPU
                   if torch.cuda.is_available():
                       chunk = chunk.cuda()
                       chunk_result = self.model(chunk).cpu()
                   else:
                       chunk_result = self.model(chunk)
                   
                   results.append(chunk_result)
                   
                   # Clear GPU cache after each chunk
                   if torch.cuda.is_available():
                       torch.cuda.empty_cache()
           
           return torch.cat(results, dim=0)
   
   # Example usage
   model = pr.nn.SKLinear(1024, 10, num_terms=4, low_rank=64).cuda()
   inference_engine = StreamingInference(model, chunk_size=500)
   
   # Large input that won't fit in memory all at once
   large_input = torch.randn(10000, 1024)
   predictions = inference_engine.predict_large_batch(large_input)

Memory Monitoring and Debugging
-------------------------------

**Continuous Memory Monitoring**

.. code-block:: python

   import threading
   import time
   import matplotlib.pyplot as plt
   
   class MemoryMonitor:
       """Continuous memory monitoring during training."""
       
       def __init__(self, interval=1.0):
           self.interval = interval
           self.monitoring = False
           self.memory_log = []
           self.time_log = []
           self.start_time = None
           
       def start_monitoring(self):
           """Start memory monitoring in a separate thread."""
           self.monitoring = True
           self.start_time = time.time()
           self.monitor_thread = threading.Thread(target=self._monitor_loop)
           self.monitor_thread.start()
           
       def stop_monitoring(self):
           """Stop memory monitoring."""
           self.monitoring = False
           self.monitor_thread.join()
           
       def _monitor_loop(self):
           """Continuous monitoring loop."""
           while self.monitoring:
               current_time = time.time() - self.start_time
               
               if torch.cuda.is_available():
                   memory_mb = torch.cuda.memory_allocated() / 1024**2
               else:
                   memory_mb = psutil.Process().memory_info().rss / 1024**2
               
               self.time_log.append(current_time)
               self.memory_log.append(memory_mb)
               
               time.sleep(self.interval)
           
       def plot_memory_usage(self):
           """Plot memory usage over time."""
           plt.figure(figsize=(12, 6))
           plt.plot(self.time_log, self.memory_log, 'b-', linewidth=2)
           plt.xlabel('Time (seconds)')
           plt.ylabel('Memory Usage (MB)')
           plt.title('Memory Usage Over Time')
           plt.grid(True, alpha=0.3)
           plt.tight_layout()
           plt.show()
           
           # Print statistics
           max_memory = max(self.memory_log)
           avg_memory = sum(self.memory_log) / len(self.memory_log)
           print(f"\\nMemory Statistics:")
           print(f"  Maximum: {max_memory:.2f} MB")
           print(f"  Average: {avg_memory:.2f} MB")
           print(f"  Final: {self.memory_log[-1]:.2f} MB")

Best Practices Summary
----------------------

**Memory Optimization Guidelines**

1. **Use Sketching Wisely**
   - Start with 50% sketch ratio, adjust based on accuracy requirements
   - Monitor memory vs. accuracy trade-offs
   - Use higher sketch ratios for early layers, lower for later layers

2. **Training Optimizations**
   - Enable gradient checkpointing for deep models
   - Use mixed precision training (AMP)
   - Implement adaptive batch sizing
   - Clear GPU cache regularly

3. **Model Architecture**
   - Distribute large models across multiple GPUs
   - Use streaming inference for large inputs
   - Consider model pruning after sketching

4. **Monitoring and Debugging**
   - Continuously monitor memory usage
   - Profile different components separately
   - Use memory-efficient data loading

This comprehensive memory optimization guide helps you maximize efficiency when working with large-scale Panther models.
