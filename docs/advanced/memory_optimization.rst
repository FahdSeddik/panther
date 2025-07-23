Memory Optimization
==================

This guide covers advanced memory optimization techniques when using Panther for large-scale applications.

Memory Profiling and Analysis
-----------------------------

**Understanding Memory Usage Patterns**

.. code-block:: python

   import torch
   import psutil
   import panther as pr
   from torch.profiler import profile, ProfilerActivity
   
   class MemoryProfiler:
       """Comprehensive memory profiling for Panther operations."""
       
       def __init__(self):
           self.baseline_memory = self._get_memory_usage()
           
       def _get_memory_usage(self):
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
       
       def profile_operation(self, operation, operation_name="Operation"):
           """Profile memory usage of a specific operation."""
           
           # Clear GPU cache
           if torch.cuda.is_available():
               torch.cuda.empty_cache()
           
           # Measure before
           memory_before = self._get_memory_usage()
           
           # Run operation
           result = operation()
           
           # Measure after
           memory_after = self._get_memory_usage()
           
           # Calculate differences
           memory_diff = {
               key: memory_after[key] - memory_before[key]
               for key in memory_before.keys()
           }
           
           print(f"\\n{operation_name} Memory Usage:")
           print(f"  CPU: {memory_diff['cpu']:+.2f} MB")
           print(f"  GPU Allocated: {memory_diff['gpu_allocated']:+.2f} MB")
           print(f"  GPU Cached: {memory_diff['gpu_cached']:+.2f} MB")
           
           return result, memory_diff
   
   # Example usage
   profiler = MemoryProfiler()
   
   def test_sklinear_memory():
       # Large SKLinear layer
       layer = pr.nn.SKLinear(4096, 2048, sketch_size=1024).cuda()
       x = torch.randn(512, 4096, device='cuda')
       return layer(x)
   
   output, memory_usage = profiler.profile_operation(
       test_sklinear_memory, "SKLinear Forward Pass"
   )

**Detailed Memory Breakdown**

.. code-block:: python

   def analyze_layer_memory_components():
       """Analyze memory usage of different layer components."""
       
       layer_configs = [
           {'in_features': 1024, 'out_features': 512, 'sketch_size': 256},
           {'in_features': 2048, 'out_features': 1024, 'sketch_size': 512},
           {'in_features': 4096, 'out_features': 2048, 'sketch_size': 1024},
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
               config['sketch_size']
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
           
           # Sketched intermediate (only for SKLinear)
           sketch_mem = batch_size * config['sketch_size'] * 4 / 1024**2
           
           print(f"  Activation Memory (batch_size={batch_size}):")
           print(f"    Input: {input_mem:.2f} MB")
           print(f"    Output: {output_mem:.2f} MB")
           print(f"    Sketch intermediate: {sketch_mem:.2f} MB")

Memory-Efficient Training Strategies
------------------------------------

**Gradient Checkpointing with Sketched Layers**

.. code-block:: python

   import torch.utils.checkpoint as checkpoint
   
   class MemoryEfficientSketchedModel(torch.nn.Module):
       """Model with gradient checkpointing for memory efficiency."""
       
       def __init__(self, layer_sizes, sketch_ratios, checkpoint_segments=4):
           super().__init__()
           
           self.checkpoint_segments = checkpoint_segments
           
           # Create sketched layers
           layers = []
           for i in range(len(layer_sizes) - 1):
               in_size = layer_sizes[i]
               out_size = layer_sizes[i + 1]
               sketch_size = int(in_size * sketch_ratios[i])
               
               layers.append(pr.nn.SKLinear(in_size, out_size, sketch_size))
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
   sketch_ratios = [0.5, 0.5, 0.5, 0.5]
   
   model = MemoryEfficientSketchedModel(
       layer_sizes, sketch_ratios, checkpoint_segments=2
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
       model = pr.nn.Sequential(
           pr.nn.SKLinear(2048, 1024, sketch_size=512),
           torch.nn.ReLU(),
           pr.nn.SKLinear(1024, 512, sketch_size=256),
           torch.nn.ReLU(),
           pr.nn.SKLinear(512, 10, sketch_size=128)
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

**Dynamic Batch Size Adjustment**

.. code-block:: python

   class AdaptiveBatchTrainer:
       """Trainer that adapts batch size based on memory usage."""
       
       def __init__(self, model, optimizer, max_memory_mb=8000):
           self.model = model
           self.optimizer = optimizer
           self.max_memory_mb = max_memory_mb
           self.current_batch_size = 32
           self.min_batch_size = 8
           self.max_batch_size = 512
           
       def _get_gpu_memory_mb(self):
           """Get current GPU memory usage in MB."""
           if torch.cuda.is_available():
               return torch.cuda.memory_allocated() / 1024**2
           return 0
       
       def _find_optimal_batch_size(self, data_loader):
           """Find optimal batch size through binary search."""
           
           low, high = self.min_batch_size, self.max_batch_size
           optimal_batch_size = low
           
           while low <= high:
               mid = (low + high) // 2
               
               try:
                   # Test with this batch size
                   success = self._test_batch_size(data_loader, mid)
                   
                   if success:
                       optimal_batch_size = mid
                       low = mid + 1
                   else:
                       high = mid - 1
                       
               except RuntimeError as e:
                   if "out of memory" in str(e):
                       high = mid - 1
                   else:
                       raise e
               
               # Clear GPU cache
               torch.cuda.empty_cache()
           
           return optimal_batch_size
       
       def _test_batch_size(self, data_loader, batch_size):
           """Test if a batch size works within memory constraints."""
           
           # Create test batch
           sample_batch = next(iter(data_loader))
           if len(sample_batch[0]) < batch_size:
               return False
           
           x = sample_batch[0][:batch_size].cuda()
           y = sample_batch[1][:batch_size].cuda()
           
           # Test forward and backward pass
           self.optimizer.zero_grad()
           output = self.model(x)
           loss = torch.nn.functional.cross_entropy(output, y)
           loss.backward()
           
           # Check memory usage
           memory_used = self._get_gpu_memory_mb()
           
           return memory_used < self.max_memory_mb
       
       def train_epoch(self, data_loader):
           """Train for one epoch with adaptive batch size."""
           
           # Find optimal batch size
           optimal_batch_size = self._find_optimal_batch_size(data_loader)
           print(f"Using batch size: {optimal_batch_size}")
           
           # Create new data loader with optimal batch size
           from torch.utils.data import DataLoader
           adaptive_loader = DataLoader(
               data_loader.dataset,
               batch_size=optimal_batch_size,
               shuffle=True
           )
           
           total_loss = 0
           for batch_idx, (x, y) in enumerate(adaptive_loader):
               x, y = x.cuda(), y.cuda()
               
               self.optimizer.zero_grad()
               output = self.model(x)
               loss = torch.nn.functional.cross_entropy(output, y)
               loss.backward()
               self.optimizer.step()
               
               total_loss += loss.item()
               
               # Monitor memory usage
               if batch_idx % 10 == 0:
                   memory_mb = self._get_gpu_memory_mb()
                   print(f"Batch {batch_idx}, Loss: {loss.item():.4f}, "
                         f"Memory: {memory_mb:.1f} MB")
           
           return total_loss / len(adaptive_loader)

Memory Optimization for Large Models
------------------------------------

**Model Parallelism with Sketched Layers**

.. code-block:: python

   class DistributedSketchedModel(torch.nn.Module):
       """Large model split across multiple GPUs."""
       
       def __init__(self, layer_configs, devices):
           super().__init__()
           
           self.devices = devices
           self.layers = torch.nn.ModuleList()
           
           # Distribute layers across devices
           layers_per_device = len(layer_configs) // len(devices)
           
           for i, config in enumerate(layer_configs):
               device_idx = min(i // layers_per_device, len(devices) - 1)
               device = devices[device_idx]
               
               layer = pr.nn.SKLinear(
                   config['in_features'],
                   config['out_features'], 
                   config['sketch_size']
               ).to(device)
               
               self.layers.append(layer)
           
       def forward(self, x):
           """Forward pass across multiple devices."""
           
           current_device = x.device
           
           for i, layer in enumerate(self.layers):
               # Move input to layer's device if necessary
               if x.device != layer.weight.device:
                   x = x.to(layer.weight.device)
               
               x = layer(x)
               
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
           {'in_features': 4096, 'out_features': 2048, 'sketch_size': 1024},
           {'in_features': 2048, 'out_features': 1024, 'sketch_size': 512},
           {'in_features': 1024, 'out_features': 512, 'sketch_size': 256},
           {'in_features': 512, 'out_features': 256, 'sketch_size': 128},
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
   model = pr.nn.SKLinear(1024, 10, sketch_size=512).cuda()
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
