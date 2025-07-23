Tensor Core Optimization
========================

This guide covers using NVIDIA Tensor Cores with Panther for maximum performance on modern GPUs.

Overview
--------

NVIDIA Tensor Cores are specialized hardware units designed for mixed-precision matrix operations. Panther automatically leverages Tensor Cores when:

* Using supported data types (float16, bfloat16)
* Matrix dimensions are multiples of specific values
* Operations meet size thresholds

Tensor Core Requirements
------------------------

**Supported Data Types**

.. code-block:: python

   import torch
   import panther as pr
   
   # Tensor Core compatible data types
   supported_dtypes = [
       torch.float16,    # Half precision (most common)
       torch.bfloat16,   # Brain floating point
   ]
   
   # Example with half precision
   device = torch.device('cuda')
   A = torch.randn(1024, 512, dtype=torch.float16, device=device)
   
   # SKLinear automatically uses Tensor Cores when possible
   layer = pr.nn.SKLinear(512, 256, dtype=torch.float16, device=device)
   output = layer(A)

**Matrix Dimension Requirements**

Tensor Cores perform optimally with specific matrix dimensions:

.. code-block:: python

   # Optimal dimensions for different Tensor Core generations
   tensor_core_multiples = {
       'V100': 8,    # Dimensions should be multiples of 8
       'A100': 16,   # Dimensions should be multiples of 16  
       'H100': 32,   # Dimensions should be multiples of 32
   }
   
   def optimize_dimensions_for_tensor_cores(m, n, k, target_gpu='A100'):
       """Adjust dimensions for optimal Tensor Core usage."""
       multiple = tensor_core_multiples[target_gpu]
       
       # Round up to nearest multiple
       m_opt = ((m + multiple - 1) // multiple) * multiple
       n_opt = ((n + multiple - 1) // multiple) * multiple 
       k_opt = ((k + multiple - 1) // multiple) * multiple
       
       return m_opt, n_opt, k_opt
   
   # Example usage
   original_dims = (1000, 500, 250)
   optimized_dims = optimize_dimensions_for_tensor_cores(*original_dims)
   print(f"Original: {original_dims}")
   print(f"Optimized: {optimized_dims}")

Mixed Precision Training
------------------------

**Automatic Mixed Precision (AMP)**

.. code-block:: python

   import torch
   from torch.cuda.amp import autocast, GradScaler
   
   # Enable automatic mixed precision
   scaler = GradScaler()
   
   model = pr.nn.Sequential(
       pr.nn.SKLinear(784, 512, sketch_size=256),
       torch.nn.ReLU(),
       pr.nn.SKLinear(512, 256, sketch_size=128),
       torch.nn.ReLU(),
       pr.nn.SKLinear(256, 10, sketch_size=64)
   ).half().cuda()
   
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   
   # Training loop with AMP
   for batch_idx, (data, target) in enumerate(dataloader):
       data, target = data.half().cuda(), target.cuda()
       
       optimizer.zero_grad()
       
       # Forward pass with autocast
       with autocast():
           output = model(data)
           loss = torch.nn.functional.cross_entropy(output, target)
       
       # Backward pass with gradient scaling
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()

**Manual Mixed Precision**

.. code-block:: python

   class MixedPrecisionSKLinear(torch.nn.Module):
       """SKLinear with manual mixed precision control."""
       
       def __init__(self, in_features, out_features, sketch_size):
           super().__init__()
           self.sk_linear = pr.nn.SKLinear(
               in_features, out_features, sketch_size,
               dtype=torch.float16
           )
           
       def forward(self, x):
           # Convert input to half precision
           x_half = x.half()
           
           # Compute in half precision
           output_half = self.sk_linear(x_half)
           
           # Convert back to float32 for stable gradients
           return output_half.float()

Performance Monitoring
----------------------

**Tensor Core Utilization**

.. code-block:: python

   import torch.profiler
   
   def profile_tensor_core_usage():
       """Profile Tensor Core utilization during training."""
       
       model = pr.nn.SKLinear(1024, 512, sketch_size=256).half().cuda()
       x = torch.randn(128, 1024, dtype=torch.float16, device='cuda')
       
       with torch.profiler.profile(
           activities=[torch.profiler.ProfilerActivity.CUDA],
           record_shapes=True,
           with_stack=True
       ) as prof:
           for _ in range(10):
               output = model(x)
               loss = output.sum()
               loss.backward()
       
       # Analyze Tensor Core usage
       events = prof.events()
       tensor_core_ops = [
           event for event in events 
           if 'gemm' in event.name.lower() and 'half' in event.name.lower()
       ]
       
       print(f"Tensor Core operations: {len(tensor_core_ops)}")
       print(f"Total CUDA time: {prof.key_averages().total_average().cuda_time:.2f}μs")
       
       return prof
   
   # Run profiling
   profiler_result = profile_tensor_core_usage()

**Memory Bandwidth Optimization**

.. code-block:: python

   def benchmark_tensor_core_performance():
       """Benchmark different configurations for Tensor Core performance."""
       
       import time
       
       configurations = [
           {'batch_size': 128, 'in_features': 1024, 'out_features': 512, 'sketch_size': 256},
           {'batch_size': 256, 'in_features': 2048, 'out_features': 1024, 'sketch_size': 512},
           {'batch_size': 512, 'in_features': 4096, 'out_features': 2048, 'sketch_size': 1024},
       ]
       
       results = []
       
       for config in configurations:
           # Create model and data
           model = pr.nn.SKLinear(
               config['in_features'], 
               config['out_features'],
               config['sketch_size']
           ).half().cuda()
           
           x = torch.randn(
               config['batch_size'], 
               config['in_features'],
               dtype=torch.float16, 
               device='cuda'
           )
           
           # Warmup
           for _ in range(10):
               _ = model(x)
           
           torch.cuda.synchronize()
           
           # Benchmark
           start_time = time.time()
           for _ in range(100):
               output = model(x)
           torch.cuda.synchronize()
           end_time = time.time()
           
           throughput = 100 * config['batch_size'] / (end_time - start_time)
           
           results.append({
               'config': config,
               'throughput': throughput,
               'time_per_batch': (end_time - start_time) / 100
           })
           
           print(f"Config: {config}")
           print(f"Throughput: {throughput:.1f} samples/sec")
           print(f"Time per batch: {(end_time - start_time) / 100 * 1000:.2f}ms")
           print("-" * 50)
       
       return results

Optimization Guidelines
-----------------------

**Best Practices**

1. **Dimension Alignment**

.. code-block:: python

   # Good: Dimensions aligned to Tensor Core requirements
   good_layer = pr.nn.SKLinear(1024, 512, sketch_size=256)  # All multiples of 16
   
   # Suboptimal: Misaligned dimensions
   bad_layer = pr.nn.SKLinear(1000, 500, sketch_size=250)   # Not aligned

2. **Batch Size Optimization**

.. code-block:: python

   # Tensor Cores work best with larger batch sizes
   optimal_batch_sizes = [128, 256, 512]  # Better utilization
   suboptimal_batch_sizes = [1, 4, 8]     # Poor utilization

3. **Memory Layout**

.. code-block:: python

   # Ensure contiguous memory layout
   x = torch.randn(128, 1024, dtype=torch.float16, device='cuda')
   x = x.contiguous()  # Important for Tensor Core performance

**Troubleshooting**

Common issues and solutions:

.. code-block:: python

   def diagnose_tensor_core_issues():
       """Diagnose common Tensor Core performance issues."""
       
       # Check 1: Data type support
       def check_dtype(tensor):
           supported = tensor.dtype in [torch.float16, torch.bfloat16]
           print(f"Data type {tensor.dtype}: {'✓' if supported else '✗'}")
           return supported
       
       # Check 2: Dimension alignment
       def check_dimensions(m, n, k):
           aligned = all(dim % 16 == 0 for dim in [m, n, k])
           print(f"Dimensions {m}x{n}x{k}: {'✓' if aligned else '✗'}")
           return aligned
       
       # Check 3: GPU capability
       def check_gpu_capability():
           if torch.cuda.is_available():
               capability = torch.cuda.get_device_capability()
               supports_tensor_cores = capability[0] >= 7  # Volta and newer
               print(f"GPU capability {capability}: {'✓' if supports_tensor_cores else '✗'}")
               return supports_tensor_cores
           return False
       
       # Example checks
       tensor = torch.randn(128, 1024, dtype=torch.float32)
       check_dtype(tensor)
       check_dimensions(1024, 512, 256)
       check_gpu_capability()

Integration with Panther
-------------------------

**SKLinear Tensor Core Integration**

.. code-block:: python

   # SKLinear automatically detects and uses Tensor Cores
   layer = pr.nn.SKLinear(
       in_features=1024,
       out_features=512, 
       sketch_size=256,
       dtype=torch.float16,      # Enable Tensor Cores
       device='cuda'
   )
   
   # Verify Tensor Core usage
   print(f"Tensor Core compatible: {layer.tensor_core_compatible}")

**Sketching with Tensor Cores**

.. code-block:: python

   # Sketching operators also benefit from Tensor Cores
   from panther.sketch import GaussianSketch, SRHTSketch
   
   # Use half precision for sketching
   sketch = GaussianSketch(
       input_dim=2048,
       sketch_dim=512,
       dtype=torch.float16,
       device='cuda'
   )
   
   # Apply sketch with Tensor Core acceleration
   x = torch.randn(256, 2048, dtype=torch.float16, device='cuda')
   sketched = sketch(x)

This guide provides comprehensive coverage of Tensor Core optimization with Panther. For more advanced topics, see the CUDA kernels documentation.
