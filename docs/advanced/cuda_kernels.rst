CUDA Kernels and GPU Optimization
==================================

This guide covers Panther's GPU acceleration and optimization techniques.

.. note::
   Panther's layers automatically leverage GPU acceleration when tensors are on CUDA devices.
   Many of the techniques in this guide (like mixed precision training and memory monitoring) 
   are standard PyTorch practices that apply to any model, not just Panther layers.

GPU Acceleration
----------------

Panther includes optimized CUDA kernels for GPU acceleration that are automatically used when tensors are on CUDA devices.

.. code-block:: python

   import torch
   import panther as pr
   
   # Check for GPU availability
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"Using device: {device}")
   
   # Create layer on GPU - CUDA kernels used automatically
   layer = pr.nn.SKLinear(
       in_features=1024,
       out_features=512,
       num_terms=4,
       low_rank=64,
       device=device
   )
   
   # Forward pass uses GPU acceleration
   x = torch.randn(128, 1024, device=device)
   y = layer(x)

Tensor Core Optimization
-------------------------

Modern NVIDIA GPUs (V100, A100, RTX 30/40 series) include Tensor Cores for accelerated operations.

**Requirements for Tensor Core Usage**

1. **Dimension Constraints**: Dimensions should be multiples of 16
2. **Data Types**: FP16 or mixed precision
3. **GPU Device**: Tensor with CUDA device

.. code-block:: python

   import torch
   import panther as pr
   
   # Optimal configuration for Tensor Cores
   layer = pr.nn.SKLinear(
       in_features=1024,    # Multiple of 16 ✓
       out_features=512,    # Multiple of 16 ✓
       num_terms=4,
       low_rank=64,         # Multiple of 16 ✓
       dtype=torch.float16  # FP16 for Tensor Cores
   )
   
   # Batch size should also be multiple of 16
   x = torch.randn(128, 1024, dtype=torch.float16, device='cuda')
   
   # This will use Tensor Cores automatically
   with torch.cuda.amp.autocast():
       y = layer(x)

**Checking Tensor Core Support**

.. code-block:: python

   from panther.utils.compatibility import has_tensor_core_support
   
   if has_tensor_core_support():
       print("Tensor Core support available")
   else:
       print("Tensor Cores not available")

Memory Optimization Techniques
-------------------------------

**1. Mixed Precision Training**

Use PyTorch's automatic mixed precision for memory efficiency:

.. code-block:: python

   from torch.cuda.amp import autocast, GradScaler
   import torch.nn as nn
   import panther as pr
   
   model = pr.nn.SKLinear(4096, 4096, num_terms=4, low_rank=64).cuda()
   
   scaler = GradScaler()
   optimizer = torch.optim.Adam(model.parameters())
   
   for epoch in range(num_epochs):
       for batch in dataloader:
           optimizer.zero_grad()
           
           with autocast():
               output = model(batch.cuda())
               loss = criterion(output, target.cuda())
           
           scaler.scale(loss).backward()
           scaler.step(optimizer)
           scaler.update()

**2. Gradient Checkpointing**

For very deep networks, use gradient checkpointing:

.. code-block:: python

   import torch.utils.checkpoint as checkpoint
   import torch.nn as nn
   import panther as pr
   
   class MemoryEfficientModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.layers = nn.ModuleList([
               pr.nn.SKLinear(4096, 4096, num_terms=2, low_rank=32)
               for _ in range(20)
           ])
       
       def forward(self, x):
           for layer in self.layers:
               x = checkpoint.checkpoint(layer, x, use_reentrant=False)
           return x

**3. Memory Monitoring**

.. code-block:: python

   import torch
   
   def print_gpu_memory():
       if torch.cuda.is_available():
           allocated = torch.cuda.memory_allocated() / 1024**3
           reserved = torch.cuda.memory_reserved() / 1024**3
           print(f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
   
   print_gpu_memory()
   model = create_model()
   print_gpu_memory()

Performance Best Practices
---------------------------

**1. Dimension Alignment**

For best performance, use dimensions that are multiples of 16:

.. code-block:: python

   # Good - dimensions are multiples of 16
   layer_good = pr.nn.SKLinear(1024, 512, num_terms=4, low_rank=64)
   
   # Suboptimal - dimensions not aligned
   layer_suboptimal = pr.nn.SKLinear(1000, 500, num_terms=3, low_rank=50)

**2. Batch Size Selection**

Use batch sizes that are multiples of 16 for Tensor Core acceleration:

.. code-block:: python

   # Optimal batch sizes: 16, 32, 64, 128, 256, etc.
   x_optimal = torch.randn(128, 1024, device='cuda')
   
   # Suboptimal batch size
   x_suboptimal = torch.randn(100, 1024, device='cuda')

**3. Data Type Selection**

.. code-block:: python

   # FP16 for maximum speed on modern GPUs
   model_fp16 = pr.nn.SKLinear(1024, 512, num_terms=4, low_rank=64, 
                               dtype=torch.float16, device='cuda')
   
   # FP32 for better numerical stability if needed
   model_fp32 = pr.nn.SKLinear(1024, 512, num_terms=4, low_rank=64,
                               dtype=torch.float32, device='cuda')

See Also
--------

* :doc:`../tutorials/performance_optimization` - Detailed performance tuning guide
* :doc:`../examples/performance_benchmarks` - Benchmark results and comparisons

Performance Profiling and Debugging
------------------------------------

**1. Using NVIDIA Nsight**

Profile CUDA kernels with Nsight Compute:

.. code-block:: bash

   # Profile a Python script
   ncu --target-processes all python train_model.py
   
   # Profile specific kernels
   ncu --kernel-name "sketched_linear" python train_model.py

**2. PyTorch Profiler Integration**

.. code-block:: python

   import torch.profiler
   
   with torch.profiler.profile(
       activities=[
           torch.profiler.ProfilerActivity.CPU,
           torch.profiler.ProfilerActivity.CUDA,
       ],
       record_shapes=True,
       with_stack=True,
   ) as prof:
       model = pr.nn.SKLinear(1024, 1024, num_terms=8, low_rank=64).cuda()
       x = torch.randn(128, 1024).cuda()
       
       for _ in range(100):
           y = model(x)
   
   # Export Chrome trace
   prof.export_chrome_trace("trace.json")

**3. Memory Profiling**

.. code-block:: python

   def profile_memory_usage():
       torch.cuda.reset_peak_memory_stats()
       
       # Your model operations here
       model = create_model()
       for batch in dataloader:
           output = model(batch)
       
       peak_memory = torch.cuda.max_memory_allocated() / 1024**3
       print(f"Peak memory usage: {peak_memory:.2f}GB")
       
       return peak_memory

Advanced GPU Features
---------------------

**1. CUDA Streams**

Overlap computation and memory transfers:

.. code-block:: python

   import torch.cuda
   
   # Create CUDA streams
   stream1 = torch.cuda.Stream()
   stream2 = torch.cuda.Stream()
   
   with torch.cuda.stream(stream1):
       # Computation on stream 1
       y1 = model1(x1)
   
   with torch.cuda.stream(stream2):
       # Computation on stream 2 (parallel)
       y2 = model2(x2)
   
   # Synchronize streams
   torch.cuda.synchronize()

**2. Multi-GPU Support**

Scale to multiple GPUs:

.. code-block:: python

   if torch.cuda.device_count() > 1:
       model = nn.DataParallel(model)
   
   # Or use DistributedDataParallel for better performance
   from torch.nn.parallel import DistributedDataParallel as DDP
   
   model = DDP(model, device_ids=[local_rank])

Troubleshooting Common Issues
-----------------------------

**1. Out of Memory Errors**

.. code-block:: python

   # Reduce batch size
   batch_size = 32  # Instead of 128
   
   # Use gradient accumulation
   accumulation_steps = 4
   effective_batch_size = batch_size * accumulation_steps
   
   for i, batch in enumerate(dataloader):
       output = model(batch)
       loss = criterion(output, target) / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()

**2. Slow Kernel Launch**

.. code-block:: python

   # Warmup CUDA kernels
   for _ in range(10):
       dummy_output = model(dummy_input)
   
   torch.cuda.synchronize()
   
   # Now measure actual performance
   start_time = time.time()
   for _ in range(100):
       output = model(input_tensor)
   torch.cuda.synchronize()
   end_time = time.time()

**3. Debugging CUDA Errors**

.. code-block:: python

   # Enable CUDA error checking
   import os
   os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
   
   # Check for CUDA errors explicitly
   try:
       output = model(input_tensor)
       torch.cuda.synchronize()
   except RuntimeError as e:
       print(f"CUDA error: {e}")

This guide provides a comprehensive overview of CUDA optimization in Panther. For more advanced topics, consult the CUDA programming guide and PyTorch documentation.
