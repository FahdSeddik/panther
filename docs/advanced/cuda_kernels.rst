CUDA Kernels and GPU Optimization
==================================

This guide covers Panther's CUDA kernels and how to optimize GPU performance.

Understanding Panther's CUDA Architecture
------------------------------------------

Panther uses custom CUDA kernels to accelerate sketched operations. The main kernels are:

**Linear Operations**
- ``sketched_linear_forward_cuda``: Forward pass for sketched linear layers
- ``sketched_linear_backward_cuda``: Backward pass with gradient computation

**Matrix Operations**
- ``batch_matmul_cuda``: Optimized batch matrix multiplication
- ``tensor_core_matmul``: Tensor Core accelerated matrix multiplication

**Sketching Operations**
- ``dense_sketch_cuda``: Apply dense sketching matrices
- ``sparse_sketch_cuda``: Apply sparse sketching matrices

Tensor Core Optimization
-------------------------

Modern NVIDIA GPUs (V100, A100, RTX 30/40 series) include Tensor Cores for accelerated mixed-precision operations.

**Requirements for Tensor Core Usage**

1. **Dimension Constraints**: All dimensions must be multiples of 16
2. **Data Types**: FP16, BF16, or mixed precision
3. **Memory Alignment**: Proper tensor memory layout

.. code-block:: python

   import torch
   import panther as pr
   
   # Optimal configuration for Tensor Cores
   layer = pr.nn.SKLinear(
       in_features=1024,    # Multiple of 16 ✓
       out_features=512,    # Multiple of 16 ✓
       num_terms=1,
       low_rank=32,         # Multiple of 16 ✓
       dtype=torch.float16  # FP16 on Tensor Cores
   )
   
   # Batch size should also be multiple of 16
   x = torch.randn(128, 1024, dtype=torch.float16, device='cuda')
   
   # This will use Tensor Cores automatically
   with torch.cuda.amp.autocast():
       y = layer(x)

**Checking Tensor Core Usage**

.. code-block:: python

   from panther.utils.compatibility import has_tensor_core_support
   
   print(f"Tensor Core support: {has_tensor_core_support()}")
   
   # Check if specific layer will use Tensor Cores
   layer = pr.nn.SKLinear(1024, 512, num_terms=1, low_rank=64)
   print(f"Layer uses GPU: {layer.use_gpu}")
   print(f"Has Tensor Core support: {layer.has_tensor_core}")

Memory Optimization Techniques
-------------------------------

**1. Gradient Checkpointing**

For very deep networks, use gradient checkpointing to trade computation for memory:

.. code-block:: python

   import torch.utils.checkpoint as checkpoint
   
   class MemoryEfficientModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.layers = nn.ModuleList([
               pr.nn.SKLinear(4096, 4096, num_terms=1, low_rank=16)
               for _ in range(20)  # Very deep network
           ])
       
       def forward(self, x):
           for layer in self.layers:
               # Use checkpointing to save memory
               x = checkpoint.checkpoint(layer, x)
           return x

**2. Mixed Precision Training**

Reduce memory usage and increase speed with automatic mixed precision:

.. code-block:: python

   from torch.cuda.amp import autocast, GradScaler
   
   model = pr.nn.SKLinear(4096, 4096, num_terms=1, low_rank=16)
   model = model.cuda().half()  # Convert to FP16
   
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

**3. Memory Pooling**

Use PyTorch's memory allocator efficiently:

.. code-block:: python

   # Pre-allocate memory pool
   torch.cuda.empty_cache()
   torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
   
   # Monitor memory usage
   def print_gpu_memory():
       allocated = torch.cuda.memory_allocated() / 1024**3
       cached = torch.cuda.memory_reserved() / 1024**3
       print(f"Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
   
   print_gpu_memory()
   model = create_large_model()
   print_gpu_memory()

Custom CUDA Kernel Development
-------------------------------

**Understanding the Kernel Interface**

Panther's CUDA kernels follow this pattern:

.. code-block:: cpp

   // C++ header (linear.h)
   torch::Tensor sketched_linear_forward_cuda(
       const torch::Tensor& input,
       const torch::Tensor& S1s,
       const torch::Tensor& S2s,
       const torch::Tensor& U1s, 
       const torch::Tensor& U2s,
       const torch::Tensor& bias
   );

**Kernel Implementation Structure**

.. code-block:: cuda

   // CUDA kernel (linear_cuda.cu)
   __global__ void sketched_linear_kernel(
       const float* __restrict__ input,     // [batch_size, in_features]
       const float* __restrict__ S1s,       // [num_terms, in_features, low_rank]
       const float* __restrict__ S2s,       // [num_terms, low_rank, out_features]
       const float* __restrict__ U1s,       // [num_terms, low_rank, out_features]
       const float* __restrict__ U2s,       // [num_terms, low_rank, in_features]
       float* __restrict__ output,          // [batch_size, out_features]
       int batch_size,
       int in_features,
       int out_features,
       int num_terms,
       int low_rank
   ) {
       // Kernel implementation
       int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
       int output_idx = blockIdx.y * blockDim.y + threadIdx.y;
       
       if (batch_idx < batch_size && output_idx < out_features) {
           float result = 0.0f;
           
           // Compute sketched linear transformation
           for (int term = 0; term < num_terms; term++) {
               // Implementation details...
           }
           
           output[batch_idx * out_features + output_idx] = result;
       }
   }

**Tensor Core Kernel Example**

.. code-block:: cuda

   #include <mma.h>
   using namespace nvcuda;
   
   __global__ void tensor_core_matmul_kernel(
       const half* A,    // [M, K] - must be aligned
       const half* B,    // [K, N] - must be aligned  
       half* C,          // [M, N] - output
       int M, int N, int K
   ) {
       // Tensor Core fragment declarations
       wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
       wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
       wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
       
       // Initialize accumulator
       wmma::fill_fragment(c_frag, 0.0f);
       
       // Compute matrix multiplication using Tensor Cores
       for (int k = 0; k < K; k += 16) {
           wmma::load_matrix_sync(a_frag, A + k, K);
           wmma::load_matrix_sync(b_frag, B + k * N, N);
           wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
       }
       
       // Store result
       wmma::store_matrix_sync(C, c_frag, N, wmma::mem_row_major);
   }

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

Optimization Best Practices
----------------------------

**1. Kernel Launch Configuration**

Choose optimal block and grid sizes:

.. code-block:: cpp

   // Calculate optimal launch configuration
   int block_size = 256;  // Common choice
   int grid_size = (total_elements + block_size - 1) / block_size;
   
   // 2D grid for matrix operations
   dim3 block_2d(16, 16);
   dim3 grid_2d(
       (width + block_2d.x - 1) / block_2d.x,
       (height + block_2d.y - 1) / block_2d.y
   );

**2. Memory Access Patterns**

Optimize for coalesced memory access:

.. code-block:: cuda

   // Good: Coalesced access
   __global__ void coalesced_kernel(float* data, int width) {
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       int idy = blockIdx.y * blockDim.y + threadIdx.y;
       
       // Adjacent threads access adjacent memory locations
       data[idy * width + idx] = computation(idx, idy);
   }

**3. Shared Memory Usage**

Use shared memory for frequently accessed data:

.. code-block:: cuda

   __global__ void shared_memory_kernel(float* input, float* output) {
       __shared__ float shared_data[256];
       
       int tid = threadIdx.x;
       int gid = blockIdx.x * blockDim.x + threadIdx.x;
       
       // Load data into shared memory
       shared_data[tid] = input[gid];
       __syncthreads();
       
       // Compute using shared memory
       float result = 0.0f;
       for (int i = 0; i < blockDim.x; i++) {
           result += shared_data[i] * shared_data[tid];
       }
       
       output[gid] = result;
   }

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

**3. Dynamic Shapes and Compilation**

Use TorchScript for optimization:

.. code-block:: python

   # JIT compile for better performance
   @torch.jit.script
   def optimized_function(x: torch.Tensor, layer: pr.nn.SKLinear) -> torch.Tensor:
       return layer(x)
   
   # Trace the model
   traced_model = torch.jit.trace(model, example_input)

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
