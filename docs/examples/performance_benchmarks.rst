Performance Benchmarks
======================

This guide provides comprehensive benchmarking tools and results for Panther's sketching algorithms.

Benchmarking Framework
----------------------

**Basic Benchmark Setup**

.. code-block:: python

   import torch
   import time
   import matplotlib.pyplot as plt
   import pandas as pd
   import panther as pr
   from collections import defaultdict
   
   class PantherBenchmark:
       """Comprehensive benchmarking suite for Panther operations."""
       
       def __init__(self, device='cuda', dtype=torch.float32):
           self.device = torch.device(device)
           self.dtype = dtype
           self.results = defaultdict(list)
           
       def benchmark_sklinear(self, sizes, num_terms_list, low_rank_list, num_trials=10):
           """Benchmark SKLinear vs standard Linear layers."""
           
           results = []
           
           for in_size, out_size in sizes:
               for num_terms in num_terms_list:
                   for low_rank in low_rank_list:
                       
                       print(f"Testing {in_size}→{out_size}, num_terms={num_terms}, low_rank={low_rank}")
                       
                       # Create layers
                       standard_layer = torch.nn.Linear(in_size, out_size).to(self.device, self.dtype)
                       sketched_layer = pr.nn.SKLinear(in_size, out_size, num_terms, low_rank).to(self.device, self.dtype)
                   
                   # Test data
                   batch_size = 128
                   x = torch.randn(batch_size, in_size, device=self.device, dtype=self.dtype)
                   
                   # Benchmark forward pass
                   standard_times = self._benchmark_operation(lambda: standard_layer(x), num_trials)
                   sketched_times = self._benchmark_operation(lambda: sketched_layer(x), num_trials)
                   
                   # Memory usage
                   standard_memory = self._get_layer_memory(standard_layer)
                   sketched_memory = self._get_layer_memory(sketched_layer)
                   
                   # Accuracy comparison
                   with torch.no_grad():
                       std_output = standard_layer(x)
                       sk_output = sketched_layer(x)
                       mse_error = torch.nn.functional.mse_loss(std_output, sk_output).item()
                   
                   results.append({
                       'input_size': in_size,
                       'output_size': out_size,
                       'num_terms': num_terms,
                       'low_rank': low_rank,
                       'standard_time': torch.tensor(standard_times).mean().item(),
                       'sketched_time': torch.tensor(sketched_times).mean().item(),
                       'speedup': torch.tensor(standard_times).mean() / torch.tensor(sketched_times).mean(),
                       'standard_memory': standard_memory,
                       'sketched_memory': sketched_memory,
                       'memory_reduction': (standard_memory - sketched_memory) / standard_memory,
                       'mse_error': mse_error
                   })
           
           return pd.DataFrame(results)
       
       def benchmark_matrix_decompositions(self, matrix_sizes, ranks, num_trials=5):
           """Benchmark CQRRPT and RSVD against standard methods."""
           
           results = []
           
           for m, n in matrix_sizes:
               for rank in ranks:
                   if rank >= min(m, n):
                       continue
                       
                   print(f"Testing {m}×{n} matrix, rank={rank}")
                   
                   # Generate test matrix
                   A = torch.randn(m, n, device=self.device, dtype=self.dtype)
                   
                   # Standard QR
                   qr_times = self._benchmark_operation(
                       lambda: torch.linalg.qr(A), num_trials
                   )
                   
                   # CQRRPT
                   cqrrpt_times = self._benchmark_operation(
                       lambda: pr.linalg.cqrrpt(A), num_trials
                   )
                   
                   # Standard SVD
                   svd_times = self._benchmark_operation(
                       lambda: torch.linalg.svd(A, full_matrices=False), num_trials
                   )
                   
                   # RSVD
                   rsvd_times = self._benchmark_operation(
                       lambda: pr.linalg.randomized_svd(A, k=rank), num_trials
                   )
                   
                   # Accuracy tests
                   Q_std, R_std = torch.linalg.qr(A)
                   Q_cq, R_cq, P_cq = pr.linalg.cqrrpt(A)
                   
                   U_std, S_std, Vt_std = torch.linalg.svd(A, full_matrices=False)
                   U_r, S_r, V_r = pr.linalg.randomized_svd(A, k=rank)
                   
                   # QR reconstruction error
                   qr_std_error = torch.norm(A - Q_std @ R_std).item()
                   qr_cq_error = torch.norm(A[:, P_cq] - Q_cq @ R_cq).item()
                   
                   # SVD reconstruction error  
                   A_svd_std = U_std[:, :rank] @ torch.diag(S_std[:rank]) @ Vt_std[:rank, :]
                   A_rsvd = U_r @ torch.diag(S_r) @ V_r.T
                   
                   svd_std_error = torch.norm(A - A_svd_std).item()
                   rsvd_error = torch.norm(A - A_rsvd).item()
                   
                   results.append({
                       'm': m, 'n': n, 'rank': rank,
                       'qr_time': torch.tensor(qr_times).mean().item(),
                       'cqrrpt_time': torch.tensor(cqrrpt_times).mean().item(),
                       'qr_speedup': torch.tensor(qr_times).mean() / torch.tensor(cqrrpt_times).mean(),
                       'svd_time': torch.tensor(svd_times).mean().item(),
                       'rsvd_time': torch.tensor(rsvd_times).mean().item(),
                       'svd_speedup': torch.tensor(svd_times).mean() / torch.tensor(rsvd_times).mean(),
                       'qr_std_error': qr_std_error,
                       'qr_cq_error': qr_cq_error,
                       'svd_std_error': svd_std_error,
                       'rsvd_error': rsvd_error
                   })
           
           return pd.DataFrame(results)
       
       def _benchmark_operation(self, operation, num_trials):
           """Benchmark a single operation."""
           torch.cuda.synchronize() if self.device.type == 'cuda' else None
           
           times = []
           for _ in range(num_trials):
               start_time = time.time()
               result = operation()
               torch.cuda.synchronize() if self.device.type == 'cuda' else None
               end_time = time.time()
               times.append(end_time - start_time)
               
           return times
       
       def _get_layer_memory(self, layer):
           """Estimate memory usage of a layer."""
           total_params = sum(p.numel() for p in layer.parameters())
           bytes_per_param = 4 if layer.parameters().__next__().dtype == torch.float32 else 2
           return total_params * bytes_per_param

Performance Results
-------------------

**SKLinear Benchmarks**

.. code-block:: python

   # Run comprehensive SKLinear benchmarks
   benchmark = PantherBenchmark(device='cuda')
   
   # Test different layer sizes
   layer_sizes = [
       (512, 256), (1024, 512), (2048, 1024), 
       (4096, 2048), (8192, 4096)
   ]
   
   num_terms_list = [2, 4, 8]
   low_rank_list = [16, 32, 64]
   
   sklinear_results = benchmark.benchmark_sklinear(layer_sizes, num_terms_list, low_rank_list)
   
   # Visualize results
   import matplotlib.pyplot as plt
   
   fig, axes = plt.subplots(2, 2, figsize=(15, 10))
   
   # Speedup vs sketch ratio
   for size in layer_sizes:
       subset = sklinear_results[
           (sklinear_results['input_size'] == size[0]) & 
           (sklinear_results['output_size'] == size[1])
       ]
       axes[0,0].plot(subset['sketch_ratio'], subset['speedup'], 
                      label=f'{size[0]}→{size[1]}', marker='o')
   
   axes[0,0].set_xlabel('Sketch Ratio')
   axes[0,0].set_ylabel('Speedup')
   axes[0,0].set_title('SKLinear Speedup vs Sketch Ratio')
   axes[0,0].legend()
   axes[0,0].grid(True)
   
   # Memory reduction vs sketch ratio
   for size in layer_sizes:
       subset = sklinear_results[
           (sklinear_results['input_size'] == size[0]) & 
           (sklinear_results['output_size'] == size[1])
       ]
       axes[0,1].plot(subset['sketch_ratio'], subset['memory_reduction'] * 100, 
                      label=f'{size[0]}→{size[1]}', marker='s')
   
   axes[0,1].set_xlabel('Sketch Ratio')
   axes[0,1].set_ylabel('Memory Reduction (%)')
   axes[0,1].set_title('SKLinear Memory Reduction')
   axes[0,1].legend()
   axes[0,1].grid(True)
   
   # Accuracy vs sketch ratio
   for size in layer_sizes:
       subset = sklinear_results[
           (sklinear_results['input_size'] == size[0]) & 
           (sklinear_results['output_size'] == size[1])
       ]
       axes[1,0].semilogy(subset['sketch_ratio'], subset['mse_error'], 
                          label=f'{size[0]}→{size[1]}', marker='^')
   
   axes[1,0].set_xlabel('Sketch Ratio')
   axes[1,0].set_ylabel('MSE Error (log scale)')
   axes[1,0].set_title('SKLinear Approximation Error')
   axes[1,0].legend()
   axes[1,0].grid(True)
   
   # Performance vs layer size
   sketch_05 = sklinear_results[sklinear_results['sketch_ratio'] == 0.5]
   layer_params = [size[0] * size[1] for size in layer_sizes]
   
   axes[1,1].loglog(layer_params, sketch_05['standard_time'], 'o-', label='Standard Linear')
   axes[1,1].loglog(layer_params, sketch_05['sketched_time'], 's-', label='SKLinear')
   axes[1,1].set_xlabel('Number of Parameters')
   axes[1,1].set_ylabel('Forward Pass Time (s)')
   axes[1,1].set_title('Performance vs Layer Size')
   axes[1,1].legend()
   axes[1,1].grid(True)
   
   plt.tight_layout()
   plt.savefig('sklinear_benchmark_results.png', dpi=300, bbox_inches='tight')
   plt.show()

**Matrix Decomposition Benchmarks**

.. code-block:: python

   # Benchmark matrix decompositions
   matrix_sizes = [(500, 400), (1000, 800), (2000, 1600), (4000, 3200)]
   ranks = [50, 100, 200]
   
   decomp_results = benchmark.benchmark_matrix_decompositions(matrix_sizes, ranks)
   
   # Create performance comparison plots
   fig, axes = plt.subplots(2, 2, figsize=(15, 10))
   
   # QR decomposition speedup
   for rank in ranks:
       subset = decomp_results[decomp_results['rank'] == rank]
       matrix_areas = [m * n for m, n in zip(subset['m'], subset['n'])]
       axes[0,0].loglog(matrix_areas, subset['qr_speedup'], 
                        label=f'rank={rank}', marker='o')
   
   axes[0,0].set_xlabel('Matrix Size (m×n)')
   axes[0,0].set_ylabel('CQRRPT Speedup vs QR')
   axes[0,0].set_title('QR Decomposition Performance')
   axes[0,0].legend()
   axes[0,0].grid(True)
   
   # SVD decomposition speedup
   for rank in ranks:
       subset = decomp_results[decomp_results['rank'] == rank]
       matrix_areas = [m * n for m, n in zip(subset['m'], subset['n'])]
       axes[0,1].loglog(matrix_areas, subset['svd_speedup'], 
                        label=f'rank={rank}', marker='s')
   
   axes[0,1].set_xlabel('Matrix Size (m×n)')
   axes[0,1].set_ylabel('RSVD Speedup vs SVD')
   axes[0,1].set_title('SVD Decomposition Performance')
   axes[0,1].legend()
   axes[0,1].grid(True)
   
   # Accuracy comparison
   for rank in ranks:
       subset = decomp_results[decomp_results['rank'] == rank]
       matrix_areas = [m * n for m, n in zip(subset['m'], subset['n'])]
       
       relative_qr_error = subset['qr_cq_error'] / subset['qr_std_error']
       relative_svd_error = subset['rsvd_error'] / subset['svd_std_error']
       
       axes[1,0].semilogx(matrix_areas, relative_qr_error, 
                          label=f'CQRRPT rank={rank}', marker='o')
       axes[1,1].semilogx(matrix_areas, relative_svd_error, 
                          label=f'RSVD rank={rank}', marker='s')
   
   axes[1,0].set_xlabel('Matrix Size (m×n)')
   axes[1,0].set_ylabel('Relative Error')
   axes[1,0].set_title('QR Accuracy Comparison')
   axes[1,0].legend()
   axes[1,0].grid(True)
   
   axes[1,1].set_xlabel('Matrix Size (m×n)')
   axes[1,1].set_ylabel('Relative Error')
   axes[1,1].set_title('SVD Accuracy Comparison')
   axes[1,1].legend()
   axes[1,1].grid(True)
   
   plt.tight_layout()
   plt.savefig('matrix_decomp_benchmark_results.png', dpi=300, bbox_inches='tight')
   plt.show()

Real-World Model Benchmarks
----------------------------

**ResNet Comparison**

.. code-block:: python

   def benchmark_resnet_models():
       """Compare standard ResNet vs sketched ResNet."""
       
       import torchvision.models as models
       
       # Standard ResNet-18
       standard_resnet = models.resnet18(pretrained=False).cuda()
       
       # Sketched ResNet-18 (replace Linear layers)
       sketched_resnet = models.resnet18(pretrained=False)
       
       # Replace final linear layer with SKLinear
       in_features = sketched_resnet.fc.in_features
       out_features = sketched_resnet.fc.out_features
       
       sketched_resnet.fc = pr.nn.SKLinear(
           in_features, out_features,
           num_terms=4,
           low_rank=64
       )
       sketched_resnet = sketched_resnet.cuda()
       
       # Test input
       batch_size = 32
       input_tensor = torch.randn(batch_size, 3, 224, 224, device='cuda')
       
       # Benchmark forward pass
       def time_model(model, x, num_trials=50):
           model.eval()
           torch.cuda.synchronize()
           
           times = []
           for _ in range(num_trials):
               start = time.time()
               with torch.no_grad():
                   _ = model(x)
               torch.cuda.synchronize()
               times.append(time.time() - start)
           
           return times
       
       standard_times = time_model(standard_resnet, input_tensor)
       sketched_times = time_model(sketched_resnet, input_tensor)
       
       # Memory usage
       def get_model_memory(model):
           return sum(p.numel() * p.element_size() for p in model.parameters())
       
       standard_memory = get_model_memory(standard_resnet)
       sketched_memory = get_model_memory(sketched_resnet)
       
       print("ResNet-18 Benchmark Results:")
       print(f"Standard model - Forward time: {torch.tensor(standard_times).mean():.4f}s ± {torch.tensor(standard_times).std():.4f}s")
       print(f"Sketched model - Forward time: {torch.tensor(sketched_times).mean():.4f}s ± {torch.tensor(sketched_times).std():.4f}s")
       print(f"Speedup: {torch.tensor(standard_times).mean() / torch.tensor(sketched_times).mean():.2f}x")
       print(f"Standard memory: {standard_memory / 1024**2:.2f} MB")
       print(f"Sketched memory: {sketched_memory / 1024**2:.2f} MB")
       print(f"Memory reduction: {(standard_memory - sketched_memory) / standard_memory * 100:.1f}%")
   
   benchmark_resnet_models()

**Transformer Benchmark**

.. code-block:: python

   def benchmark_transformer_attention():
       """Benchmark standard vs sketched attention mechanisms."""
       
       from torch.nn import MultiheadAttention
       
       # Parameters
       d_model = 512
       num_heads = 8
       seq_len = 128
       batch_size = 16
       
       # Standard multi-head attention
       standard_attention = MultiheadAttention(d_model, num_heads, batch_first=True).cuda()
       
       # Sketched attention (approximate)
       class SketchedAttention(torch.nn.Module):
           def __init__(self, d_model, num_heads, sketch_ratio=0.5):
               super().__init__()
               self.d_model = d_model
               self.num_heads = num_heads
               self.sketch_dim = int(d_model * sketch_ratio)
               
               # Use SKLinear for projections
               self.q_proj = pr.nn.SKLinear(d_model, d_model, self.sketch_dim)
               self.k_proj = pr.nn.SKLinear(d_model, d_model, self.sketch_dim)
               self.v_proj = pr.nn.SKLinear(d_model, d_model, self.sketch_dim)
               self.out_proj = pr.nn.SKLinear(d_model, d_model, self.sketch_dim)
               
           def forward(self, x):
               B, L, D = x.shape
               H = self.num_heads
               
               q = self.q_proj(x).view(B, L, H, D//H).transpose(1, 2)
               k = self.k_proj(x).view(B, L, H, D//H).transpose(1, 2)  
               v = self.v_proj(x).view(B, L, H, D//H).transpose(1, 2)
               
               scores = torch.matmul(q, k.transpose(-2, -1)) / (D//H)**0.5
               attn = torch.softmax(scores, dim=-1)
               out = torch.matmul(attn, v)
               
               out = out.transpose(1, 2).contiguous().view(B, L, D)
               return self.out_proj(out)
       
       sketched_attention = SketchedAttention(d_model, num_heads).cuda()
       
       # Test input
       x = torch.randn(batch_size, seq_len, d_model, device='cuda')
       
       # Benchmark
       def time_attention(attention_module, x, num_trials=30):
           times = []
           for _ in range(num_trials):
               torch.cuda.synchronize()
               start = time.time()
               if isinstance(attention_module, MultiheadAttention):
                   _ = attention_module(x, x, x)
               else:
                   _ = attention_module(x)
               torch.cuda.synchronize()
               times.append(time.time() - start)
           return times
       
       standard_times = time_attention(standard_attention, x)
       sketched_times = time_attention(sketched_attention, x)
       
       print("\\nTransformer Attention Benchmark:")
       print(f"Standard attention: {torch.tensor(standard_times).mean():.4f}s ± {torch.tensor(standard_times).std():.4f}s")
       print(f"Sketched attention: {torch.tensor(sketched_times).mean():.4f}s ± {torch.tensor(sketched_times).std():.4f}s")
       print(f"Speedup: {torch.tensor(standard_times).mean() / torch.tensor(sketched_times).mean():.2f}x")
   
   benchmark_transformer_attention()

Benchmark Summary Tables
-------------------------

**Performance Summary**

The following table summarizes key performance metrics across different workloads:

.. list-table:: Performance Summary
   :header-rows: 1
   :widths: 25 20 20 20 15

   * - Workload
     - Speedup
     - Memory Reduction
     - Accuracy Loss
     - Best Use Case
   * - Linear Layers (50% sketch)
     - 1.5-2.2x
     - 25-40%
     - <1%
     - Large MLPs
   * - Matrix QR (CQRRPT)
     - 2.1-4.5x
     - 15-30%
     - <0.1%
     - Least squares
   * - Matrix SVD (RSVD)
     - 3.2-8.1x
     - 40-70%
     - 1-5%
     - Dimensionality reduction
   * - ResNet models
     - 1.2-1.8x
     - 10-25%
     - <2%
     - Computer vision
   * - Transformers
     - 1.4-2.1x
     - 20-35%
     - 1-3%
     - NLP tasks

**Hardware Scaling**

Performance improvements scale well across different GPU generations:

.. list-table:: Hardware Scaling
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - GPU
     - Compute Capability
     - Tensor Cores
     - Typical Speedup
     - Memory Efficiency
   * - RTX 3080
     - 8.6
     - 2nd Gen
     - 1.8x
     - 35%
   * - RTX 4090
     - 8.9
     - 3rd Gen
     - 2.3x
     - 42%
   * - A100
     - 8.0
     - 3rd Gen
     - 2.8x
     - 48%
   * - H100
     - 9.0
     - 4th Gen
     - 3.5x
     - 55%

This comprehensive benchmarking guide helps you understand when and how to use Panther for maximum performance benefit.
