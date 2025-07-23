Basic Usage Examples
====================

This page demonstrates common usage patterns and examples for Panther.

Matrix Decompositions
---------------------

**Randomized QR Decomposition**

.. code-block:: python

   import torch
   import panther as pr
   import matplotlib.pyplot as plt
   
   # Create a test matrix
   m, n = 1000, 600
   A = torch.randn(m, n, dtype=torch.float32)
   
   # Add some structure (low rank + noise)
   U = torch.randn(m, 50)
   V = torch.randn(50, n) 
   A = U @ V + 0.1 * torch.randn(m, n)
   
   print(f"Original matrix: {A.shape}")
   
   # Perform CQRRPT decomposition
   Q, R, J = pr.linalg.cqrrpt(A, gamma=1.25)
   
   print(f"Q (orthogonal): {Q.shape}")
   print(f"R (triangular): {R.shape}")
   print(f"J (permutation): {J.shape}")
   
   # Verify orthogonality of Q
   I_approx = Q.T @ Q
   orthogonality_error = torch.norm(I_approx - torch.eye(Q.shape[1]))
   print(f"Orthogonality error: {orthogonality_error:.2e}")
   
   # Verify reconstruction
   A_reconstructed = Q @ R[:, J]
   reconstruction_error = torch.norm(A - A_reconstructed)
   print(f"Reconstruction error: {reconstruction_error:.2e}")

**Randomized SVD**

.. code-block:: python

   import torch
   import panther as pr
   import numpy as np
   
   # Create a matrix with known rank
   true_rank = 25
   m, n = 800, 600
   
   U_true = torch.randn(m, true_rank)
   S_true = torch.linspace(10, 1, true_rank)  # Decaying singular values
   V_true = torch.randn(n, true_rank)
   
   A = U_true @ torch.diag(S_true) @ V_true.T
   A += 0.01 * torch.randn(m, n)  # Add small amount of noise
   
   print(f"Matrix shape: {A.shape}")
   print(f"True rank: {true_rank}")
   
   # Compute randomized SVD
   k = 30  # Slightly overestimate rank
   U, S, V = pr.linalg.randomized_svd(A, k=k, tol=1e-6)
   
   print(f"Computed SVD: U{U.shape}, S{S.shape}, V{V.shape}")
   
   # Compare singular values
   print("\\nSingular values comparison:")
   print("True (first 10):", S_true[:10].numpy())
   print("Computed (first 10):", S[:10].numpy())
   
   # Compute approximation error
   A_approx = U @ torch.diag(S) @ V.T
   approx_error = torch.norm(A - A_approx) / torch.norm(A)
   print(f"\\nRelative approximation error: {approx_error:.2e}")
   
   # Plot singular values
   plt.figure(figsize=(10, 6))
   plt.semilogy(S_true.numpy(), 'o-', label='True', markersize=4)
   plt.semilogy(S.numpy(), 's-', label='Computed', markersize=4)
   plt.xlabel('Index')
   plt.ylabel('Singular Value')
   plt.title('Singular Value Comparison')
   plt.legend()
   plt.grid(True)
   plt.show()

Linear Layer Replacements
--------------------------

**Direct Replacement of nn.Linear**

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   
   # Original model with standard linear layers
   class StandardMLP(nn.Module):
       def __init__(self):
           super().__init__()
           self.layers = nn.Sequential(
               nn.Linear(4096, 2048),
               nn.ReLU(),
               nn.Dropout(0.2),
               nn.Linear(2048, 8192),
               nn.ReLU(), 
               nn.Dropout(0.2),
               nn.Linear(8192, 4096),
               nn.ReLU(),
               nn.Linear(4096, 2048)
           )
           
       def forward(self, x):
           return self.layers(x)
   
   # Sketched version - drop-in replacement
   class SketchedMLP(nn.Module):
       def __init__(self):
           super().__init__()
           self.layers = nn.Sequential(
               pr.nn.SKLinear(4096, 2048, num_terms=1, low_rank=16),
               nn.ReLU(),
               nn.Dropout(0.2),
               pr.nn.SKLinear(2048, 8192, num_terms=1, low_rank=16),
               nn.ReLU(),
               nn.Dropout(0.2), 
               pr.nn.SKLinear(8192, 4096, num_terms=1, low_rank=16),
               nn.ReLU(),
               pr.nn.SKLinear(4096, 2048, num_terms=1, low_rank=16)
           )
           
       def forward(self, x):
           return self.layers(x)
   
   # Compare parameter counts
   standard_model = StandardMLP()
   sketched_model = SketchedMLP()
   
   standard_params = sum(p.numel() for p in standard_model.parameters())
   sketched_params = sum(p.numel() for p in sketched_model.parameters())
   
   print(f"Standard model parameters: {standard_params:,}")
   print(f"Sketched model parameters: {sketched_params:,}")
   print(f"Parameter reduction: {(1 - sketched_params/standard_params)*100:.1f}%")
   
   # Test with sample input
   x = torch.randn(32, 784)  # Batch of 32 MNIST-like samples
   
   y_standard = standard_model(x)
   y_sketched = sketched_model(x)
   
   print(f"\\nOutput shapes:")
   print(f"Standard: {y_standard.shape}")
   print(f"Sketched: {y_sketched.shape}")

**Performance Comparison**

.. code-block:: python

   import time
   import torch
   import torch.nn as nn
   import panther as pr
   
   def benchmark_layer(layer, input_tensor, num_runs=100):
       \"\"\"Benchmark forward and backward pass.\"\"\\"
       layer.train()
       
       # Warmup
       for _ in range(10):
           output = layer(input_tensor)
           loss = output.sum()
           loss.backward()
           layer.zero_grad()
       
       # Time forward pass
       torch.cuda.synchronize() if input_tensor.is_cuda else None
       start_time = time.time()
       
       for _ in range(num_runs):
           output = layer(input_tensor)
           torch.cuda.synchronize() if input_tensor.is_cuda else None
           
       forward_time = (time.time() - start_time) / num_runs
       
       # Time backward pass  
       torch.cuda.synchronize() if input_tensor.is_cuda else None
       start_time = time.time()
       
       for _ in range(num_runs):
           output = layer(input_tensor)
           loss = output.sum()
           loss.backward()
           layer.zero_grad()
           torch.cuda.synchronize() if input_tensor.is_cuda else None
           
       backward_time = (time.time() - start_time) / num_runs - forward_time
       
       return forward_time * 1000, backward_time * 1000  # Convert to ms
   
   # Compare different layer sizes
   layer_configs = [
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
        (16384, 16384)
   ]
   
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
   print(f"Benchmarking on {device}")
   print("=" * 80)
   print(f"{'Layer Size':<15} {'Standard (ms)':<20} {'Sketched (ms)':<20} {'Speedup':<10}")
   print("=" * 80)
   
   for in_features, out_features in layer_configs:
       # Create layers
       standard = nn.Linear(in_features, out_features).to(device)
       sketched = pr.nn.SKLinear(
           in_features, out_features, 
           num_terms=8, low_rank=min(64, min(in_features, out_features)//8)
       ).to(device)
       
       # Create input
       batch_size = 64
       x = torch.randn(batch_size, in_features, device=device)
       
       # Benchmark
       std_forward, std_backward = benchmark_layer(standard, x)
       sk_forward, sk_backward = benchmark_layer(sketched, x)
       
       std_total = std_forward + std_backward
       sk_total = sk_forward + sk_backward
       speedup = std_total / sk_total
       
       print(f"{in_features}→{out_features:<8} {std_total:<20.2f} {sk_total:<20.2f} {speedup:<10.2f}x")

Memory Usage Analysis
---------------------

**Memory Profiling**

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   import psutil
   import os
   
   def get_memory_usage():
       \"\"\"Get current memory usage in MB.\"\"\""
       if torch.cuda.is_available():
           gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
           return f"GPU: {gpu_memory:.1f}MB"
       else:
           process = psutil.Process(os.getpid())
           cpu_memory = process.memory_info().rss / 1024**2  # MB
           return f"CPU: {cpu_memory:.1f}MB"
   
   print("Memory usage comparison")
   print("=" * 50)
   
   initial_memory = get_memory_usage()
   print(f"Initial memory: {initial_memory}")
   
   # Large standard model
   print("\\nCreating standard model...")
   standard_model = nn.Sequential(
       nn.Linear(4096, 4096),
       nn.ReLU(),
       nn.Linear(4096, 4096),
       nn.ReLU(), 
       nn.Linear(4096, 1024)
   )
   
   if torch.cuda.is_available():
       standard_model = standard_model.cuda()
   
   standard_memory = get_memory_usage()
   print(f"After standard model: {standard_memory}")
   
   # Clean up
   del standard_model
   torch.cuda.empty_cache() if torch.cuda.is_available() else None
   
   # Equivalent sketched model
   print("\\nCreating sketched model...")
   sketched_model = nn.Sequential(
       pr.nn.SKLinear(4096, 4096, num_terms=1, low_rank=64),
       nn.ReLU(),
       pr.nn.SKLinear(4096, 4096, num_terms=1, low_rank=64),
       nn.ReLU(),
       pr.nn.SKLinear(4096, 1024, num_terms=1, low_rank=64)
   )
   
   if torch.cuda.is_available():
       sketched_model = sketched_model.cuda()
   
   sketched_memory = get_memory_usage()
   print(f"After sketched model: {sketched_memory}")

Sketching Examples
------------------

**Dimensionality Reduction**

.. code-block:: python

   import torch
   import panther as pr
   import numpy as np
   from sklearn.datasets import make_classification
   from sklearn.decomposition import PCA
   
   # Generate high-dimensional data
   X_np, y_np = make_classification(
       n_samples=1000, 
       n_features=2000,
       n_informative=50,
       n_redundant=50,
       random_state=42
   )
   
   X = torch.from_numpy(X_np).float()
   print(f"Original data shape: {X.shape}")
   
   # Method 1: Sketching-based dimensionality reduction
   target_dim = 100
   sketch_matrix = pr.sketch.dense_sketch_operator(
       m=target_dim,
       n=X.shape[1],
       distribution=pr.sketch.DistributionFamily.Gaussian
   )
   
   X_sketched = X @ sketch_matrix.T
   print(f"Sketched data shape: {X_sketched.shape}")
   
   # Method 2: PCA for comparison
   pca = PCA(n_components=target_dim)
   X_pca = torch.from_numpy(pca.fit_transform(X_np)).float()
   print(f"PCA data shape: {X_pca.shape}")
   
   # Compare preservation of distances
   def compare_distances(X_orig, X_reduced, n_samples=100):
       \"\"\"Compare pairwise distances before/after reduction.\"\"\""
       indices = torch.randperm(X_orig.shape[0])[:n_samples]
       
       orig_dists = torch.pdist(X_orig[indices])
       reduced_dists = torch.pdist(X_reduced[indices])
       
       correlation = torch.corrcoef(torch.stack([orig_dists, reduced_dists]))[0, 1]
       return correlation.item()
   
   sketch_correlation = compare_distances(X, X_sketched)
   pca_correlation = compare_distances(X, X_pca)
   
   print(f"\\nDistance preservation:")
   print(f"Sketching correlation: {sketch_correlation:.3f}")
   print(f"PCA correlation: {pca_correlation:.3f}")

**Fast Matrix Multiplication**

.. code-block:: python

   import torch
   import panther as pr
   import time
   
   def fast_matrix_multiply(A, B, sketch_size=None):
       \"\"\"Approximate A @ B using sketching.\"\"\""
       m, k = A.shape
       k2, n = B.shape
       assert k == k2, "Matrix dimensions must match"
       
       if sketch_size is None:
           sketch_size = min(m, n) // 2
       
       # Sketch A from the left
       S = pr.sketch.dense_sketch_operator(
           m=sketch_size,
           n=m,
           distribution=pr.sketch.DistributionFamily.Gaussian
       )
       
       SA = S @ A  # (sketch_size, k)
       
       # Compute SA @ B
       SAB = SA @ B  # (sketch_size, n)
       
       # Reconstruct approximation using pseudoinverse
       S_pinv = torch.linalg.pinv(S)
       approx_result = S_pinv @ SAB
       
       return approx_result
   
   # Test with large matrices
   m, k, n = 2000, 1500, 1000
   A = torch.randn(m, k)
   B = torch.randn(k, n)
   
   print(f"Matrix sizes: A{A.shape}, B{B.shape}")
   
   # Exact computation
   start_time = time.time()
   exact_result = A @ B
   exact_time = time.time() - start_time
   
   # Sketched computation  
   start_time = time.time()
   approx_result = fast_matrix_multiply(A, B, sketch_size=500)
   sketch_time = time.time() - start_time
   
   # Compare results
   error = torch.norm(exact_result - approx_result) / torch.norm(exact_result)
   speedup = exact_time / sketch_time
   
   print(f"\\nTiming:")
   print(f"Exact computation: {exact_time:.3f}s")
   print(f"Sketched computation: {sketch_time:.3f}s")
   print(f"Speedup: {speedup:.2f}x")
   print(f"Relative error: {error:.2e}")

Advanced Usage Patterns
------------------------

**Custom Layer with Sketching**

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   
   class AdaptiveSketchedLayer(nn.Module):
       \"\"\"Layer that adapts sketching parameters based on input.\"\"\""
       
       def __init__(self, in_features, out_features, max_terms=16, max_rank=128):
           super().__init__()
           self.in_features = in_features
           self.out_features = out_features
           self.max_terms = max_terms
           self.max_rank = max_rank
           
           # Create multiple sketched layers with different parameters
           self.layers = nn.ModuleList([
               pr.nn.SKLinear(in_features, out_features, 
                             num_terms=terms, low_rank=rank)
               for terms, rank in [(4, 32), (8, 64), (16, 128)]
           ])
           
           # Selection network
           self.selector = nn.Sequential(
               nn.Linear(in_features, 64),
               nn.ReLU(),
               nn.Linear(64, len(self.layers)),
               nn.Softmax(dim=-1)
           )
       
       def forward(self, x):
           # Compute selection weights based on input statistics
           input_norm = torch.norm(x, dim=-1, keepdim=True)
           input_std = torch.std(x, dim=-1, keepdim=True)
           input_features = torch.cat([input_norm, input_std], dim=-1)
           
           # Get selection weights
           weights = self.selector(input_features)  # (batch, num_layers)
           
           # Compute outputs from all layers
           outputs = torch.stack([layer(x) for layer in self.layers], dim=-1)
           
           # Weighted combination
           weighted_output = torch.sum(outputs * weights.unsqueeze(-2), dim=-1)
           
           return weighted_output
   
   # Test the adaptive layer
   adaptive_layer = AdaptiveSketchedLayer(512, 256)
   
   # Different types of inputs
   normal_input = torch.randn(32, 512)
   sparse_input = torch.randn(32, 512) * (torch.rand(32, 512) > 0.8)
   
   normal_output = adaptive_layer(normal_input)
   sparse_output = adaptive_layer(sparse_input)
   
   print(f"Normal input output: {normal_output.shape}")
   print(f"Sparse input output: {sparse_output.shape}")

**Progressive Sketching Training**

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   
   class ProgressiveSketchedModel(nn.Module):
       \"\"\"Model that gradually increases sketching complexity during training.\"\"\""
       
       def __init__(self, layer_sizes):
           super().__init__()
           self.layer_sizes = layer_sizes
           self.num_layers = len(layer_sizes) - 1
           
           # Start with minimal sketching
           self.layers = nn.ModuleList([
               pr.nn.SKLinear(layer_sizes[i], layer_sizes[i+1], 
                             num_terms=2, low_rank=16)
               for i in range(self.num_layers)
           ])
           
           self.training_step = 0
           
       def increase_complexity(self):
           \"\"\"Increase sketching parameters.\"\"\""
           for layer in self.layers:
               if hasattr(layer, 'num_terms'):
                   # Gradually increase parameters
                   new_terms = min(layer.num_terms + 1, 16)
                   new_rank = min(layer.low_rank + 8, 128)
                   
                   # Create new layer with increased parameters
                   new_layer = pr.nn.SKLinear(
                       layer.in_features, layer.out_features,
                       num_terms=new_terms, low_rank=new_rank
                   )
                   
                   # Copy existing parameters (simplified)
                   with torch.no_grad():
                       if layer.bias is not None:
                           new_layer.bias.copy_(layer.bias)
                   
                   # Replace layer
                   layer = new_layer
       
       def forward(self, x):
           for layer in self.layers:
               x = torch.relu(layer(x))
           return x
       
       def training_step_callback(self):
           \"\"\"Call this every few training steps.\"\"\""
           self.training_step += 1
           if self.training_step % 1000 == 0:  # Every 1000 steps
               self.increase_complexity()
               print(f"Increased model complexity at step {self.training_step}")
   
   # Example usage
   model = ProgressiveSketchedModel([784, 512, 256, 10])
   
   # Simulate training
   for step in range(5000):
       # Your training code here
       x = torch.randn(32, 784)
       output = model(x)
       
       # Update complexity periodically
       model.training_step_callback()
       
       if step % 1000 == 0:
           total_params = sum(p.numel() for p in model.parameters())
           print(f"Step {step}: Total parameters = {total_params:,}")

Performance Tips
----------------

**1. Optimal Batch Sizes**

.. code-block:: python

   import torch
   import panther as pr
   
   # For Tensor Core optimization, use batch sizes that are multiples of 16
   optimal_batch_sizes = [16, 32, 64, 128, 256]
   
   layer = pr.nn.SKLinear(1024, 512, num_terms=8, low_rank=64)
   layer = layer.cuda() if torch.cuda.is_available() else layer
   
   for batch_size in optimal_batch_sizes:
       x = torch.randn(batch_size, 1024)
       x = x.cuda() if torch.cuda.is_available() else x
       
       # This will use optimized kernels
       output = layer(x)
       print(f"Batch size {batch_size}: Output shape {output.shape}")

**2. Mixed Precision Training**

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   from torch.cuda.amp import autocast, GradScaler
   
   # Model with sketched layers
   model = nn.Sequential(
       pr.nn.SKLinear(1024, 512, num_terms=8, low_rank=64),
       nn.ReLU(),
       pr.nn.SKLinear(512, 256, num_terms=4, low_rank=32),
       nn.ReLU(),
       pr.nn.SKLinear(256, 10, num_terms=2, low_rank=16)
   )
   
   if torch.cuda.is_available():
       model = model.cuda()
       
       # Mixed precision training
       scaler = GradScaler()
       optimizer = torch.optim.Adam(model.parameters())
       
       for epoch in range(5):
           for batch_idx in range(100):  # Simulate batches
               x = torch.randn(64, 1024, device='cuda')
               target = torch.randint(0, 10, (64,), device='cuda')
               
               optimizer.zero_grad()
               
               # Forward pass with autocast
               with autocast():
                   output = model(x)
                   loss = nn.CrossEntropyLoss()(output, target)
               
               # Backward pass with gradient scaling
               scaler.scale(loss).backward()
               scaler.step(optimizer)
               scaler.update()
               
               if batch_idx % 50 == 0:
                   print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

This covers the fundamental usage patterns and examples for Panther. For more advanced topics, see the other sections of the documentation.
