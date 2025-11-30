Custom Sketching Implementation
===============================

This tutorial shows how to use Panther's sketching operators for custom applications.

Understanding Sketching Fundamentals
-------------------------------------

**Mathematical Foundation**

Sketching reduces the dimensionality of data while preserving important properties. The key insight is that random projections can preserve distances and inner products with high probability.

For a matrix :math:`A \\in \\mathbb{R}^{m \\times n}`, a sketch :math:`S \\in \\mathbb{R}^{k \\times m}` produces:

.. math::

   \\widetilde{A} = SA

where :math:`k \\ll m` and :math:`\\widetilde{A}` preserves key properties of :math:`A`.

**Using Panther's Built-in Sketching Operators**

Panther provides several sketching operators through the `panther.sketch` module:

.. code-block:: python

   import torch
   import panther as pr
   
   # Create a matrix to sketch
   A = torch.randn(1000, 500)
   
   # Gaussian sketching operator
   S_gaussian = pr.sketch.gaussian_skop(
       m=100,  # output dimension
       n=1000  # input dimension
   )
   
   # Apply sketch
   SA = S_gaussian @ A
   print(f"Original: {A.shape}, Sketched: {SA.shape}")
   
   # Sparse sketching (more memory efficient)
   S_sparse = pr.sketch.sparse_sketch_operator(
       m=100,
       n=1000,
       vec_nnz=3,  # non-zeros per column
       axis=pr.sketch.Axis.Rows
   )
   
   # SRHT (Subsampled Randomized Hadamard Transform)
   # Note: SRHT operates on 1D tensors, not matrices
   x = torch.randn(1024)  # must be power of 2
   x_srht = pr.sketch.srht(
       x=x,
       m=100
   )
   
   # Count sketch
   S_count = pr.sketch.count_skop(
       m=100,
       n=1000
   )

Available Sketching Methods
----------------------------

**1. Gaussian Sketching**

.. code-block:: python

   import torch
   import panther as pr
   
   # Dense Gaussian sketching matrix
   m, n = 200, 1000
   S = pr.sketch.gaussian_skop(m, n)
   
   # Apply to data
   X = torch.randn(1000, 500)
   X_sketched = S @ X
   
   print(f"Compression ratio: {n/m:.1f}x")

**2. Sparse Sketching**

.. code-block:: python

   # Sparse sketching for memory efficiency
   S_sparse = pr.sketch.sparse_sketch_operator(
       m=200,
       n=5000,
       vec_nnz=3,
       axis=pr.sketch.Axis.Rows
   )
   
   # Much less memory than dense sketching
   X_large = torch.randn(5000, 2000)
   X_sparse_sketched = S_sparse @ X_large

**3. SRHT (Fast Walsh-Hadamard Transform)**

.. code-block:: python

   # SRHT is very fast for power-of-2 dimensions
   # SRHT operates on 1D tensors
   n = 1024  # power of 2
   m = 128
   x = torch.randn(n)
   x_srht_sketched = pr.sketch.srht(x, m)
   print(f"Sketched from {n} to {m} dimensions")

**4. Count Sketch**

.. code-block:: python

   # Count sketch using feature hashing
   S_count = pr.sketch.count_skop(m=100, n=1000)
   X = torch.randn(1000, 400)
   X_count_sketched = S_count @ X

**5. SJLT (Sparse Johnson-Lindenstrauss Transform)**

.. code-block:: python

   # SJLT with sparse random projections
   S_sjlt = pr.sketch.sjlt_skop(m=100, n=1000)
   X = torch.randn(1000, 300)
   X_sjlt_sketched = S_sjlt @ X

**Using Sketching in Custom Neural Network Layers**

Creating custom layers that leverage Panther's sketching operators:

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   
   class SimpleSketchedLayer(nn.Module):
       """Custom layer using sketching for dimensionality reduction."""
       
       def __init__(self, input_dim, sketch_dim, output_dim):
           super().__init__()
           self.input_dim = input_dim
           self.sketch_dim = sketch_dim
           
           # Create sketching operator
           self.register_buffer('sketch_matrix', 
               pr.sketch.gaussian_skop(sketch_dim, input_dim))
           
           # Linear transformation after sketching
           self.linear = nn.Linear(sketch_dim, output_dim)
       
       def forward(self, x):
           # Apply sketching
           x_sketched = x @ self.sketch_matrix.T
           # Apply linear transformation
           return self.linear(x_sketched)
   
   # Example usage
   layer = SimpleSketchedLayer(1000, 200, 100)
   x = torch.randn(32, 1000)
   output = layer(x)
   print(f"Input: {x.shape}, Output: {output.shape}")

**Combining Multiple Sketching Methods**

.. code-block:: python

   class MultiSketchLayer(nn.Module):
       """Layer that combines different sketching operators."""
       
       def __init__(self, input_dim, sketch_dim, output_dim):
           super().__init__()
           
           # Different sketching operators
           self.register_buffer('gaussian_sketch',
               pr.sketch.gaussian_skop(sketch_dim, input_dim))
           self.register_buffer('sparse_sketch',
               pr.sketch.sparse_sketch_operator(
                   sketch_dim, input_dim, vec_nnz=3,
                   axis=pr.sketch.Axis.Rows))
           
           # Combine sketches
           self.fusion = nn.Linear(sketch_dim * 2, sketch_dim)
           self.output = nn.Linear(sketch_dim, output_dim)
       
       def forward(self, x):
           # Apply both sketching methods
           gaussian_result = x @ self.gaussian_sketch.T
           sparse_result = x @ self.sparse_sketch.T
           
           # Concatenate and fuse
           combined = torch.cat([gaussian_result, sparse_result], dim=-1)
           fused = self.fusion(combined)
           
           return self.output(fused)

Practical Applications
----------------------

**Matrix Compression**

.. code-block:: python

   import torch
   import panther as pr
   
   # Large weight matrix
   W = torch.randn(2000, 5000)
   
   # Compress using sketching
   sketch_dim = 500
   S = pr.sketch.gaussian_skop(sketch_dim, 2000)
   
   # Compressed representation
   W_sketched = S @ W
   
   print(f"Original size: {W.numel() * 4 / 1024**2:.2f} MB")
   print(f"Compressed size: {W_sketched.numel() * 4 / 1024**2:.2f} MB")
   print(f"Compression ratio: {W.numel() / W_sketched.numel():.2f}x")

**Fast Approximate Matrix Multiplication**

.. code-block:: python

   # Two large matrices
   A = torch.randn(10000, 5000)
   B = torch.randn(5000, 8000)
   
   # Sketch first matrix
   sketch_size = 1000
   S = pr.sketch.gaussian_skop(sketch_size, 10000)
   
   A_sketched = S @ A
   
   # Approximate product
   result_approx = A_sketched @ B
   
   # Note: result_approx is sketch_size x 8000 instead of 10000 x 8000
   print(f"Approximate result shape: {result_approx.shape}")

Comparing Sketching Methods
----------------------------

**Performance Comparison**

.. code-block:: python

   import time
   import torch
   import panther as pr
   
   def benchmark_sketch_methods(n=10000, m=2000, k=500):
       """Compare different sketching methods."""
       
       x = torch.randn(n)
       
       methods = {
           'Gaussian': lambda: pr.sketch.gaussian_skop(m, n),
           'Sparse': lambda: pr.sketch.sparse_sketch_operator(
               m, n, vec_nnz=3, axis=pr.sketch.Axis.Rows),
           'Count': lambda: pr.sketch.count_skop(m, n),
           'SJLT': lambda: pr.sketch.sjlt_skop(m, n)
       }
       
       results = {}
       for name, create_sketch in methods.items():
           S = create_sketch()
           
           # Warmup
           for _ in range(5):
               _ = S @ x
           
           # Timing
           start = time.time()
           for _ in range(100):
               _ = S @ x
           elapsed = time.time() - start
           
           results[name] = elapsed / 100
           print(f"{name}: {results[name]*1000:.3f} ms")
       
       return results
   
   # Run benchmark
   benchmark_sketch_methods()

See Also
--------

* :doc:`../api/sketch` - Complete sketching API reference
* :doc:`neural_networks` - Using sketched layers in neural networks
* :doc:`performance_optimization` - Optimization techniques
