Custom Sketching Methods
========================

This guide covers the available sketching operators in Panther and how to use them effectively.

Available Sketching Operators
------------------------------

Panther provides several built-in sketching operators through the ``panther.sketch`` module:

* **scaled_sign_sketch**: Scaled sign (Rademacher) sketching matrices
* **dense_sketch_operator**: Dense random matrices from various distributions
* **sparse_sketch_operator**: Sparse sketching matrices for efficiency
* **gaussian_skop**: Gaussian random projection
* **sjlt_skop**: Sparse Johnson-Lindenstrauss Transform
* **count_skop**: Count sketch operator
* **srht**: Subsampled Randomized Hadamard Transform

Using Built-in Sketching Operators
-----------------------------------

**Basic Sketching Example**

.. code-block:: python

   import torch
   from panther.sketch import scaled_sign_sketch, dense_sketch_operator
   from panther.sketch import DistributionFamily
   
   # Create a scaled sign sketch matrix
   m, n = 100, 500  # sketch to 100 dimensions from 500
   S = scaled_sign_sketch(m, n)
   print(f"Sketch matrix shape: {S.shape}")
   
   # Apply sketching to a matrix
   A = torch.randn(1000, 500)  # Original matrix
   A_sketched = A @ S.T  # Sketched version
   print(f"Sketched matrix shape: {A_sketched.shape}")

**Using Different Distributions**

.. code-block:: python

   from panther.sketch import dense_sketch_operator, DistributionFamily
   
   # Gaussian sketching
   S_gaussian = dense_sketch_operator(
       m=100, n=500, 
       distribution=DistributionFamily.Gaussian
   )
   
   # Apply to data
   X = torch.randn(1000, 500)
   X_sketched = X @ S_gaussian.T

**Tensor Sketching Along Specific Axes**

.. code-block:: python

   from panther.sketch import sketch_tensor, DistributionFamily
   
   # Sketch a 3D tensor along axis 1
   tensor = torch.randn(32, 512, 128)  # (batch, features, time)
   
   # Reduce feature dimension from 512 to 128
   sketched, sketch_matrix = sketch_tensor(
       input=tensor,
       axis=1,
       new_size=128,
       distribution=DistributionFamily.Gaussian
   )
   
   print(f"Original shape: {tensor.shape}")
   print(f"Sketched shape: {sketched.shape}")
   print(f"Sketch matrix shape: {sketch_matrix.shape}")

Using Sketching Operators with Panther Layers
----------------------------------------------

Panther's built-in layers (SKLinear, SKConv2d) use optimized sketching internally. 
For advanced use cases, you can use the sketching operators directly with your custom layers.

.. code-block:: python

   class SketchedAttention(nn.Module):
       """Attention mechanism with sketched key-value matrices."""
       
       def __init__(self, d_model, num_heads, sketch_ratio=0.5):
           super().__init__()
           self.d_model = d_model
           self.num_heads = num_heads
           self.d_head = d_model // num_heads
           self.sketch_dim = int(d_model * sketch_ratio)
           
           # Sketching for keys and values
           self.key_sketch = HadamardSketch(d_model, self.sketch_dim)
           self.value_sketch = HadamardSketch(d_model, self.sketch_dim)
           
           # Projection layers
           self.query_proj = nn.Linear(d_model, d_model)
           self.key_proj = nn.Linear(self.sketch_dim, d_model)
           self.value_proj = nn.Linear(self.sketch_dim, d_model)
           self.output_proj = nn.Linear(d_model, d_model)
       
       def forward(self, query, key, value, mask=None):
           batch_size, seq_len = query.shape[:2]
           
           # Sketch keys and values
           key_sketched = self.key_sketch(key)
           value_sketched = self.value_sketch(value)
           
           # Project queries, keys, values
           Q = self.query_proj(query)
           K = self.key_proj(key_sketched)
           V = self.value_proj(value_sketched)
           
           # Reshape for multi-head attention
           Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
           K = K.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
           V = V.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
           
           # Compute attention
           scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_head))
           
           if mask is not None:
               scores.masked_fill_(mask == 0, -1e9)
           
           attention_weights = torch.softmax(scores, dim=-1)
           context = torch.matmul(attention_weights, V)
           
           # Reshape and project output
**Sparse Sketching**

.. code-block:: python

   from panther.sketch import sparse_sketch_operator, DistributionFamily
   
   # Create sparse sketching matrix
   m, n = 100, 1000
   S_sparse = sparse_sketch_operator(m, n, DistributionFamily.Gaussian)
   
   # Apply to data - more memory efficient for large n
   X = torch.randn(500, 1000)
   X_sketched = X @ S_sparse.T
   print(f"Sketched shape: {X_sketched.shape}")

**Count Sketch**

.. code-block:: python

   from panther.sketch import count_skop
   
   # Create count sketch operator
   m, n = 128, 1024
   S_count = count_skop(m, n)
   
   # Apply sketching
   X = torch.randn(256, 1024)
   X_sketched = X @ S_count.T

.. code-block:: python

   from panther.sketch import sparse_sketch_operator, DistributionFamily
   
   # Create sparse sketch (more memory efficient)
   S_sparse = sparse_sketch_operator(
       m=100, n=5000,
       distribution=DistributionFamily.Gaussian
   )
   
   # Use with large matrices
   X = torch.randn(1000, 5000)
   X_sketched = X @ S_sparse.T

**Subsampled Randomized Hadamard Transform**

.. code-block:: python

   from panther.sketch import srht
   
   # SRHT is particularly efficient for powers of 2
   input_tensor = torch.randn(1000, 512)  # 512 is power of 2
   
   # Apply SRHT sketching
   sketched = srht(input_tensor, new_size=128)
   print(f"Sketched shape: {sketched.shape}")

Integration with Neural Networks
---------------------------------

Panther's sketched layers already integrate these sketching operators internally. Here's how they're used:

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   
   class SketchedNetwork(nn.Module):
       """Example network using Panther's sketched layers."""
       
       def __init__(self):
           super().__init__()
           # Sketched layers use the built-in sketching operators
           self.layer1 = pr.nn.SKLinear(784, 512, num_terms=4, low_rank=64)
           self.layer2 = pr.nn.SKLinear(512, 256, num_terms=2, low_rank=32)
           self.relu = nn.ReLU()
       
       def forward(self, x):
           x = self.relu(self.layer1(x))
           x = self.layer2(x)
           return x
   
   model = SketchedNetwork()
   x = torch.randn(32, 784)
   output = model(x)

Best Practices
--------------

**Choosing the Right Sketching Method**

* **dense_sketch_operator**: General purpose, good for most applications
* **sparse_sketch_operator**: Better for very large dimensions
* **srht**: Fastest when dimensions are powers of 2
* **scaled_sign_sketch**: Simple and effective for many use cases

**Performance Considerations**

.. code-block:: python

   import time
   from panther.sketch import scaled_sign_sketch, srht, DistributionFamily
   from panther.sketch import dense_sketch_operator
   
   # Compare sketching methods
   X = torch.randn(10000, 1024, device='cuda' if torch.cuda.is_available() else 'cpu')
   
   # Time different methods
   methods = {
       'scaled_sign': lambda: X @ scaled_sign_sketch(256, 1024, device=X.device).T,
       'dense_gaussian': lambda: X @ dense_sketch_operator(
           256, 1024, DistributionFamily.Gaussian, device=X.device).T,
   }
   
   for name, method in methods.items():
       start = time.time()
       result = method()
       elapsed = time.time() - start
       print(f"{name}: {elapsed:.4f} seconds")

See Also
--------

* :doc:`../api/sketch` - Complete sketching API reference
* :doc:`../tutorials/performance_optimization` - Performance optimization techniques
* :doc:`../examples/basic_usage` - Basic usage examples
