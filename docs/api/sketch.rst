Sketching API
=============

The :mod:`panther.sketch` module provides sketching operators for dimensionality reduction and fast linear algebra.

.. currentmodule:: panther.sketch

Sketching Operators
-------------------

.. autofunction:: dense_sketch_operator
   :no-index:

.. autofunction:: sparse_sketch_operator
   :no-index:

.. autofunction:: sketch_tensor
   :no-index:

.. autofunction:: scaled_sign_sketch
   :no-index:

Specific Sketching Methods
--------------------------

.. autofunction:: gaussian_skop
   :no-index:

.. autofunction:: srht
   :no-index:

.. autofunction:: sjlt_skop
   :no-index:

.. autofunction:: count_skop
   :no-index:

Distribution Families
---------------------

.. py:class:: DistributionFamily

   Enumeration of available distribution families for sketching operators.
   
   .. note::
      This class is provided by the pawX C++ extension and may not be available in all environments.

.. py:class:: Axis

   Enumeration of axis specifications for sketching operations.
   
   .. note::
      This class is provided by the pawX C++ extension and may not be available in all environments.

Examples
--------

**Basic Sketching**

.. code-block:: python

   import torch
   import panther as pr
   
   # Create a random matrix to sketch
   A = torch.randn(1000, 500)
   
   # Create Gaussian sketching matrix (100 x 1000)
   S = pr.sketch.dense_sketch_operator(
       m=100,                                    # output dimension
       n=1000,                                   # input dimension
       distribution=pr.sketch.DistributionFamily.Gaussian
   )
   
   # Apply sketching: SA has shape (100, 500)
   SA = S @ A
   print(f"Original: {A.shape}, Sketched: {SA.shape}")

**Sparse Sketching for Memory Efficiency**

.. code-block:: python

   # Sparse sketching matrix (much less memory)
   S_sparse = pr.sketch.sparse_sketch_operator(
       m=200,                              # output dimension
       n=5000,                             # input dimension  
       vec_nnz=3,                          # non-zeros per column
       axis=pr.sketch.Axis.Short
   )
   
   # Apply to large matrix
   A_large = torch.randn(5000, 2000)
   SA_sparse = S_sparse @ A_large
   print(f"Sparse sketched: {SA_sparse.shape}")

**Direct Tensor Sketching**

.. code-block:: python

   # Sketch along different axes
   A = torch.randn(1000, 800)
   
   # Sketch rows (reduce first dimension)
   sketched_rows = pr.sketch.sketch_tensor(
       input=A,
       axis=0,                             # sketch along rows
       new_size=200,                       # new row count
       distribution=pr.sketch.DistributionFamily.Gaussian
   )
   print(f"Row sketched: {sketched_rows[0].shape}")  # (200, 800)
   
   # Sketch columns (reduce second dimension)  
   sketched_cols = pr.sketch.sketch_tensor(
       input=A,
       axis=1,                             # sketch along columns
       new_size=300,                       # new column count
       distribution=pr.sketch.DistributionFamily.Uniform
   )
   print(f"Column sketched: {sketched_cols[0].shape}")  # (1000, 300)

**Subsampled Randomized Hadamard Transform (SRHT)**

.. code-block:: python

   # SRHT is very fast for power-of-2 dimensions
   # SRHT operates on 1D tensors, not matrices
   x = torch.randn(1024)  # dimension must be power of 2
   
   # Apply SRHT to reduce dimensionality
   x_sketched = pr.sketch.srht(
       x=x,                                # input 1D tensor
       m=256                               # output dimension
   )
   
   print(f"SRHT sketched: {x_sketched.shape}")  # (256,)

**Scaled Sign Sketching**

.. code-block:: python

   # Generate scaled sign (Rademacher-like) sketching matrix
   sign_matrix = pr.sketch.scaled_sign_sketch(
       m=64,                               # sketch dimension (rows)
       n=256                               # original dimension (columns)
   )
   
   print(f"Sign matrix: {sign_matrix.shape}")  # (64, 256)
   print(f"Values are ±1/√m: {sign_matrix[0, :5]}")

Distribution Families
---------------------

**Gaussian Sketching**

- **Pros**: Best theoretical guarantees, universally applicable
- **Cons**: Requires more memory (dense matrices)
- **Use when**: Maximum accuracy is needed

.. code-block:: python

   gaussian_sketch = pr.sketch.dense_sketch_operator(
       100, 500, pr.sketch.DistributionFamily.Gaussian
   )

**Uniform Sketching**

- **Pros**: Faster to generate, good performance
- **Cons**: Slightly different properties than Gaussian
- **Use when**: Speed is more important than optimal theoretical constants

.. code-block:: python

   uniform_sketch = pr.sketch.dense_sketch_operator(
       100, 500, pr.sketch.DistributionFamily.Uniform
   )

**Sparse Sketching**

- **Pros**: Very memory efficient, can handle huge matrices
- **Cons**: May require larger sketch size for same accuracy
- **Use when**: Working with very large matrices

.. code-block:: python

   sparse_sketch = pr.sketch.sparse_sketch_operator(
       100, 500, vec_nnz=3, axis=pr.sketch.Axis.Rows
   )

**SRHT (Subsampled Randomized Hadamard Transform)**

- **Pros**: Extremely fast, good for structured matrices
- **Cons**: Requires power-of-2 dimensions, operates on 1D tensors only
- **Use when**: Input dimension is power of 2

.. code-block:: python

   x = torch.randn(1024)  # 1024 = 2^10
   x_srht = pr.sketch.srht(x, 128)

Performance Comparison
----------------------

Sketching matrix generation time (for 1000×5000 matrix):

.. list-table::
   :header-rows: 1
   
   * - Method
     - Time (ms)
     - Memory (MB)
     - Quality
   * - Gaussian
     - 45.2
     - 19.1
     - Excellent
   * - Uniform
     - 15.2
     - 19.1
     - Very Good
   * - Sparse (nnz=3)
     - 8.4
     - 0.6
     - Good
   * - SRHT
     - 2.1
     - 0.1
     - Very Good

GPU Acceleration
----------------

All sketching operations support GPU:

.. code-block:: python

   device = torch.device('cuda')
   
   # Create sketching matrix on GPU
   S = pr.sketch.dense_sketch_operator(
       200, 1000, 
       pr.sketch.DistributionFamily.Gaussian,
       device=device
   )
   
   # Apply to GPU tensor
   A = torch.randn(1000, 800, device=device)
   SA = S @ A  # Computed on GPU
   
   print(f"Result device: {SA.device}")

Best Practices
--------------

**1. Choose sketch size appropriately**

.. code-block:: python

   # Rule of thumb: sketch_size ≈ 2-4 × target_rank
   import numpy as np
   target_rank = 50
   sketch_size = 4 * target_rank  # 200
   
   # For ε-accuracy: sketch_size ≥ k + log(1/ε) 
   epsilon = 0.01
   sketch_size_theory = target_rank + int(np.log(1/epsilon))

**2. Use sparse sketching for very large matrices**

.. code-block:: python

   # For matrices larger than available memory
   very_large_sketch = pr.sketch.sparse_sketch_operator(
       m=1000,
       n=1000000,    # 1M dimensions
       vec_nnz=4,    # Only 4 non-zeros per column
       axis=pr.sketch.Axis.Long
   )