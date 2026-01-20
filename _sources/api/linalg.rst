Linear Algebra API
==================

The :mod:`panther.linalg` module provides randomized linear algebra operations for efficient matrix decompositions.


Core Functions
--------------

.. automodule:: panther.linalg
   :members: cqrrpt, randomized_svd
   :undoc-members:
   :show-inheritance:

Distribution Families
---------------------

.. autoclass:: panther.linalg.DistributionFamily

   Enumeration of available distribution families for linear algebra operations.
   
   .. note::
      This class is provided by the pawX C++ extension and may not be available in all environments.

Examples
--------

**Randomized QR Decomposition with Column Pivoting**

.. code-block:: python

   import torch
   import panther as pr
   
   # Create a matrix
   A = torch.randn(1000, 500)
   
   # Perform CQRRPT decomposition  
   Q, R, J = pr.linalg.cqrrpt(A, gamma=1.5)
   
   # Q is orthogonal, R is upper triangular
   print(f"Q shape: {Q.shape}")  # (1000, 500)
   print(f"R shape: {R.shape}")  # (500, 500)  
   print(f"J shape: {J.shape}")  # (500,)
   
   # Verify decomposition: A[:, J] ≈ Q @ R
   reconstruction_error = torch.norm(A[:, J] - Q @ R)
   print(f"Reconstruction error: {reconstruction_error.item()}")

**Randomized SVD**

.. code-block:: python

   import torch
   import panther as pr
   
   # Create a large matrix
   A = torch.randn(2000, 1500)
   
   # Compute top 100 singular values/vectors
   U, S, V = pr.linalg.randomized_svd(A, k=100, tol=1e-6)
   
   print(f"U shape: {U.shape}")  # (2000, 100)
   print(f"S shape: {S.shape}")  # (100,)
   print(f"V shape: {V.shape}")  # (1500, 100)
   
   # Reconstruct approximation
   A_approx = U @ torch.diag(S) @ V.T
   approximation_error = torch.norm(A - A_approx)
   print(f"Approximation error: {approximation_error.item()}")

GPU Support
-----------

All linear algebra operations support GPU acceleration:

.. code-block:: python

   import torch
   import panther as pr
   
   device = torch.device('cuda')
   A = torch.randn(5000, 3000, device=device)
   
   # GPU computation
   Q, R, J = pr.linalg.cqrrpt(A)
   
   # Results are on the same device
   print(f"Q device: {Q.device}")  # cuda:0

Performance Notes
-----------------

* **Matrix Size**: Randomized algorithms are most beneficial for large matrices (> 1000x1000)
* **Rank**: For randomized SVD, choose k based on the desired accuracy vs. speed tradeoff
* **Tolerance**: Lower tolerance values in randomized SVD provide higher accuracy but require more computation
* **GPU Memory**: Large matrices may require significant GPU memory. Monitor usage with ``torch.cuda.memory_allocated()``

Theoretical Background
----------------------

The randomized algorithms implemented in Panther are based on:

* **CQRRPT**: Randomized QR decomposition with column pivoting for numerical stability
* **Randomized SVD**: Fast approximate SVD using random projections
* **Sketching**: Dimensionality reduction techniques for large-scale linear algebra

These methods provide:

* **Speed**: Much faster than deterministic algorithms for large matrices
* **Memory Efficiency**: Lower memory footprint through sketching
* **Accuracy**: Theoretical guarantees on approximation quality
* **Scalability**: Handle matrices that don't fit in memory
