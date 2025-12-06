Matrix Decompositions
=====================

This tutorial covers Panther's randomized matrix decomposition algorithms: CQRRPT and RSVD.

Introduction to Randomized Matrix Decomposition**Mathematical Background**

RSVD computes a low-rank approximation of the SVD: :math:`A \approx U \Sigma V^T` where:

* :math:`U` has orthonormal columns
* :math:`\Sigma` is diagonal with singular values
* :math:`V` has orthonormal columns

The algorithm uses random sampling to efficiently compute the dominant singular vectors.

**Internal Algorithm Details**

RSVD's implementation uses a sophisticated blocked approach:

1. **Power Iteration Sketching**: Creates initial random sketch :math:`\Omega \in \mathbb{R}^{n \times k}` and applies alternating power iterations:
   
   .. math::
      \begin{align}
      Y^{(0)} &= A \Omega \\
      Y^{(1)} &= A^T Y^{(0)} \\
      Y^{(2)} &= A Y^{(1)} \\
      &\vdots
      \end{align}
   
   with periodic orthonormalization for numerical stability

2. **Range Finding**: Computes orthonormal basis :math:`Q` spanning the approximate range of :math:`A`:
   
   .. math::
      Y = A \Omega \quad \text{and} \quad Q, R = \text{QR}(Y)

3. **Blocked QB Decomposition**: Uses adaptive blocking to build factorization :math:`A \approx Q B`:
   
   - Processes matrix in blocks of size 64 (or smaller near target rank)
   - For each block, finds range of current residual: :math:`A_{res} - Q_i B_i`
   - Re-orthogonalizes against existing :math:`Q` to maintain orthogonality
   - Updates residual: :math:`A_{res} \leftarrow A_{res} - Q_i B_i`

4. **Adaptive Termination**: Monitors approximation error using matrix norms:
   
   .. math::
      \text{error} = \sqrt{|\|A\|_F^2 - \|B\|_F^2|} \cdot \frac{\sqrt{\|A\|_F^2 + \|B\|_F^2}}{\|A\|_F}
   
   Terminates when error stops decreasing or falls below tolerance

5. **Final SVD**: Performs exact SVD on the much smaller matrix :math:`B`:
   
   .. math::
      B = \tilde{U} \Sigma V^T
   
   Then reconstructs: :math:`U = Q \tilde{U}`

This blocked approach is more memory-efficient and numerically stable than standard randomized SVD, especially for matrices where the target rank is a significant fraction of the matrix dimensions.-------------------------------------------

Traditional matrix decompositions like QR and SVD have :math:`O(n^3)` complexity for :math:`n \\times n` matrices. Randomized methods can achieve near-optimal results with :math:`O(n^2)` complexity, making them suitable for large-scale problems.

Panther provides two main randomized decompositions:

* **CQRRPT**: CholeskyQR with Randomized column Pivoting for tall matrices
* **RSVD**: Randomized Singular Value Decomposition

CholeskyQR with Randomized Pivoting (CQRRPT)
----------------------------------------------

**Mathematical Background**

CQRRPT computes a QR decomposition :math:`A = QR` where:

* :math:`Q` has orthonormal columns
* :math:`R` is upper triangular
* Column pivoting improves numerical stability

The algorithm uses random sampling and tournament pivoting to reduce communication costs.

**Internal Algorithm Details**

CQRRPT's implementation follows this detailed process:

1. **Random Sketching Phase**: First computes :math:`M_{sk} = S \cdot M` where :math:`S` is a :math:`d \times m` sketching matrix with :math:`d = \lceil \gamma \cdot n \rceil` (compression ratio)

2. **Pivoted QR on Sketch**: Performs QR with column pivoting on the much smaller sketched matrix using LAPACK's ``GEQP3`` routine:
   
   .. math::
      M_{sk} \Pi = Q_{sk} R_{sk}
   
   where :math:`\Pi` is the permutation matrix encoding column pivots

3. **Rank Estimation**: Estimates numerical rank by examining diagonal elements of :math:`R_{sk}`:
   
   .. math::
      k = |\{i : |R_{sk}(i,i)| > \epsilon\}|
   
   with :math:`\epsilon = 10^{-6}` for float32, :math:`10^{-12}` for float64

4. **Column Selection**: Selects the :math:`k` most important columns: :math:`M_k = M[:, \Pi[:k]]`

5. **Triangular Solve**: Solves :math:`R_{sk}[:k,:k] \cdot M_{pre} = M_k^T` to get preliminary factorization

6. **Cholesky Refinement**: Computes :math:`G = M_{pre}^T M_{pre}` and its Cholesky factorization :math:`G = L L^T`

7. **Adaptive Rank Refinement**: Monitors condition number via Cholesky diagonal:
   
   .. math::
      \text{condition} = \frac{\max(|L_{ii}|)}{\min(|L_{ii}|)} \leq \sqrt{\frac{\epsilon}{u}}
   
   where :math:`u` is machine epsilon, stopping when condition becomes too large

8. **Final Factorization**: Computes final :math:`Q_k` and :math:`R_k` using the refined rank

**Basic Usage**

.. code-block:: python

   import torch
   import panther as pr
   
   # Create a test matrix
   m, n = 1000, 500
   A = torch.randn(m, n, dtype=torch.float64)
   
   # Standard CQRRPT
   Q, R, P = pr.linalg.cqrrpt(A)
   
   print(f"Original matrix: {A.shape}")
   print(f"Q matrix: {Q.shape}")
   print(f"R matrix: {R.shape}")
   print(f"Permutation: {P.shape}")
   
   # Verify decomposition: A[:, P] ≈ Q @ R
   reconstruction_error = torch.norm(A[:, P] - Q @ R)
   print(f"Reconstruction error: {reconstruction_error:.2e}")

**Customizing CQRRPT Parameters**

.. code-block:: python

   # CQRRPT with custom parameters
   Q, R, P = pr.linalg.cqrrpt(
       A,
       gamma=1.5,                                    # Oversampling parameter
       F=pr.linalg.DistributionFamily.Gaussian       # Distribution family
   )
   
   # Check orthogonality of Q
   orthogonality_error = torch.norm(Q.T @ Q - torch.eye(Q.shape[1], dtype=A.dtype))
   print(f"Q orthogonality error: {orthogonality_error:.2e}")

**Performance Comparison**

.. code-block:: python

    import time
    import torch
    import panther as pr

    def compare_qr_methods(matrix_sizes):
        """Compare CQRRPT with standard QR."""
        results = {}

        for m, n in matrix_sizes:
            print(f"\nTesting {m}x{n} matrix:")
            A = torch.randn(m, n, dtype=torch.float64)

            # Standard PyTorch QR
            start_time = time.time()
            Q_torch, R_torch = torch.linalg.qr(A)
            torch_time = time.time() - start_time

            # Panther CQRRPT
            start_time = time.time()
            Q_cqrrpt, R_cqrrpt, P_cqrrpt = pr.linalg.cqrrpt(A)
            cqrrpt_time = time.time() - start_time

            # Compare accuracy
            torch_error = torch.norm(A - Q_torch @ R_torch)
            cqrrpt_error = torch.norm(A[:, P_cqrrpt] - Q_cqrrpt @ R_cqrrpt)

            # Store results
            results[(m, n)] = {
                'torch_time': torch_time,
                'cqrrpt_time': cqrrpt_time,
                'speedup': torch_time / cqrrpt_time,
                'torch_error': torch_error.item(),
                'cqrrpt_error': cqrrpt_error.item()
            }

            # Print results
            print(f"  PyTorch QR: {torch_time:.4f}s, error: {torch_error:.2e}")
            print(f"  CQRRPT:    {cqrrpt_time:.4f}s, error: {cqrrpt_error:.2e}")
            print(f"  Speedup:   {torch_time / cqrrpt_time:.2f}x")

        return results


    # Test different matrix sizes
    sizes = [(500, 250), (1000, 500), (2000, 1000)]
    comparison_results = compare_qr_methods(sizes)


**Using CQRRPT for Least Squares**

.. code-block:: python

   def solve_least_squares_cqrrpt(A, b):
       """Solve least squares problem using CQRRPT."""
       
       # Compute CQRRPT decomposition
       Q, R, P = pr.linalg.cqrrpt(A)
       
       # Solve R x = Q^T b for the permuted variables
       QtB = Q.T @ b
       QtB = QtB.unsqueeze(1)
       x_permuted = torch.linalg.solve_triangular(R, QtB, upper=True)
       
       # Unpermute the solution
       x = torch.zeros_like(x_permuted)
       x[P] = x_permuted
       
       return x
   
   # Example: Solve linear regression problem
   n_samples, n_features = 1000, 200
   A = torch.randn(n_samples, n_features, dtype=torch.float64)
   x_true = torch.randn(n_features, dtype=torch.float64)
   b = A @ x_true + 0.01 * torch.randn(n_samples, dtype=torch.float64)
   
   # Solve using CQRRPT
   x_estimated = solve_least_squares_cqrrpt(A, b)
   
   # Compare with true solution
   solution_error = torch.norm(x_estimated - x_true)
   print(f"Solution error: {solution_error:.4f}")
   
   # Verify residual
   residual = torch.norm(A @ x_estimated - b)
   print(f"Residual norm: {residual:.4f}")

Randomized SVD (RSVD)
---------------------

**Mathematical Background**

RSVD computes a low-rank approximation of the SVD: :math:`A \approx U \Sigma V^T` where:

* :math:`U` has orthonormal columns
* :math:`\Sigma` is diagonal with singular values
* :math:`V` has orthonormal columns

The algorithm uses random sampling to efficiently compute the dominant singular vectors.

**Basic RSVD Usage**

.. code-block:: python

   # Create a low-rank matrix plus noise
   m, n, rank = 1000, 800, 50
   
   # Generate low-rank matrix
   U_true = torch.randn(m, rank)
   V_true = torch.randn(n, rank)
   sigma_true = torch.logspace(0, -2, rank)  # Decreasing singular values
   
   A_lowrank = U_true @ torch.diag(sigma_true) @ V_true.T
   noise = 0.01 * torch.randn(m, n)
   A = A_lowrank + noise
   
   print(f"Matrix shape: {A.shape}")
   print(f"True rank: {rank}")
   
   # Compute RSVD
   target_rank = 60  # Slightly higher than true rank
   U, S, V = pr.linalg.randomized_svd(A, k=target_rank, tol=1e-5)
   
   # Note: RSVD uses blocked QB internally:
   # - Processes in blocks of min(64, target_rank) 
   # - Adaptively determines actual rank based on tolerance
   # - May return fewer than target_rank components if early termination occurs
   
   print(f"\\nRSVD results:")
   print(f"U shape: {U.shape}")
   print(f"S shape: {S.shape}")
   print(f"V shape: {V.shape}")
   
   # Reconstruct and check error
   A_reconstructed = U @ torch.diag(S) @ V.T
   reconstruction_error = torch.norm(A - A_reconstructed)
   frobenius_norm = torch.norm(A)
   
   print(f"\\nReconstruction error: {reconstruction_error:.4f}")
   print(f"Relative error: {reconstruction_error/frobenius_norm:.6f}")

**Advanced RSVD Parameters**

.. code-block:: python

   # RSVD with custom tolerance for higher accuracy
   U, S, V = pr.linalg.randomized_svd(A, k=50, tol=1e-8)
   
   # Analyze singular values
   print("Top 10 singular values:")
   for i in range(min(10, len(S))):
       print(f"  σ_{i+1}: {S[i]:.6f}")
   
   # Compare with exact SVD (for smaller matrices)
   if min(A.shape) <= 500:
       U_exact, S_exact, V_exact = torch.linalg.svd(A, full_matrices=False)
       
       # Compare singular values
       rank_to_compare = min(len(S), len(S_exact))
       sv_error = torch.norm(S[:rank_to_compare] - S_exact[:rank_to_compare])
       print(f"\\nSingular value error: {sv_error:.2e}")

**Adaptive Rank Selection**

.. code-block:: python

   def adaptive_rsvd(A, energy_threshold=0.99, max_rank=None):
       """Automatically determine rank based on energy threshold."""
       
       max_rank = max_rank or min(A.shape) // 2
       
       # Compute RSVD with generous rank
       U, S, V = pr.linalg.randomized_svd(A, k=max_rank, tol=1e-6)
       
       # Find rank that captures desired energy
       total_energy = torch.sum(S**2)
       cumulative_energy = torch.cumsum(S**2, dim=0)
       energy_ratio = cumulative_energy / total_energy
       
       # Find first index where energy threshold is exceeded
       rank_needed = torch.argmax((energy_ratio >= energy_threshold).float()) + 1
       
       print(f"Adaptive rank selection:")
       print(f"  Energy threshold: {energy_threshold}")
       print(f"  Selected rank: {rank_needed}")
       print(f"  Energy captured: {energy_ratio[rank_needed-1]:.6f}")
       
       # Return truncated decomposition
       return U[:, :rank_needed], S[:rank_needed], V[:, :rank_needed], rank_needed
   
   # Example usage
   U_adaptive, S_adaptive, V_adaptive, optimal_rank = adaptive_rsvd(A, energy_threshold=0.95)
   
   A_adaptive = U_adaptive @ torch.diag(S_adaptive) @ V_adaptive.T
   adaptive_error = torch.norm(A - A_adaptive)
   print(f"Adaptive reconstruction error: {adaptive_error:.4f}")

**RSVD for Principal Component Analysis**

.. code-block:: python

   def rsvd_pca(X, n_components, center_data=True):
    """Perform PCA using randomized SVD."""

    # Center data
    if center_data:
        mean = torch.mean(X, dim=0)
        X_centered = X - mean
    else:
        mean = torch.zeros(X.shape[1])
        X_centered = X

    U, S, V = pr.linalg.randomized_svd(X_centered, k=n_components, tol=1e-6)

    components = V  
    explained_variance = S**2 / (X.shape[0] - 1)
    total_variance = torch.sum(torch.var(X_centered, dim=0))
    explained_variance_ratio = explained_variance / total_variance

    X_transformed = X_centered @ components

    return {
        'components': components,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
        'singular_values': S,
        'mean': mean,
        'X_transformed': X_transformed
    }
   
   # Example: PCA on synthetic data
   n_samples, n_features = 1000, 200
   
   # Generate data with known structure
   latent_dim = 10
   latent_factors = torch.randn(n_samples, latent_dim)
   loading_matrix = torch.randn(n_features, latent_dim)
   X = latent_factors @ loading_matrix.T + 0.1 * torch.randn(n_samples, n_features)
   
   # Perform RSVD-PCA
   pca_result = rsvd_pca(X, n_components=15)
   
   print(f"Data shape: {X.shape}")
   print(f"Transformed shape: {pca_result['X_transformed'].shape}")
   print(f"Cumulative explained variance: {torch.cumsum(pca_result['explained_variance_ratio'], 0)[:5]}")

**Matrix Completion with RSVD**

.. code-block:: python

   def matrix_completion_rsvd(A_observed, mask, rank, n_iterations=10):
       """Simple matrix completion using iterative RSVD."""
       
       # Initialize missing entries with column means
       A_filled = A_observed.clone()
       for j in range(A_filled.shape[1]):
           col_mask = mask[:, j]
           if col_mask.sum() > 0:
               col_mean = A_observed[col_mask, j].mean()
               A_filled[~col_mask, j] = col_mean
       
       for iteration in range(n_iterations):
           # Compute low-rank approximation
           U, S, V = pr.linalg.randomized_svd(A_filled, k=rank, tol=1e-6)
           A_lowrank = U @ torch.diag(S) @ V.T
           
           # Keep observed entries, update missing ones
           A_filled = torch.where(mask, A_observed, A_lowrank)
           
           # Compute objective (only on observed entries)
           if iteration % 2 == 0:
               objective = torch.norm((A_observed - A_lowrank)[mask])
               print(f"Iteration {iteration}: Objective = {objective:.6f}")
       
       return A_filled
   
   # Example: Matrix completion
   m, n, true_rank = 200, 150, 10
   
   # Generate low-rank matrix
   U_true = torch.randn(m, true_rank)
   V_true = torch.randn(n, true_rank)
   A_complete = U_true @ V_true.T
   
   # Create random mask (observe 60% of entries)
   mask = torch.rand(m, n) < 0.6
   A_observed = torch.where(mask, A_complete, torch.tensor(0.0))
   
   print(f"Matrix completion problem:")
   print(f"  Matrix size: {A_complete.shape}")
   print(f"  True rank: {true_rank}")
   print(f"  Observed entries: {mask.sum()}/{mask.numel()} ({100*mask.float().mean():.1f}%)")
   
   # Perform matrix completion
   A_completed = matrix_completion_rsvd(A_observed, mask, rank=15, n_iterations=20)
   
   # Evaluate completion quality
   test_mask = ~mask
   completion_error = torch.norm((A_complete - A_completed)[test_mask])
   test_norm = torch.norm(A_complete[test_mask])
   relative_error = completion_error / test_norm
   
   print(f"  Completion results:")
   print(f"  Test error: {completion_error:.4f}")
   print(f"  Relative test error: {relative_error:.6f}")

GPU Acceleration
----------------

**Using CUDA for Large Matrices**

.. code-block:: python

   if torch.cuda.is_available():
       device = torch.device('cuda')
       
       # Large matrix on GPU
       m, n = 5000, 3000
       A_gpu = torch.randn(m, n, device=device, dtype=torch.float32)
       
       print(f"GPU matrix shape: {A_gpu.shape}")
       print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
       
       # RSVD on GPU
       start_time = time.time()
       U_gpu, S_gpu, V_gpu = pr.linalg.randomized_svd(A_gpu, k=100, tol=1e-5)
       gpu_time = time.time() - start_time
       
       print(f"GPU RSVD time: {gpu_time:.4f}s")
       
       # Memory cleanup
       del A_gpu, U_gpu, S_gpu, V_gpu
       torch.cuda.empty_cache()

**Memory-Efficient Large Matrix Processing**

.. code-block:: python

   def chunked_rsvd(A, rank, chunk_size=1000):
       """Process very large matrices in chunks."""
       
       m, n = A.shape
       
       if min(m, n) <= chunk_size:
           # Small enough to process directly
           return pr.linalg.randomized_svd(A, k=rank, tol=1e-6)
       
       # Process in chunks and combine
       if m >= n:
           # Tall matrix: chunk rows
           Q_list = []
           for i in range(0, m, chunk_size):
               end_i = min(i + chunk_size, m)
               A_chunk = A[i:end_i, :]
               
               # Random projection
               Omega = torch.randn(n, rank + 10)
               Y_chunk = A_chunk @ Omega
               Q_chunk, _ = torch.linalg.qr(Y_chunk)
               Q_list.append(Q_chunk)
           
           # Combine Q matrices
           Q_combined = torch.cat(Q_list, dim=0)
           
           # Final RSVD on smaller matrix
           B = Q_combined.T @ A
           U_B, S, V = pr.linalg.randomized_svd(B, k=rank, tol=1e-6)
           U = Q_combined @ U_B
           
       else:
           # Wide matrix: chunk columns  
           # Similar process but transpose operations
           U, S, V = chunked_rsvd(A.T, rank)
           U, V = V.T, U.T
       
       return U, S, V

Practical Applications
----------------------

**1. Image Compression**

.. code-block:: python

   def compress_image_rsvd(image_tensor, compression_ratio=0.1):
       """Compress image using RSVD."""
       
       # image_tensor should be (height, width, channels)
       h, w, c = image_tensor.shape
       
       compressed_channels = []
       for channel in range(c):
           # Extract channel
           channel_data = image_tensor[:, :, channel]
           
           # Determine rank for compression
           max_rank = min(h, w)
           target_rank = int(max_rank * compression_ratio)
           
           # Apply RSVD
           U, S, V = pr.linalg.randomized_svd(channel_data, k=target_rank, tol=1e-6)
           
           # Reconstruct channel
           compressed_channel = U @ torch.diag(S) @ V.T
           compressed_channels.append(compressed_channel)
       
       # Combine channels
       compressed_image = torch.stack(compressed_channels, dim=2)
       
       # Calculate compression statistics
       original_elements = h * w * c
       compressed_elements = c * target_rank * (h + w + 1)
       actual_ratio = compressed_elements / original_elements
       
       return compressed_image, actual_ratio, target_rank
   
   # Example usage (requires PIL/torchvision for real images)
   # For demonstration, create a synthetic image
   h, w, c = 256, 256, 3
   synthetic_image = torch.randn(h, w, c)
   
   compressed, ratio, rank = compress_image_rsvd(synthetic_image, compression_ratio=0.2)
   
   print(f"Image compression:")
   print(f"  Original size: {h}x{w}x{c}")
   print(f"  Compression ratio: {ratio:.3f}")
   print(f"  Rank used: {rank}")
   
   compression_error = torch.norm(synthetic_image - compressed)
   print(f"  Reconstruction error: {compression_error:.4f}")

**2. Dimensionality Reduction Pipeline**

.. code-block:: python

   class RSVDDimensionalityReducer:
       """Complete dimensionality reduction pipeline using RSVD."""
       
       def __init__(self, n_components, whiten=False):
           self.n_components = n_components
           self.whiten = whiten
           self.components_ = None
           self.mean_ = None
           self.singular_values_ = None
           
       def fit(self, X):
           """Fit the model with X."""
           # Center data
           self.mean_ = torch.mean(X, dim=0)
           X_centered = X - self.mean_
           
           # Compute RSVD
           U, S, V = pr.linalg.randomized_svd(X_centered, k=self.n_components, tol=1e-6)
           
           self.singular_values_ = S
           self.components_ = V  # Components as columns
           
           return self
       
       def transform(self, X):
           """Apply dimensionality reduction to X."""
           X_centered = X - self.mean_
           X_transformed = X_centered @ self.components_
           
           if self.whiten:
               X_transformed = X_transformed / self.singular_values_
           
           return X_transformed
       
       def fit_transform(self, X):
           """Fit model and transform X."""
           return self.fit(X).transform(X)
       
       def inverse_transform(self, X_transformed):
           """Transform back to original space."""
           if self.whiten:
               X_transformed = X_transformed * self.singular_values_
           
           X_reconstructed = X_transformed @ self.components_.T + self.mean_
           return X_reconstructed
   
   # Example: Dimensionality reduction on synthetic data
   n_samples, n_features = 1000, 500
   n_informative = 50
   
   # Generate data with structure
   informative_data = torch.randn(n_samples, n_informative)
   random_projection = torch.randn(n_informative, n_features)
   X = informative_data @ random_projection + 0.1 * torch.randn(n_samples, n_features)
   
   # Apply dimensionality reduction
   reducer = RSVDDimensionalityReducer(n_components=100)
   X_reduced = reducer.fit_transform(X)
   X_reconstructed = reducer.inverse_transform(X_reduced)
   
   print(f"Dimensionality reduction results:")
   print(f"  Original shape: {X.shape}")
   print(f"  Reduced shape: {X_reduced.shape}")
   print(f"  Reconstruction error: {torch.norm(X - X_reconstructed):.4f}")
   print(f"  Explained variance: {torch.sum(reducer.singular_values_**2):.2f}")

Computational Complexity and Algorithm Comparison
-------------------------------------------------

**Complexity Analysis**

Both algorithms achieve significant computational savings over classical methods:

.. list-table:: Complexity Comparison
   :header-rows: 1
   :widths: 25 25 25 25

   * - Algorithm
     - Classical
     - CQRRPT
     - RSVD
   * - QR Decomposition
     - :math:`O(mn^2)`
     - :math:`O(mnd + d^3)`
     - N/A
   * - SVD
     - :math:`O(mn \min(m,n))`
     - N/A
     - :math:`O(mnk + k^3)`
   * - Memory
     - :math:`O(mn)`
     - :math:`O(mn + dn)`
     - :math:`O(mn + mk)`

Where:
- :math:`m, n`: matrix dimensions
- :math:`d`: sketch size (:math:`d \approx 1.2n` to :math:`2n` for CQRRPT)
- :math:`k`: target rank (typically :math:`k \ll \min(m,n)`)

**Key Algorithmic Differences**

1. **CQRRPT Features**:
   
   - **Exact decomposition** with column pivoting for numerical stability
   - **Adaptive rank detection** using Cholesky-based condition monitoring
   - **Randomized sketching** design reduces computational complexity
   - **Best for**: Full-rank problems, least squares, when exact QR is needed

2. **RSVD Features**:
   
   - **Low-rank approximation** optimized for dominant singular vectors
   - **Blocked processing** with adaptive error monitoring
   - **Power iterations** enhance accuracy for well-separated singular values
   - **Best for**: Dimensionality reduction, PCA, matrix completion

**Choosing the Right Algorithm**

.. code-block:: python

   def choose_algorithm(A, use_case):
       """Guide for selecting between CQRRPT and RSVD."""
       m, n = A.shape
       
       if use_case == "least_squares":
           # CQRRPT provides exact QR with pivoting
           return "CQRRPT", "Exact solution with numerical stability"
       
       elif use_case == "dimensionality_reduction":
           # RSVD excels at low-rank approximations
           return "RSVD", "Efficient low-rank approximation"
       
       elif use_case == "matrix_completion":
           # RSVD's iterative nature suits completion algorithms
           return "RSVD", "Low-rank structure assumption"
       
       elif use_case == "numerical_rank":
           # CQRRPT's rank detection is more reliable
           return "CQRRPT", "Robust rank estimation via pivoting"
       
       # Matrix shape considerations
       if min(m, n) < 1000:
           return "Classical", "Small matrices: use standard algorithms"
       elif m >> n or n >> m:
           return "CQRRPT", "Rectangular matrices benefit from sketching"
       else:
           return "RSVD", "Square-ish matrices: target rank matters"
   
   # Example usage
   A = torch.randn(2000, 800)
   for use_case in ["least_squares", "dimensionality_reduction", "matrix_completion"]:
       alg, reason = choose_algorithm(A, use_case)
       print(f"{use_case}: Use {alg} - {reason}")

Performance Tips and Best Practices
------------------------------------

**1. Choosing Parameters**

.. code-block:: python

   def recommend_parameters(matrix_shape, target_rank):
       """Recommend RSVD parameters based on matrix properties."""
       m, n = matrix_shape
       min_dim = min(m, n)
       
       # Tolerance recommendations based on target rank
       if target_rank < min_dim * 0.1:
           tolerance = 1e-8  # High accuracy for low-rank approximations
       elif target_rank < min_dim * 0.5:
           tolerance = 1e-6  # Balanced accuracy
       else:
           tolerance = 1e-4  # Faster computation for high-rank
       
       return {
           'k': target_rank,
           'tol': tolerance
       }

**2. Error Analysis**

.. code-block:: python

   def analyze_decomposition_error(A, U, S, Vt, rank_range=None):
       """Analyze approximation error vs rank."""
       
       if rank_range is None:
           rank_range = range(1, len(S) + 1, max(1, len(S) // 20))
       
       errors = []
       for r in rank_range:
           A_approx = U[:, :r] @ torch.diag(S[:r]) @ V[:r, :].T
           error = torch.norm(A - A_approx)
           errors.append(error.item())
       
       return list(rank_range), errors

**3. Numerical Stability Analysis**

.. code-block:: python

   def numerical_stability_test():
       """Compare numerical stability of CQRRPT vs RSVD."""
       
       # Create ill-conditioned matrix
       m, n = 500, 300
       U_cond = torch.randn(m, n)
       # Create exponentially decaying singular values
       singular_values = torch.logspace(0, -10, n)  # condition number ≈ 10^10
       V_cond = torch.randn(n, n)
       
       A_ill = U_cond @ torch.diag(singular_values) @ V_cond.T
       
       print(f"Matrix condition number: {torch.linalg.cond(A_ill):.2e}")
       
       # Test CQRRPT stability
       Q_cqrrpt, R_cqrrpt, P_cqrrpt = pr.linalg.cqrrpt(A_ill)
       cqrrpt_ortho = torch.norm(Q_cqrrpt.T @ Q_cqrrpt - torch.eye(Q_cqrrpt.shape[1]))
       cqrrpt_recon = torch.norm(A_ill[:, P_cqrrpt] - Q_cqrrpt @ R_cqrrpt)
       
       # Test RSVD stability  
       k = min(100, n-10)  # Conservative rank
       U_rsvd, S_rsvd, V_rsvd = pr.linalg.randomized_svd(A_ill, k=k, tol=1e-8)
       rsvd_ortho_u = torch.norm(U_rsvd.T @ U_rsvd - torch.eye(U_rsvd.shape[1]))
       rsvd_ortho_v = torch.norm(V_rsvd.T @ V_rsvd - torch.eye(V_rsvd.shape[1]))
       rsvd_recon = torch.norm(A_ill - U_rsvd @ torch.diag(S_rsvd) @ V_rsvd.T)
       
       print(f"\\nOrthogonality errors:")
       print(f"  CQRRPT Q^T Q: {cqrrpt_ortho:.2e}")
       print(f"  RSVD U^T U: {rsvd_ortho_u:.2e}")
       print(f"  RSVD V^T V: {rsvd_ortho_v:.2e}")
       
       print(f"\\nReconstruction errors:")
       print(f"  CQRRPT: {cqrrpt_recon:.2e}")
       print(f"  RSVD: {rsvd_recon:.2e}")
       
       # Rank detection comparison
       print(f"\\nRank detection:")
       print(f"  CQRRPT effective rank: {Q_cqrrpt.shape[1]}")
       print(f"  RSVD computed rank: {len(S_rsvd)}")
       print(f"  True numerical rank (σ > 1e-12): {(singular_values > 1e-12).sum()}")
   
   numerical_stability_test()

**4. Precision and Tolerance Guidelines**

.. code-block:: python

   def precision_guidelines():
       """Guidelines for choosing tolerances and precision."""
       
       guidelines = {
           "float32": {
               "cqrrpt_eps": 1e-6,
               "rsvd_tol": 1e-5,
               "use_case": "Memory-constrained, moderate accuracy",
               "max_reliable_condition": 1e6
           },
           "float64": {
               "cqrrpt_eps": 1e-12, 
               "rsvd_tol": 1e-8,
               "use_case": "High accuracy requirements",
               "max_reliable_condition": 1e12
           }
       }
       
       print("Precision Guidelines:")
       print("=" * 50)
       
       for dtype, params in guidelines.items():
           print(f"\\n{dtype.upper()}:")
           for key, value in params.items():
               print(f"  {key}: {value}")
       
       print(f"\\nAdaptive tolerance selection:")
       print(f"  For condition number C:")
       print(f"    - Low (C < 1e6): Use default tolerances")
       print(f"    - Medium (1e6 ≤ C < 1e10): Increase tolerance 10x")
       print(f"    - High (C ≥ 1e10): Consider regularization")
   
   precision_guidelines()

This tutorial provides a comprehensive guide to using Panther's matrix decomposition algorithms. The next tutorial will cover building complete neural networks with sketched layers.
