Custom Sketching Implementation
===============================

This tutorial shows how to create custom sketching operators and extend Panther's functionality with your own algorithms.

Understanding Sketching Fundamentals
-------------------------------------

**Mathematical Foundation**

Sketching reduces the dimensionality of data while preserving important properties. The key insight is that random projections can preserve distances and inner products with high probability.

For a matrix :math:`A \\in \\mathbb{R}^{m \\times n}`, a sketch :math:`S \\in \\mathbb{R}^{k \\times m}` produces:

.. math::

   \\widetilde{A} = SA

where :math:`k \\ll m` and :math:`\\widetilde{A}` preserves key properties of :math:`A`.

**Basic Custom Sketching Operator**

.. code-block:: python

   import torch
   import torch.nn as nn
   import numpy as np
   from abc import ABC, abstractmethod
   
   class BaseSketchingOperator(nn.Module, ABC):
       \"\"\"Base class for all sketching operators.\"\"\""
       
       def __init__(self, input_dim, sketch_dim, device=None):
           super().__init__()
           self.input_dim = input_dim
           self.sketch_dim = sketch_dim
           self.device = device or torch.device('cpu')
           
           # Initialize sketching matrix
           self._init_sketch_matrix()
           
       @abstractmethod
       def _init_sketch_matrix(self):
           \"\"\"Initialize the sketching matrix.\"\"\""
           pass
       
       @abstractmethod
       def forward(self, x):
           \"\"\"Apply sketching to input tensor.\"\"\""
           pass
       
       def sketch_dimension_reduction(self):
           \"\"\"Return the compression ratio.\"\"\""
           return self.input_dim / self.sketch_dim
   
   class GaussianSketch(BaseSketchingOperator):
       \"\"\"Gaussian random projection sketching.\"\"\""
       
       def _init_sketch_matrix(self):
           # Gaussian random matrix with proper normalization
           sketch_matrix = torch.randn(self.sketch_dim, self.input_dim, device=self.device)
           sketch_matrix /= np.sqrt(self.sketch_dim)
           self.register_buffer('sketch_matrix', sketch_matrix)
       
       def forward(self, x):
           # x: (batch_size, input_dim)
           # output: (batch_size, sketch_dim)
           return torch.matmul(x, self.sketch_matrix.t())
   
   # Example usage
   input_dim, sketch_dim = 1000, 200
   gaussian_sketch = GaussianSketch(input_dim, sketch_dim)
   
   x = torch.randn(64, input_dim)
   sketched_x = gaussian_sketch(x)
   
   print(f"Original shape: {x.shape}")
   print(f"Sketched shape: {sketched_x.shape}")
   print(f"Compression ratio: {gaussian_sketch.sketch_dimension_reduction():.2f}x")

Advanced Sketching Algorithms
------------------------------

**Subsampled Randomized Hadamard Transform (SRHT)**

.. code-block:: python

   class SubsampledRandomizedHadamardTransform(BaseSketchingOperator):
       \"\"\"SRHT sketching operator with fast Walsh-Hadamard transform.\"\"\""
       
       def _init_sketch_matrix(self):
           # Find next power of 2 for Hadamard transform
           self.hadamard_size = 2 ** int(np.ceil(np.log2(self.input_dim)))
           
           # Random diagonal matrix D with ±1 entries
           diagonal = torch.randint(0, 2, (self.hadamard_size,), device=self.device) * 2 - 1
           self.register_buffer('diagonal', diagonal.float())
           
           # Random subsampling indices
           indices = torch.randperm(self.hadamard_size, device=self.device)[:self.sketch_dim]
           self.register_buffer('indices', indices)
           
           # Normalization factor
           self.norm_factor = np.sqrt(self.hadamard_size / self.sketch_dim)
       
       def fast_hadamard_transform(self, x):
           \"\"\"Compute fast Walsh-Hadamard transform.\"\"\""
           batch_size, input_size = x.shape
           
           # Pad to power of 2
           if input_size < self.hadamard_size:
               padding = torch.zeros(batch_size, self.hadamard_size - input_size, 
                                   device=x.device, dtype=x.dtype)
               x_padded = torch.cat([x, padding], dim=1)
           else:
               x_padded = x[:, :self.hadamard_size]
           
           # Apply diagonal scaling
           x_scaled = x_padded * self.diagonal
           
           # Fast Walsh-Hadamard Transform
           h = self.hadamard_size
           while h > 1:
               h //= 2
               x_reshaped = x_scaled.view(batch_size, -1, 2 * h)
               left = x_reshaped[:, :, :h]
               right = x_reshaped[:, :, h:]
               x_scaled = torch.cat([left + right, left - right], dim=2)
               x_scaled = x_scaled.view(batch_size, -1)
           
           return x_scaled
       
       def forward(self, x):
           # Apply Hadamard transform
           x_transformed = self.fast_hadamard_transform(x)
           
           # Subsample
           x_subsampled = x_transformed[:, self.indices]
           
           # Normalize
           return x_subsampled * self.norm_factor

**Sparse Johnson-Lindenstrauss Transform**

.. code-block:: python

   class SparseJohnsonLindenstrauss(BaseSketchingOperator):
       \"\"\"Sparse JL transform with very few non-zeros per column.\"\"\""
       
       def _init_sketch_matrix(self):
           # Sparsity parameter (number of non-zeros per column)
           self.sparsity = max(1, int(np.log(self.input_dim)))
           
           # Create sparse sketching matrix
           sketch_matrix = torch.zeros(self.sketch_dim, self.input_dim, device=self.device)
           
           for col in range(self.input_dim):
               # Random row indices for this column
               row_indices = torch.randperm(self.sketch_dim)[:self.sparsity]
               
               # Random ±1 values
               values = torch.randint(0, 2, (self.sparsity,), device=self.device) * 2 - 1
               sketch_matrix[row_indices, col] = values.float()
           
           # Normalize
           sketch_matrix *= np.sqrt(self.sketch_dim / self.sparsity)
           
           # Convert to sparse format for efficiency
           self.register_buffer('sketch_matrix', sketch_matrix)
       
       def forward(self, x):
           return torch.matmul(x, self.sketch_matrix.t())

**CountSketch (Feature Hashing)**

.. code-block:: python

   class CountSketch(BaseSketchingOperator):
       \"\"\"CountSketch using feature hashing.\"\"\""
       
       def _init_sketch_matrix(self):
           # Hash function: maps each feature to a bucket
           hash_indices = torch.randint(0, self.sketch_dim, (self.input_dim,), device=self.device)
           self.register_buffer('hash_indices', hash_indices)
           
           # Sign function: ±1 for each feature
           signs = torch.randint(0, 2, (self.input_dim,), device=self.device) * 2 - 1
           self.register_buffer('signs', signs.float())
       
       def forward(self, x):
           batch_size = x.shape[0]
           
           # Apply signs
           x_signed = x * self.signs
           
           # Initialize output
           output = torch.zeros(batch_size, self.sketch_dim, device=x.device, dtype=x.dtype)
           
           # Hash and accumulate
           for i in range(self.input_dim):
               bucket = self.hash_indices[i]
               output[:, bucket] += x_signed[:, i]
           
           return output

Custom Neural Network Layers
-----------------------------

**Learnable Sketching Layer**

.. code-block:: python

   class LearnableSketchingLayer(nn.Module):
       \"\"\"Sketching layer with learnable parameters.\"\"\""
       
       def __init__(self, input_dim, sketch_dim, sketch_type='gaussian', 
                    learnable=True, regularization_weight=0.01):
           super().__init__()
           
           self.input_dim = input_dim
           self.sketch_dim = sketch_dim
           self.learnable = learnable
           self.regularization_weight = regularization_weight
           
           if sketch_type == 'gaussian':
               self._init_gaussian_sketch()
           elif sketch_type == 'orthogonal':
               self._init_orthogonal_sketch()
           else:
               raise ValueError(f"Unknown sketch type: {sketch_type}")
       
       def _init_gaussian_sketch(self):
           if self.learnable:
               # Learnable sketching matrix
               self.sketch_matrix = nn.Parameter(
                   torch.randn(self.sketch_dim, self.input_dim) / np.sqrt(self.sketch_dim)
               )
           else:
               # Fixed sketching matrix
               sketch_matrix = torch.randn(self.sketch_dim, self.input_dim) / np.sqrt(self.sketch_dim)
               self.register_buffer('sketch_matrix', sketch_matrix)
       
       def _init_orthogonal_sketch(self):
           # Initialize with orthogonal matrix
           if self.sketch_dim <= self.input_dim:
               full_matrix = torch.randn(self.input_dim, self.input_dim)
               q, _ = torch.linalg.qr(full_matrix)
               sketch_matrix = q[:self.sketch_dim, :]
           else:
               # More sketch dimensions than input - use transpose
               full_matrix = torch.randn(self.sketch_dim, self.sketch_dim)
               q, _ = torch.linalg.qr(full_matrix)
               sketch_matrix = q[:, :self.input_dim]
           
           if self.learnable:
               self.sketch_matrix = nn.Parameter(sketch_matrix)
           else:
               self.register_buffer('sketch_matrix', sketch_matrix)
       
       def forward(self, x):
           sketched = torch.matmul(x, self.sketch_matrix.t())
           return sketched
       
       def regularization_loss(self):
           \"\"\"Regularization to maintain sketching properties.\"\"\""
           if not self.learnable:
               return torch.tensor(0.0, device=self.sketch_matrix.device)
           
           # Encourage orthogonality
           gram = torch.matmul(self.sketch_matrix, self.sketch_matrix.t())
           identity = torch.eye(self.sketch_dim, device=gram.device)
           orthogonality_loss = torch.norm(gram - identity)
           
           return self.regularization_weight * orthogonality_loss

**Adaptive Sketching Layer**

.. code-block:: python

   class AdaptiveSketchingLayer(nn.Module):
       \"\"\"Sketching layer that adapts its compression based on input.\"\"\""
       
       def __init__(self, input_dim, min_sketch_dim, max_sketch_dim):
           super().__init__()
           
           self.input_dim = input_dim
           self.min_sketch_dim = min_sketch_dim
           self.max_sketch_dim = max_sketch_dim
           
           # Controller network to determine sketch dimension
           self.controller = nn.Sequential(
               nn.Linear(input_dim, 64),
               nn.ReLU(),
               nn.Linear(64, 32),
               nn.ReLU(),
               nn.Linear(32, 1),
               nn.Sigmoid()
           )
           
           # Multiple sketching matrices for different dimensions
           self.sketch_matrices = nn.ParameterList([
               nn.Parameter(torch.randn(dim, input_dim) / np.sqrt(dim))
               for dim in range(min_sketch_dim, max_sketch_dim + 1)
           ])
       
       def forward(self, x):
           batch_size = x.shape[0]
           
           # Determine sketch dimension based on input statistics
           input_stats = torch.mean(torch.abs(x), dim=1, keepdim=True)  # (batch_size, 1)
           sketch_ratios = self.controller(x)  # (batch_size, 1)
           
           # Convert ratio to actual dimensions
           sketch_dims = (self.min_sketch_dim + 
                         sketch_ratios.squeeze() * (self.max_sketch_dim - self.min_sketch_dim))
           sketch_dims = torch.round(sketch_dims).long()
           
           # Process each sample with appropriate sketch dimension
           outputs = []
           for i in range(batch_size):
               dim_idx = sketch_dims[i] - self.min_sketch_dim
               sketch_matrix = self.sketch_matrices[dim_idx][:sketch_dims[i]]
               
               sample_output = torch.matmul(x[i:i+1], sketch_matrix.t())
               
               # Pad to maximum dimension for batch processing
               if sketch_dims[i] < self.max_sketch_dim:
                   padding = torch.zeros(1, self.max_sketch_dim - sketch_dims[i], 
                                       device=x.device, dtype=x.dtype)
                   sample_output = torch.cat([sample_output, padding], dim=1)
               
               outputs.append(sample_output)
           
           return torch.cat(outputs, dim=0)

Multi-Scale Sketching
---------------------

**Hierarchical Sketching**

.. code-block:: python

   class HierarchicalSketch(nn.Module):
       \"\"\"Multi-scale sketching with different resolutions.\"\"\""
       
       def __init__(self, input_dim, sketch_dims, sketch_types=None):
           super().__init__()
           
           self.input_dim = input_dim
           self.sketch_dims = sketch_dims
           
           if sketch_types is None:
               sketch_types = ['gaussian'] * len(sketch_dims)
           
           # Create sketching operators for each scale
           self.sketchers = nn.ModuleList()
           for dim, sketch_type in zip(sketch_dims, sketch_types):
               if sketch_type == 'gaussian':
                   sketcher = GaussianSketch(input_dim, dim)
               elif sketch_type == 'srht':
                   sketcher = SubsampledRandomizedHadamardTransform(input_dim, dim)
               elif sketch_type == 'count':
                   sketcher = CountSketch(input_dim, dim)
               else:
                   raise ValueError(f"Unknown sketch type: {sketch_type}")
               
               self.sketchers.append(sketcher)
           
           # Fusion network to combine sketches
           total_sketch_dim = sum(sketch_dims)
           self.fusion_network = nn.Sequential(
               nn.Linear(total_sketch_dim, total_sketch_dim // 2),
               nn.ReLU(),
               nn.Linear(total_sketch_dim // 2, input_dim)
           )
       
       def forward(self, x):
           # Apply all sketching operators
           sketches = []
           for sketcher in self.sketchers:
               sketch = sketcher(x)
               sketches.append(sketch)
           
           # Concatenate all sketches
           combined_sketch = torch.cat(sketches, dim=1)
           
           # Optional: fusion for reconstruction
           reconstructed = self.fusion_network(combined_sketch)
           
           return combined_sketch, reconstructed

**Wavelet-Based Sketching**

.. code-block:: python

   class WaveletSketch(BaseSketchingOperator):
       \"\"\"Sketching using wavelet transform.\"\"\""
       
       def _init_sketch_matrix(self):
           # Simple Haar wavelet matrix (for demonstration)
           # In practice, use more sophisticated wavelets
           self.wavelet_size = 2 ** int(np.ceil(np.log2(self.input_dim)))
           
           # Generate Haar wavelet matrix
           wavelet_matrix = self._generate_haar_matrix(self.wavelet_size)
           
           # Select top coefficients (approximation + some detail)
           self.register_buffer('wavelet_matrix', wavelet_matrix[:self.sketch_dim])
       
       def _generate_haar_matrix(self, size):
           \"\"\"Generate Haar wavelet matrix.\"\"\""
           if size == 1:
               return torch.tensor([[1.0]])
           
           # Recursive construction
           half_size = size // 2
           smaller_matrix = self._generate_haar_matrix(half_size)
           
           # Scaling functions
           top_left = torch.kron(smaller_matrix, torch.tensor([1.0, 1.0])) / np.sqrt(2)
           
           # Wavelet functions
           bottom_left = torch.kron(torch.eye(half_size), torch.tensor([1.0, -1.0])) / np.sqrt(2)
           
           matrix = torch.zeros(size, size)
           matrix[:half_size] = top_left
           matrix[half_size:] = bottom_left
           
           return matrix
       
       def forward(self, x):
           batch_size, input_size = x.shape
           
           # Pad to wavelet size
           if input_size < self.wavelet_size:
               padding = torch.zeros(batch_size, self.wavelet_size - input_size, 
                                   device=x.device, dtype=x.dtype)
               x_padded = torch.cat([x, padding], dim=1)
           else:
               x_padded = x[:, :self.wavelet_size]
           
           # Apply wavelet transform and select coefficients
           wavelet_coeffs = torch.matmul(x_padded, self.wavelet_matrix.t())
           
           return wavelet_coeffs

Custom Sketched Linear Layers
------------------------------

**Matrix-Free Sketched Linear Layer**

.. code-block:: python

   class MatrixFreeSketchedLinear(nn.Module):
       \"\"\"Sketched linear layer without explicit matrix storage.\"\"\""
       
       def __init__(self, in_features, out_features, sketch_dim, seed=42):
           super().__init__()
           
           self.in_features = in_features
           self.out_features = out_features
           self.sketch_dim = sketch_dim
           self.seed = seed
           
           # Only store the low-rank factors
           self.U = nn.Parameter(torch.randn(out_features, sketch_dim) / np.sqrt(sketch_dim))
           self.V = nn.Parameter(torch.randn(in_features, sketch_dim) / np.sqrt(sketch_dim))
           self.bias = nn.Parameter(torch.zeros(out_features))
           
           # Store hash functions for implicit sketching
           self._init_hash_functions()
       
       def _init_hash_functions(self):
           \"\"\"Initialize hash functions for implicit sketching.\"\"\""
           # For reproducible random projections
           generator = torch.Generator()
           generator.manual_seed(self.seed)
           
           # Hash functions for rows and columns
           self.register_buffer('row_hashes', 
                              torch.randint(0, self.sketch_dim, (self.out_features,), 
                                          generator=generator))
           self.register_buffer('col_hashes', 
                              torch.randint(0, self.sketch_dim, (self.in_features,), 
                                          generator=generator))
           
           # Sign functions
           self.register_buffer('row_signs', 
                              torch.randint(0, 2, (self.out_features,), generator=generator) * 2 - 1)
           self.register_buffer('col_signs', 
                              torch.randint(0, 2, (self.in_features,), generator=generator) * 2 - 1)
       
       def implicit_weight_multiply(self, x):
           \"\"\"Compute Wx without explicitly forming W.\"\"\""
           batch_size = x.shape[0]
           
           # Step 1: Hash and sign input
           x_hashed = torch.zeros(batch_size, self.sketch_dim, device=x.device, dtype=x.dtype)
           for i in range(self.in_features):
               hash_idx = self.col_hashes[i]
               sign = self.col_signs[i]
               x_hashed[:, hash_idx] += sign * x[:, i]
           
           # Step 2: Apply V.T
           intermediate = torch.matmul(x_hashed, self.V.t())
           
           # Step 3: Apply U and hash to output
           output = torch.zeros(batch_size, self.out_features, device=x.device, dtype=x.dtype)
           u_intermediate = torch.matmul(intermediate, self.U.t())
           
           for i in range(self.out_features):
               hash_idx = self.row_hashes[i]
               sign = self.row_signs[i]
               output[:, i] = sign * u_intermediate[:, hash_idx]
           
           return output
       
       def forward(self, x):
           output = self.implicit_weight_multiply(x)
           return output + self.bias

**Tensorized Sketched Linear Layer**

.. code-block:: python

   class TensorizedSketchedLinear(nn.Module):
       \"\"\"Sketched linear layer using tensor decomposition.\"\"\""
       
       def __init__(self, in_features, out_features, tensor_rank, mode='cp'):
           super().__init__()
           
           self.in_features = in_features
           self.out_features = out_features
           self.tensor_rank = tensor_rank
           self.mode = mode
           
           if mode == 'cp':
               self._init_cp_decomposition()
           elif mode == 'tucker':
               self._init_tucker_decomposition()
           else:
               raise ValueError(f"Unknown tensor mode: {mode}")
           
           self.bias = nn.Parameter(torch.zeros(out_features))
       
       def _init_cp_decomposition(self):
           \"\"\"Initialize CP (CANDECOMP/PARAFAC) decomposition.\"\"\""
           # Factor matrices for CP decomposition
           self.factor_in = nn.Parameter(torch.randn(self.in_features, self.tensor_rank))
           self.factor_out = nn.Parameter(torch.randn(self.out_features, self.tensor_rank))
           
           # Weights for each rank-1 component
           self.weights = nn.Parameter(torch.ones(self.tensor_rank))
       
       def _init_tucker_decomposition(self):
           \"\"\"Initialize Tucker decomposition.\"\"\""
           # Core tensor
           core_size = min(32, self.tensor_rank)
           self.core_tensor = nn.Parameter(torch.randn(core_size, core_size))
           
           # Factor matrices
           self.factor_in = nn.Parameter(torch.randn(self.in_features, core_size))
           self.factor_out = nn.Parameter(torch.randn(self.out_features, core_size))
       
       def forward(self, x):
           if self.mode == 'cp':
               return self._cp_forward(x)
           else:
               return self._tucker_forward(x)
       
       def _cp_forward(self, x):
           \"\"\"Forward pass using CP decomposition.\"\"\""
           # Compute (x @ factor_in) * weights
           x_projected = torch.matmul(x, self.factor_in) * self.weights
           
           # Project to output space
           output = torch.matmul(x_projected, self.factor_out.t())
           
           return output + self.bias
       
       def _tucker_forward(self, x):
           \"\"\"Forward pass using Tucker decomposition.\"\"\""
           # Project input
           x_in = torch.matmul(x, self.factor_in)
           
           # Apply core tensor
           x_core = torch.matmul(x_in, self.core_tensor)
           
           # Project to output
           output = torch.matmul(x_core, self.factor_out.t())
           
           return output + self.bias

Advanced Applications
---------------------

**Online Sketching for Streaming Data**

.. code-block:: python

   class OnlineSketchingLayer(nn.Module):
       \"\"\"Sketching layer that adapts to streaming data.\"\"\""
       
       def __init__(self, input_dim, sketch_dim, decay_rate=0.99):
           super().__init__()
           
           self.input_dim = input_dim
           self.sketch_dim = sketch_dim
           self.decay_rate = decay_rate
           
           # Current sketching matrix
           self.register_buffer('sketch_matrix', 
                              torch.randn(sketch_dim, input_dim) / np.sqrt(sketch_dim))
           
           # Running statistics
           self.register_buffer('running_mean', torch.zeros(input_dim))
           self.register_buffer('running_var', torch.ones(input_dim))
           self.register_buffer('num_batches_tracked', torch.tensor(0))
       
       def update_sketch_matrix(self, x):
           \"\"\"Update sketching matrix based on input statistics.\"\"\""
           
           # Update running statistics
           batch_mean = torch.mean(x, dim=0)
           batch_var = torch.var(x, dim=0)
           
           momentum = 1.0 / (self.num_batches_tracked + 1)
           
           self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
           self.running_var = (1 - momentum) * self.running_var + momentum * batch_var
           
           # Adapt sketching matrix based on feature importance
           importance = torch.sqrt(self.running_var + 1e-8)
           importance = importance / torch.sum(importance)
           
           # Weighted random projection
           if self.num_batches_tracked % 100 == 0:  # Update every 100 batches
               new_sketch = torch.randn(self.sketch_dim, self.input_dim, device=x.device)
               new_sketch = new_sketch * importance.unsqueeze(0)
               new_sketch = new_sketch / torch.norm(new_sketch, dim=1, keepdim=True)
               
               # Exponential moving average of sketch matrix
               self.sketch_matrix = (self.decay_rate * self.sketch_matrix + 
                                   (1 - self.decay_rate) * new_sketch)
           
           self.num_batches_tracked += 1
       
       def forward(self, x):
           if self.training:
               self.update_sketch_matrix(x)
           
           return torch.matmul(x, self.sketch_matrix.t())

**Sketching for Federated Learning**

.. code-block:: python

   class FederatedSketchingLayer(nn.Module):
       \"\"\"Sketching layer designed for federated learning scenarios.\"\"\""
       
       def __init__(self, input_dim, sketch_dim, num_clients=10):
           super().__init__()
           
           self.input_dim = input_dim
           self.sketch_dim = sketch_dim
           self.num_clients = num_clients
           
           # Client-specific sketching matrices
           self.client_sketches = nn.ParameterList([
               nn.Parameter(torch.randn(sketch_dim, input_dim) / np.sqrt(sketch_dim))
               for _ in range(num_clients)
           ])
           
           # Global aggregation weights
           self.aggregation_weights = nn.Parameter(torch.ones(num_clients) / num_clients)
           
           # Privacy noise parameters
           self.noise_scale = 0.1
       
       def forward(self, x, client_id=0, add_noise=False):
           # Use client-specific sketch
           client_sketch = self.client_sketches[client_id]
           sketched = torch.matmul(x, client_sketch.t())
           
           # Add differential privacy noise if requested
           if add_noise and self.training:
               noise = torch.randn_like(sketched) * self.noise_scale
               sketched = sketched + noise
           
           return sketched
       
       def aggregate_sketches(self, client_sketches, client_weights=None):
           \"\"\"Aggregate sketches from multiple clients.\"\"\""
           
           if client_weights is None:
               client_weights = self.aggregation_weights
           
           # Weighted aggregation
           aggregated = torch.zeros_like(client_sketches[0])
           for sketch, weight in zip(client_sketches, client_weights):
               aggregated += weight * sketch
           
           return aggregated

Testing and Validation
-----------------------

**Sketching Quality Assessment**

.. code-block:: python

   class SketchingQualityTester:
       \"\"\"Test suite for validating sketching quality.\"\"\""
       
       def __init__(self, sketching_operator):
           self.sketching_op = sketching_operator
       
       def test_distance_preservation(self, num_tests=1000, tolerance=0.1):
           \"\"\"Test if sketching preserves pairwise distances.\"\"\""
           
           distance_errors = []
           
           for _ in range(num_tests):
               # Generate random vectors
               x1 = torch.randn(1, self.sketching_op.input_dim)
               x2 = torch.randn(1, self.sketching_op.input_dim)
               
               # Original distance
               original_dist = torch.norm(x1 - x2)
               
               # Sketched distance
               sx1 = self.sketching_op(x1)
               sx2 = self.sketching_op(x2)
               sketched_dist = torch.norm(sx1 - sx2)
               
               # Relative error
               if original_dist > 1e-6:
                   error = abs(sketched_dist - original_dist) / original_dist
                   distance_errors.append(error.item())
           
           avg_error = sum(distance_errors) / len(distance_errors)
           success_rate = sum(1 for e in distance_errors if e < tolerance) / len(distance_errors)
           
           return {
               'average_error': avg_error,
               'success_rate': success_rate,
               'errors': distance_errors
           }
       
       def test_inner_product_preservation(self, num_tests=1000):
           \"\"\"Test if sketching preserves inner products.\"\"\""
           
           inner_product_errors = []
           
           for _ in range(num_tests):
               x1 = torch.randn(1, self.sketching_op.input_dim)
               x2 = torch.randn(1, self.sketching_op.input_dim)
               
               # Original inner product
               original_ip = torch.dot(x1.squeeze(), x2.squeeze())
               
               # Sketched inner product
               sx1 = self.sketching_op(x1)
               sx2 = self.sketching_op(x2)
               sketched_ip = torch.dot(sx1.squeeze(), sx2.squeeze())
               
               # Relative error
               if abs(original_ip) > 1e-6:
                   error = abs(sketched_ip - original_ip) / abs(original_ip)
                   inner_product_errors.append(error.item())
           
           return {
               'average_error': sum(inner_product_errors) / len(inner_product_errors),
               'errors': inner_product_errors
           }
       
       def benchmark_performance(self, batch_sizes=[32, 64, 128, 256]):
           \"\"\"Benchmark sketching performance.\"\"\""
           
           results = {}
           
           for batch_size in batch_sizes:
               x = torch.randn(batch_size, self.sketching_op.input_dim)
               
               # Timing
               times = []
               for _ in range(50):
                   start = time.time()
                   _ = self.sketching_op(x)
                   times.append(time.time() - start)
               
               avg_time = sum(times) / len(times)
               throughput = batch_size / avg_time  # samples per second
               
               results[batch_size] = {
                   'avg_time_ms': avg_time * 1000,
                   'throughput': throughput
               }
           
           return results
   
   # Example usage
   gaussian_sketch = GaussianSketch(1000, 200)
   tester = SketchingQualityTester(gaussian_sketch)
   
   distance_results = tester.test_distance_preservation()
   print(f"Distance preservation - Avg error: {distance_results['average_error']:.4f}")
   print(f"Success rate: {distance_results['success_rate']:.2%}")
   
   performance_results = tester.benchmark_performance()
   for batch_size, metrics in performance_results.items():
       print(f"Batch {batch_size}: {metrics['avg_time_ms']:.2f}ms, {metrics['throughput']:.0f} samples/s")

This comprehensive tutorial provides the foundation for creating custom sketching operators and extending Panther's functionality. You can use these patterns to implement specialized sketching algorithms for your specific applications.
