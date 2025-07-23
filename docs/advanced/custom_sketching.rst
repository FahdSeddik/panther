Custom Sketching Methods
=======================

This advanced guide covers implementing your own sketching operators and integrating them with Panther's framework.

Creating Custom Sketching Operators
------------------------------------

**Base Sketching Class**

.. code-block:: python

   import torch
   import torch.nn as nn
   from abc import ABC, abstractmethod
   
   class SketchOperator(nn.Module, ABC):
       """Base class for all sketching operators."""
       
       def __init__(self, input_dim, sketch_dim, device=None, dtype=None):
           super().__init__()
           self.input_dim = input_dim
           self.sketch_dim = sketch_dim
           self.device = device or torch.device('cpu')
           self.dtype = dtype or torch.float32
           
           # Initialize sketching matrix
           self._initialize_sketch_matrix()
       
       @abstractmethod
       def _initialize_sketch_matrix(self):
           """Initialize the sketching matrix."""
           pass
       
       @abstractmethod
       def forward(self, x):
           """Apply sketching to input tensor."""
           pass
       
       def sketch_matrix_norm(self):
           """Compute norm of sketching matrix for analysis."""
           return torch.norm(self.sketch_matrix)

**Example: Custom Hadamard Sketch**

.. code-block:: python

   class HadamardSketch(SketchOperator):
       """Fast Hadamard Transform based sketching."""
       
       def __init__(self, input_dim, sketch_dim, device=None, dtype=None):
           # Ensure input_dim is power of 2 for Hadamard transform
           self.padded_dim = 2 ** int(torch.ceil(torch.log2(torch.tensor(input_dim))))
           super().__init__(self.padded_dim, sketch_dim, device, dtype)
           
       def _initialize_sketch_matrix(self):
           """Initialize random diagonal and sampling matrices."""
           # Random diagonal matrix for pre-conditioning
           self.diagonal = torch.randint(0, 2, (self.input_dim,), 
                                       device=self.device, dtype=self.dtype) * 2 - 1
           
           # Random sampling indices
           self.sampling_indices = torch.randperm(self.input_dim, device=self.device)[:self.sketch_dim]
           
       def _hadamard_transform(self, x):
           """Fast Hadamard Transform implementation."""
           # Simplified Walsh-Hadamard transform
           n = x.shape[-1]
           if n == 1:
               return x
           
           h1 = self._hadamard_transform(x[..., :n//2])
           h2 = self._hadamard_transform(x[..., n//2:])
           
           return torch.cat([h1 + h2, h1 - h2], dim=-1) / torch.sqrt(torch.tensor(2.0))
       
       def forward(self, x):
           """Apply Hadamard sketching."""
           batch_shape = x.shape[:-1]
           
           # Pad if necessary
           if x.shape[-1] < self.input_dim:
               padding = torch.zeros(*batch_shape, self.input_dim - x.shape[-1], 
                                   device=x.device, dtype=x.dtype)
               x_padded = torch.cat([x, padding], dim=-1)
           else:
               x_padded = x
           
           # Apply diagonal pre-conditioning
           x_diag = x_padded * self.diagonal
           
           # Apply Hadamard transform
           x_hadamard = self._hadamard_transform(x_diag)
           
           # Sample rows
           x_sketched = x_hadamard[..., self.sampling_indices]
           
           return x_sketched

**Example: Circulant Sketch**

.. code-block:: python

   class CirculantSketch(SketchOperator):
       """Circulant matrix based sketching."""
       
       def _initialize_sketch_matrix(self):
           """Initialize circulant matrix from first row."""
           # Random first row
           first_row = torch.randn(self.input_dim, device=self.device, dtype=self.dtype)
           
           # Create circulant matrix efficiently using FFT
           self.first_row_fft = torch.fft.fft(first_row)
           
           # Random sampling for dimensionality reduction
           self.sampling_indices = torch.randperm(self.input_dim, device=self.device)[:self.sketch_dim]
       
       def forward(self, x):
           """Apply circulant sketching using FFT."""
           # Convert to frequency domain
           x_fft = torch.fft.fft(x, dim=-1)
           
           # Element-wise multiplication in frequency domain
           # (equivalent to circulant matrix multiplication)
           sketched_fft = x_fft * self.first_row_fft
           
           # Convert back to time domain
           sketched = torch.fft.ifft(sketched_fft, dim=-1).real
           
           # Sample to reduce dimension
           return sketched[..., self.sampling_indices]

Advanced Sketching Techniques
-----------------------------

**Hierarchical Sketching**

.. code-block:: python

   class HierarchicalSketch(SketchOperator):
       """Multi-level hierarchical sketching for very large inputs."""
       
       def __init__(self, input_dim, sketch_dim, num_levels=3, device=None, dtype=None):
           self.num_levels = num_levels
           super().__init__(input_dim, sketch_dim, device, dtype)
           
       def _initialize_sketch_matrix(self):
           """Initialize hierarchy of sketching operators."""
           self.sketches = nn.ModuleList()
           
           current_dim = self.input_dim
           target_dim = self.sketch_dim
           
           # Calculate intermediate dimensions
           dims = [current_dim]
           ratio = (target_dim / current_dim) ** (1/self.num_levels)
           
           for i in range(self.num_levels):
               current_dim = int(current_dim * ratio)
               dims.append(max(current_dim, target_dim))
           
           # Create sketching operators for each level
           for i in range(self.num_levels):
               from panther.sketch import GaussianSketch
               sketch = GaussianSketch(
                   dims[i], dims[i+1], 
                   device=self.device, dtype=self.dtype
               )
               self.sketches.append(sketch)
       
       def forward(self, x):
           """Apply hierarchical sketching."""
           current = x
           for sketch in self.sketches:
               current = sketch(current)
           return current

**Adaptive Sketching**

.. code-block:: python

   class AdaptiveSketch(SketchOperator):
       """Sketching that adapts based on input characteristics."""
       
       def __init__(self, input_dim, sketch_dim, adaptation_rate=0.01, device=None, dtype=None):
           self.adaptation_rate = adaptation_rate
           super().__init__(input_dim, sketch_dim, device, dtype)
           
       def _initialize_sketch_matrix(self):
           """Initialize base sketching matrix and adaptation parameters."""
           self.sketch_matrix = torch.randn(self.sketch_dim, self.input_dim,
                                          device=self.device, dtype=self.dtype)
           
           # Running statistics for adaptation
           self.register_buffer('input_mean', torch.zeros(self.input_dim))
           self.register_buffer('input_var', torch.ones(self.input_dim))
           self.register_buffer('update_count', torch.tensor(0))
       
       def _update_statistics(self, x):
           """Update running statistics of input distribution."""
           batch_mean = torch.mean(x, dim=0)
           batch_var = torch.var(x, dim=0)
           
           # Exponential moving average
           alpha = self.adaptation_rate
           self.input_mean = (1 - alpha) * self.input_mean + alpha * batch_mean
           self.input_var = (1 - alpha) * self.input_var + alpha * batch_var
           
           self.update_count += 1
       
       def _adapt_sketch_matrix(self):
           """Adapt sketching matrix based on input statistics."""
           if self.update_count > 10:  # Only adapt after seeing some data
               # Weight sketching matrix by inverse variance (focus on important dimensions)
               weights = 1.0 / (self.input_var + 1e-8)
               weights = weights / torch.sum(weights) * self.input_dim  # Normalize
               
               # Apply weights to sketching matrix
               self.sketch_matrix = self.sketch_matrix * weights.unsqueeze(0)
       
       def forward(self, x):
           """Apply adaptive sketching."""
           if self.training:
               self._update_statistics(x)
               self._adapt_sketch_matrix()
           
           return torch.matmul(x, self.sketch_matrix.t())

Integration with Panther Framework
----------------------------------

**Custom SKLinear Layer**

.. code-block:: python

   class CustomSKLinear(nn.Module):
       """Linear layer with custom sketching operator."""
       
       def __init__(self, in_features, out_features, sketch_operator_class, 
                    sketch_dim, **sketch_kwargs):
           super().__init__()
           
           self.in_features = in_features
           self.out_features = out_features
           self.sketch_dim = sketch_dim
           
           # Initialize custom sketching operator
           self.sketch = sketch_operator_class(
               in_features, sketch_dim, **sketch_kwargs
           )
           
           # Sketched weight matrix
           self.weight = nn.Parameter(torch.randn(out_features, sketch_dim))
           self.bias = nn.Parameter(torch.zeros(out_features))
           
       def forward(self, x):
           """Forward pass with custom sketching."""
           # Apply sketching to input
           x_sketched = self.sketch(x)
           
           # Linear transformation on sketched input
           output = torch.matmul(x_sketched, self.weight.t()) + self.bias
           
           return output
   
   # Example usage with custom Hadamard sketching
   layer = CustomSKLinear(
       in_features=1024,
       out_features=512,
       sketch_operator_class=HadamardSketch,
       sketch_dim=256
   )

**Sketching for Attention Mechanisms**

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
           context = context.transpose(1, 2).contiguous().view(
               batch_size, seq_len, self.d_model
           )
           
           return self.output_proj(context)

Performance Analysis and Debugging
----------------------------------

**Sketching Quality Metrics**

.. code-block:: python

   def analyze_sketching_quality(original_matrix, sketch_operator, num_trials=100):
       """Analyze the quality of a sketching operator."""
       
       results = {
           'reconstruction_errors': [],
           'spectral_norms': [],
           'frobenius_norms': [],
           'sketch_times': []
       }
       
       for trial in range(num_trials):
           # Generate random test vector
           x = torch.randn(1000, original_matrix.shape[1])
           
           # Original computation
           original_result = torch.matmul(x, original_matrix.t())
           
           # Sketched computation
           start_time = time.time()
           x_sketched = sketch_operator(x)
           sketch_time = time.time() - start_time
           
           # Approximate reconstruction
           sketched_result = torch.matmul(x_sketched, 
                                        sketch_operator.sketch_matrix @ original_matrix.t())
           
           # Compute errors
           reconstruction_error = torch.norm(original_result - sketched_result)
           spectral_norm = torch.norm(original_matrix, p=2)
           frobenius_norm = torch.norm(original_matrix, p='fro')
           
           results['reconstruction_errors'].append(reconstruction_error.item())
           results['spectral_norms'].append(spectral_norm.item())
           results['frobenius_norms'].append(frobenius_norm.item())
           results['sketch_times'].append(sketch_time)
       
       # Compute statistics
       stats = {}
       for key, values in results.items():
           stats[key] = {
               'mean': torch.tensor(values).mean().item(),
               'std': torch.tensor(values).std().item(),
               'min': min(values),
               'max': max(values)
           }
       
       return stats

**Memory Usage Profiling**

.. code-block:: python

   def profile_sketching_memory(sketch_operator, input_sizes):
       """Profile memory usage of sketching operators."""
       
       import psutil
       import gc
       
       memory_usage = {}
       
       for size in input_sizes:
           gc.collect()
           torch.cuda.empty_cache() if torch.cuda.is_available() else None
           
           # Measure baseline memory
           baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
           
           # Create input tensor
           x = torch.randn(100, size)
           
           # Measure memory after tensor creation
           tensor_memory = psutil.Process().memory_info().rss / 1024 / 1024
           
           # Apply sketching
           _ = sketch_operator(x)
           
           # Measure memory after sketching
           sketch_memory = psutil.Process().memory_info().rss / 1024 / 1024
           
           memory_usage[size] = {
               'baseline': baseline_memory,
               'tensor': tensor_memory - baseline_memory,
               'sketch': sketch_memory - tensor_memory,
               'total': sketch_memory - baseline_memory
           }
           
           # Cleanup
           del x
           gc.collect()
       
       return memory_usage

Best Practices for Custom Sketching
-----------------------------------

**Design Guidelines**

1. **Preserve Important Properties**

.. code-block:: python

   def validate_sketching_properties(sketch_operator):
       """Validate that sketching operator preserves important properties."""
       
       # Test linearity: S(ax + by) = aS(x) + bS(y)
       x = torch.randn(10, sketch_operator.input_dim)
       y = torch.randn(10, sketch_operator.input_dim) 
       a, b = 2.0, 3.0
       
       lhs = sketch_operator(a * x + b * y)
       rhs = a * sketch_operator(x) + b * sketch_operator(y)
       
       linearity_error = torch.norm(lhs - rhs)
       print(f"Linearity error: {linearity_error:.2e}")
       
       # Test norm preservation (approximately)
       original_norm = torch.norm(x)
       sketched_norm = torch.norm(sketch_operator(x))
       norm_ratio = sketched_norm / original_norm
       
       print(f"Norm preservation ratio: {norm_ratio:.3f}")
       
       return linearity_error < 1e-6, 0.5 < norm_ratio < 2.0

2. **Efficient Implementation**

.. code-block:: python

   # Good: Use vectorized operations
   def efficient_sketch(x, sketch_matrix):
       return torch.matmul(x, sketch_matrix.t())
   
   # Bad: Use loops
   def inefficient_sketch(x, sketch_matrix):
       result = []
       for i in range(sketch_matrix.shape[0]):
           row_result = torch.sum(x * sketch_matrix[i], dim=-1)
           result.append(row_result)
       return torch.stack(result, dim=-1)

3. **Numerical Stability**

.. code-block:: python

   class NumericallyStableSketch(SketchOperator):
       """Sketching operator with numerical stability considerations."""
       
       def _initialize_sketch_matrix(self):
           # Use orthogonal initialization for better conditioning
           matrix = torch.randn(self.sketch_dim, self.input_dim, 
                              device=self.device, dtype=self.dtype)
           
           # QR decomposition for orthogonality
           q, r = torch.linalg.qr(matrix)
           self.sketch_matrix = q
           
       def forward(self, x):
           # Normalize input to prevent overflow
           x_norm = torch.norm(x, dim=-1, keepdim=True)
           x_normalized = x / (x_norm + 1e-8)
           
           # Apply sketching
           sketched = torch.matmul(x_normalized, self.sketch_matrix.t())
           
           # Restore original scale
           return sketched * x_norm

This advanced guide provides the foundation for implementing and integrating custom sketching methods with Panther.
