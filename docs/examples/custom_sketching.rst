Custom Sketching Techniques
============================

This guide demonstrates how to implement custom sketching methods and integrate them with Panther's framework.

Introduction to Custom Sketching
---------------------------------

While Panther provides excellent built-in sketching methods, you may want to implement custom techniques for specific applications. This guide shows how to:

* Create custom sketching operators
* Integrate them with PyTorch modules
* Optimize performance with CUDA
* Use them with AutoTuner

Custom SRHT Implementation
---------------------------

**Subsampled Randomized Hadamard Transform (SRHT)**

.. code-block:: python

   import torch
   import torch.nn as nn
   import numpy as np
   from typing import Tuple
   
   class CustomSRHT(nn.Module):
       """Custom implementation of Subsampled Randomized Hadamard Transform."""
       
       def __init__(self, input_dim: int, sketch_size: int, device=None):
           super().__init__()
           
           self.input_dim = input_dim
           self.sketch_size = sketch_size
           self.device = device or torch.device('cpu')
           
           # Find next power of 2 for Hadamard transform
           self.hadamard_size = 2 ** int(np.ceil(np.log2(input_dim)))
           
           # Random diagonal matrix D with ±1 entries
           diagonal = torch.randint(0, 2, (self.hadamard_size,), device=self.device) * 2 - 1
           self.register_buffer('diagonal', diagonal.float())
           
           # Random subsampling indices
           indices = torch.randperm(self.hadamard_size, device=self.device)[:sketch_size]
           self.register_buffer('indices', indices)
           
           # Normalization factor
           self.norm_factor = np.sqrt(self.hadamard_size / sketch_size)
       
       def hadamard_transform(self, x: torch.Tensor) -> torch.Tensor:
           """Fast Walsh-Hadamard Transform."""
           batch_size = x.shape[0]
           
           # Pad input to next power of 2
           if x.shape[1] < self.hadamard_size:
               padding = torch.zeros(batch_size, self.hadamard_size - x.shape[1], 
                                   device=x.device, dtype=x.dtype)
               x = torch.cat([x, padding], dim=1)
           
           # Apply diagonal scaling
           x = x * self.diagonal
           
           # Fast Walsh-Hadamard Transform
           h = self.hadamard_size
           while h > 1:
               h //= 2
               x = x.view(batch_size, -1, 2 * h)
               x = torch.cat([x[:, :, :h] + x[:, :, h:], x[:, :, :h] - x[:, :, h:]], dim=2)
               x = x.view(batch_size, -1)
           
           return x
       
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """Apply SRHT to input tensor."""
           # Apply Hadamard transform
           x_transformed = self.hadamard_transform(x)
           
           # Subsample
           x_sketched = x_transformed[:, self.indices]
           
           # Normalize
           return x_sketched * self.norm_factor
   
   # Example usage
   def test_custom_srht():
       """Test custom SRHT implementation."""
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
       # Create test data
       batch_size, input_dim = 32, 1000
       sketch_size = 200
       
       x = torch.randn(batch_size, input_dim, device=device)
       
       # Create SRHT operator
       srht = CustomSRHT(input_dim, sketch_size, device=device)
       
       # Apply sketching
       sketched = srht(x)
       
       print(f"Original shape: {x.shape}")
       print(f"Sketched shape: {sketched.shape}")
       print(f"Compression ratio: {input_dim / sketch_size:.2f}x")
       
       # Verify properties
       print(f"Original norm: {torch.norm(x, dim=1).mean():.4f}")
       print(f"Sketched norm: {torch.norm(sketched, dim=1).mean():.4f}")

Custom Gaussian Sketching Layer
--------------------------------

**Optimized Gaussian Random Projection**

.. code-block:: python

   class CustomGaussianSketch(nn.Module):
       """Custom Gaussian sketching with optional structured matrices."""
       
       def __init__(self, input_dim: int, output_dim: int, 
                    structured: bool = False, device=None):
           super().__init__()
           
           self.input_dim = input_dim
           self.output_dim = output_dim
           self.structured = structured
           self.device = device or torch.device('cpu')
           
           if structured:
               # Use structured random matrix (FFT-based)
               self._init_structured_matrix()
           else:
               # Dense Gaussian matrix
               self._init_dense_matrix()
       
       def _init_dense_matrix(self):
           """Initialize dense Gaussian matrix."""
           # Dense random matrix
           matrix = torch.randn(self.output_dim, self.input_dim, device=self.device)
           matrix /= np.sqrt(self.output_dim)  # Normalize
           self.register_buffer('sketch_matrix', matrix)
       
       def _init_structured_matrix(self):
           """Initialize structured matrix using FFT."""
           # Random diagonal matrix
           diagonal = torch.randn(self.input_dim, device=self.device)
           self.register_buffer('diagonal', diagonal)
           
           # Random subsampling for output
           if self.output_dim < self.input_dim:
               indices = torch.randperm(self.input_dim, device=self.device)[:self.output_dim]
               self.register_buffer('indices', indices)
           else:
               self.register_buffer('indices', torch.arange(self.input_dim, device=self.device))
       
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """Apply Gaussian sketching."""
           if self.structured:
               return self._structured_forward(x)
           else:
               return self._dense_forward(x)
       
       def _dense_forward(self, x: torch.Tensor) -> torch.Tensor:
           """Dense matrix multiplication."""
           return torch.matmul(x, self.sketch_matrix.t())
       
       def _structured_forward(self, x: torch.Tensor) -> torch.Tensor:
           """Structured sketching using FFT."""
           # Apply diagonal scaling
           x_scaled = x * self.diagonal
           
           # Apply FFT
           x_fft = torch.fft.fft(x_scaled, dim=-1)
           
           # Take real part and subsample
           x_real = x_fft.real[:, self.indices]
           
           # Normalize
           return x_real / np.sqrt(self.output_dim)

Custom Sketched Linear Layer
-----------------------------

**Building a Complete Custom Layer**

.. code-block:: python

   class CustomSketchedLinear(nn.Module):
       """Custom sketched linear layer with multiple sketching options."""
       
       def __init__(self, in_features: int, out_features: int, 
                    sketch_method: str = 'gaussian', sketch_ratio: float = 0.5,
                    bias: bool = True, device=None):
           super().__init__()
           
           self.in_features = in_features
           self.out_features = out_features
           self.sketch_method = sketch_method
           self.device = device or torch.device('cpu')
           
           # Calculate sketch size
           self.sketch_size = max(8, int(min(in_features, out_features) * sketch_ratio))
           
           # Create sketching operators
           self._init_sketching_operators()
           
           # Low-rank factors
           self.U = nn.Parameter(torch.randn(out_features, self.sketch_size, device=self.device))
           self.V = nn.Parameter(torch.randn(in_features, self.sketch_size, device=self.device))
           
           # Bias
           if bias:
               self.bias = nn.Parameter(torch.zeros(out_features, device=self.device))
           else:
               self.register_parameter('bias', None)
           
           self._init_parameters()
       
       def _init_sketching_operators(self):
           """Initialize sketching operators."""
           if self.sketch_method == 'gaussian':
               self.input_sketch = CustomGaussianSketch(
                   self.in_features, self.sketch_size, device=self.device
               )
               self.output_sketch = CustomGaussianSketch(
                   self.out_features, self.sketch_size, device=self.device
               )
           elif self.sketch_method == 'srht':
               self.input_sketch = CustomSRHT(
                   self.in_features, self.sketch_size, device=self.device
               )
               self.output_sketch = CustomSRHT(
                   self.out_features, self.sketch_size, device=self.device
               )
           else:
               raise ValueError(f"Unknown sketch method: {self.sketch_method}")
       
       def _init_parameters(self):
           """Initialize parameters using proper scaling."""
           # Xavier/Glorot initialization adapted for low-rank
           fan_in = self.in_features
           fan_out = self.out_features
           std = np.sqrt(2.0 / (fan_in + fan_out))
           
           # Scale for low-rank decomposition
           scale = std / np.sqrt(self.sketch_size)
           
           with torch.no_grad():
               self.U.normal_(0, scale)
               self.V.normal_(0, scale)
       
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """Forward pass through sketched linear layer."""
           # Compute low-rank approximation: W ≈ U @ S @ V.T
           # where S is derived from sketching
           
           # Method 1: Direct low-rank multiplication
           # output = x @ self.V @ self.U.t()
           
           # Method 2: More sophisticated sketching (example)
           x_sketched = self.input_sketch(x)  # Sketch input
           intermediate = x_sketched @ (self.V.t() @ self.U)  # Low-rank computation
           output = intermediate
           
           if self.bias is not None:
               output = output + self.bias
           
           return output
       
       def extra_repr(self) -> str:
           return (f'in_features={self.in_features}, out_features={self.out_features}, '
                  f'sketch_size={self.sketch_size}, sketch_method={self.sketch_method}, '
                  f'bias={self.bias is not None}')

Advanced Custom Network with Mixed Sketching
---------------------------------------------

**Combining Multiple Sketching Techniques**

.. code-block:: python

   class AdaptiveSketchedNetwork(nn.Module):
       """Network that adapts sketching method per layer."""
       
       def __init__(self, layer_configs: list, adaptive: bool = True):
           super().__init__()
           
           self.adaptive = adaptive
           self.layers = nn.ModuleList()
           
           for i, config in enumerate(layer_configs):
               in_feat, out_feat = config['in_features'], config['out_features']
               
               # Choose sketching method adaptively
               if adaptive:
                   sketch_method = self._choose_sketch_method(in_feat, out_feat, i)
                   sketch_ratio = self._choose_sketch_ratio(in_feat, out_feat, i)
               else:
                   sketch_method = config.get('sketch_method', 'gaussian')
                   sketch_ratio = config.get('sketch_ratio', 0.5)
               
               layer = CustomSketchedLinear(
                   in_feat, out_feat, 
                   sketch_method=sketch_method,
                   sketch_ratio=sketch_ratio
               )
               
               self.layers.append(layer)
               
               # Add activation if specified
               if config.get('activation'):
                   activation = getattr(nn, config['activation'])()
                   self.layers.append(activation)
       
       def _choose_sketch_method(self, in_feat: int, out_feat: int, layer_idx: int) -> str:
           """Adaptively choose sketching method."""
           # Use SRHT for large layers (better for high dimensions)
           if min(in_feat, out_feat) > 1000:
               return 'srht'
           # Use Gaussian for smaller layers
           else:
               return 'gaussian'
       
       def _choose_sketch_ratio(self, in_feat: int, out_feat: int, layer_idx: int) -> float:
           """Adaptively choose sketch ratio."""
           # Higher compression for earlier layers
           base_ratio = 0.3 + 0.2 * (layer_idx / max(1, len(self.layers) - 1))
           
           # Adjust based on layer size
           size_factor = min(in_feat, out_feat) / 1000
           adjusted_ratio = base_ratio * (0.5 + 0.5 * np.tanh(size_factor))
           
           return np.clip(adjusted_ratio, 0.1, 0.8)
       
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """Forward pass through adaptive network."""
           for layer in self.layers:
               x = layer(x)
           return x
       
       def get_compression_info(self) -> dict:
           """Get compression statistics for each layer."""
           info = {}
           
           for i, layer in enumerate(self.layers):
               if isinstance(layer, CustomSketchedLinear):
                   original_params = layer.in_features * layer.out_features
                   sketched_params = (layer.in_features + layer.out_features) * layer.sketch_size
                   compression_ratio = original_params / sketched_params
                   
                   info[f'layer_{i}'] = {
                       'method': layer.sketch_method,
                       'original_params': original_params,
                       'sketched_params': sketched_params,
                       'compression_ratio': compression_ratio,
                       'sketch_size': layer.sketch_size
                   }
           
           return info

CUDA-Accelerated Custom Sketching
----------------------------------

**Custom CUDA Kernel for Gaussian Sketching**

.. code-block:: python

   # Note: This requires PyTorch C++/CUDA extension compilation
   # For simplicity, we show the Python interface
   
   try:
       import pawX  # Assuming pawX provides CUDA kernels
       CUDA_AVAILABLE = True
   except ImportError:
       CUDA_AVAILABLE = False
   
   class CUDAGaussianSketch(nn.Module):
       """CUDA-accelerated Gaussian sketching."""
       
       def __init__(self, input_dim: int, sketch_size: int, device=None):
           super().__init__()
           
           self.input_dim = input_dim
           self.sketch_size = sketch_size
           self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           
           if not (CUDA_AVAILABLE and self.device.type == 'cuda'):
               print("Warning: CUDA not available, falling back to CPU implementation")
               self.use_cuda = False
           else:
               self.use_cuda = True
           
           # Pre-generate random matrix
           self._init_random_matrix()
       
       def _init_random_matrix(self):
           """Initialize random matrix for sketching."""
           if self.use_cuda:
               # For CUDA implementation, we might use structured matrices
               # or generate matrices on-the-fly in kernels
               self.register_buffer('random_matrix', 
                                  torch.randn(self.sketch_size, self.input_dim, device=self.device))
           else:
               # Standard PyTorch implementation
               matrix = torch.randn(self.sketch_size, self.input_dim, device=self.device)
               matrix /= np.sqrt(self.sketch_size)
               self.register_buffer('random_matrix', matrix)
       
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """Apply CUDA-accelerated sketching."""
           if self.use_cuda:
               # Use custom CUDA kernel (hypothetical)
               # return pawX.gaussian_sketch(x, self.random_matrix)
               pass
           
           # Fallback to standard implementation
           return torch.matmul(x, self.random_matrix.t())

AutoTuner Integration with Custom Layers
-----------------------------------------

**Using AutoTuner with Custom Implementations**

.. code-block:: python

   from panther.tuner import SkAutoTuner
   
   def evaluate_custom_network(sketch_ratio_1, sketch_ratio_2, sketch_method_idx):
       """Evaluate custom network with different parameters."""
       
       # Map index to method
       methods = ['gaussian', 'srht']
       sketch_method = methods[int(sketch_method_idx)]
       
       # Create network configuration
       layer_configs = [
           {
               'in_features': 784, 'out_features': 512,
               'sketch_method': sketch_method,
               'sketch_ratio': sketch_ratio_1,
               'activation': 'ReLU'
           },
           {
               'in_features': 512, 'out_features': 256,
               'sketch_method': sketch_method,
               'sketch_ratio': sketch_ratio_2,
               'activation': 'ReLU'
           },
           {
               'in_features': 256, 'out_features': 10,
               'sketch_method': 'gaussian',  # Fixed for output
               'sketch_ratio': 0.8
           }
       ]
       
       # Create and evaluate model
       model = AdaptiveSketchedNetwork(layer_configs, adaptive=False)
       accuracy = quick_train_and_evaluate(model)  # Your training function
       
       return accuracy
   
   # AutoTune custom parameters
   custom_tuner = SkAutoTuner(
       parameter_bounds={
           'sketch_ratio_1': (0.2, 0.8),
           'sketch_ratio_2': (0.2, 0.8),
           'sketch_method_idx': (0, 1)  # 0=gaussian, 1=srht
       },
       objective_function=evaluate_custom_network,
       n_initial_points=10,
       n_iterations=25
   )
   
   best_params, best_score = custom_tuner.optimize()

Performance Benchmarking Custom Methods
----------------------------------------

**Comparing Custom vs. Built-in Sketching**

.. code-block:: python

   import time
   import matplotlib.pyplot as plt
   
   def benchmark_sketching_methods():
       """Benchmark different sketching implementations."""
       
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       batch_size = 64
       input_dim = 2048
       sketch_size = 512
       
       # Test data
       x = torch.randn(batch_size, input_dim, device=device)
       
       # Methods to test
       methods = {
           'Custom Gaussian': CustomGaussianSketch(input_dim, sketch_size, device=device),
           'Custom SRHT': CustomSRHT(input_dim, sketch_size, device=device),
           'Panther SKLinear': pr.nn.SKLinear(input_dim, sketch_size, device=device)
       }
       
       results = {}
       
       for name, method in methods.items():
           # Warmup
           for _ in range(10):
               _ = method(x)
           
           if device.type == 'cuda':
               torch.cuda.synchronize()
           
           # Timing
           start_time = time.time()
           for _ in range(100):
               output = method(x)
               if device.type == 'cuda':
                   torch.cuda.synchronize()
           
           elapsed = time.time() - start_time
           
           results[name] = {
               'time_per_call': elapsed / 100,
               'output_shape': output.shape,
               'memory_usage': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0
           }
       
       # Print results
       print("Benchmarking Results:")
       print("-" * 50)
       for name, stats in results.items():
           print(f"{name}:")
           print(f"  Time per call: {stats['time_per_call']*1000:.2f} ms")
           print(f"  Output shape: {stats['output_shape']}")
           if device.type == 'cuda':
               print(f"  Peak memory: {stats['memory_usage']/1024**2:.1f} MB")
           print()
       
       return results

Best Practices for Custom Sketching
------------------------------------

**1. Memory Efficiency**

.. code-block:: python

   def memory_efficient_sketching(x: torch.Tensor, sketch_size: int) -> torch.Tensor:
       """Memory-efficient sketching for large tensors."""
       
       batch_size, input_dim = x.shape
       
       if input_dim > 10000:  # For very large dimensions
           # Process in chunks to avoid memory issues
           chunk_size = 1000
           chunks = []
           
           for i in range(0, input_dim, chunk_size):
               end_idx = min(i + chunk_size, input_dim)
               chunk = x[:, i:end_idx]
               
               # Apply sketching to chunk
               chunk_sketch_size = min(sketch_size, end_idx - i)
               sketched_chunk = apply_chunk_sketching(chunk, chunk_sketch_size)
               chunks.append(sketched_chunk)
           
           # Combine chunks
           return torch.cat(chunks, dim=1)
       else:
           # Standard sketching
           return apply_standard_sketching(x, sketch_size)

**2. Numerical Stability**

.. code-block:: python

   def stable_sketching(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
       """Numerically stable sketching implementation."""
       
       # Add small regularization to avoid numerical issues
       x_reg = x + epsilon * torch.randn_like(x)
       
       # Ensure proper normalization
       x_normalized = x_reg / (torch.norm(x_reg, dim=1, keepdim=True) + epsilon)
       
       # Apply sketching
       return apply_sketching(x_normalized)

**3. Gradient Flow Optimization**

.. code-block:: python

   class GradientOptimizedSketch(nn.Module):
       """Sketching layer optimized for gradient flow."""
       
       def __init__(self, input_dim: int, sketch_size: int):
           super().__init__()
           
           # Use learnable sketching matrix (optional)
           self.sketch_matrix = nn.Parameter(
               torch.randn(sketch_size, input_dim) / np.sqrt(sketch_size)
           )
           
           # Gradient scaling factor
           self.grad_scale = nn.Parameter(torch.ones(1))
       
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           # Apply sketching with gradient scaling
           sketched = torch.matmul(x, self.sketch_matrix.t())
           return sketched * self.grad_scale

This guide provides a comprehensive foundation for implementing custom sketching techniques within the Panther framework. Remember to benchmark your custom implementations against Panther's optimized built-in methods to ensure they provide the expected benefits for your specific use case.
