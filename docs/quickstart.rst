Quickstart Guide
================

This guide will get you up and running with Panther in just a few minutes.

Your First Panther Program
---------------------------

Let's start with a simple example that demonstrates Panther's core functionality:

.. code-block:: python

   import torch
   import panther as pr
   
   # Create a random matrix
   A = torch.randn(1000, 800, dtype=torch.float32)
   
   # Perform randomized QR decomposition
   Q, R, J = pr.linalg.cqrrpt(A)
   
   print(f"Original matrix: {A.shape}")
   print(f"Q (orthogonal): {Q.shape}")
   print(f"R (upper triangular): {R.shape}")
   print(f"Permutation indices: {J.shape}")
   
   # Verify the decomposition
   A_reconstructed = Q @ R[:, J]
   error = torch.norm(A - A_reconstructed)
   print(f"Reconstruction error: {error.item():.6f}")

Core Concepts
-------------

**1. Sketched Linear Layers**

Replace standard linear layers with memory-efficient sketched versions:

.. code-block:: python

   import torch.nn as nn
   import panther as pr
   
   # Standard linear layer
   standard_layer = nn.Linear(in_features=512, out_features=256)
   
   # Sketched linear layer (uses less memory)
   sketched_layer = pr.nn.SKLinear(
       in_features=512,
       out_features=256,
       num_terms=4,      # Number of sketching terms
       low_rank=64       # Rank of each sketch
   )
   
   # Both layers work identically
   x = torch.randn(32, 512)  # batch_size=32
   
   y1 = standard_layer(x)
   y2 = sketched_layer(x)
   
   print(f"Standard output: {y1.shape}")
   print(f"Sketched output: {y2.shape}")

**2. Randomized SVD**

Compute approximate singular value decomposition efficiently:

.. code-block:: python

   # Create a large matrix
   A = torch.randn(2000, 1500)
   
   # Compute top 100 singular values/vectors
   U, S, V = pr.linalg.randomized_svd(A, k=100, tol=1e-6)
   
   print(f"U: {U.shape}, S: {S.shape}, V: {V.shape}")
   
   # Reconstruct low-rank approximation
   A_approx = U @ torch.diag(S) @ V.T
   print(f"Approximation error: {torch.norm(A - A_approx):.6f}")

**3. Sketching Operations**

Create and apply sketching matrices for dimensionality reduction:

.. code-block:: python

   # Create a Gaussian sketching matrix
   sketch_matrix = pr.sketch.dense_sketch_operator(
       m=100,  # output dimension
       n=500,  # input dimension  
       distribution=pr.sketch.DistributionFamily.Gaussian
   )
   
   # Apply sketching to a matrix
   A = torch.randn(500, 1000)
   sketched_A = sketch_matrix @ A
   
   print(f"Original: {A.shape}")
   print(f"Sketched: {sketched_A.shape}")

Building Your First Neural Network
-----------------------------------

Here's how to build a simple neural network using Panther's sketched layers:

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   
   class SketchedMLP(nn.Module):
       def __init__(self, input_dim, hidden_dim, output_dim):
           super().__init__()
           
           # Use sketched linear layers for efficiency
           self.layer1 = pr.nn.SKLinear(
               in_features=input_dim,
               out_features=hidden_dim,
               num_terms=8,
               low_rank=32
           )
           
           self.layer2 = pr.nn.SKLinear(
               in_features=hidden_dim,
               out_features=hidden_dim,
               num_terms=8,
               low_rank=32
           )
           
           self.layer3 = pr.nn.SKLinear(
               in_features=hidden_dim,
               out_features=output_dim,
               num_terms=4,
               low_rank=16
           )
           
           self.relu = nn.ReLU()
           self.dropout = nn.Dropout(0.1)
       
       def forward(self, x):
           x = self.relu(self.layer1(x))
           x = self.dropout(x)
           x = self.relu(self.layer2(x))
           x = self.dropout(x)
           x = self.layer3(x)
           return x
   
   # Create model
   model = SketchedMLP(input_dim=784, hidden_dim=512, output_dim=10)
   
   # Test with sample data
   x = torch.randn(64, 784)  # batch of 64 samples
   output = model(x)
   print(f"Output shape: {output.shape}")

GPU Acceleration
----------------

Panther automatically uses GPU acceleration when available:

.. code-block:: python

   import torch
   import panther as pr
   
   # Check if CUDA is available
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   print(f"Using device: {device}")
   
   # Create model on GPU
   model = pr.nn.SKLinear(
       in_features=1024,
       out_features=512,
       num_terms=16,
       low_rank=64
   ).to(device)
   
   # Create input on GPU
   x = torch.randn(128, 1024, device=device)
   
   # Forward pass (automatically uses CUDA kernels)
   with torch.cuda.device(device):
       output = model(x)
   
   print(f"GPU computation completed. Output shape: {output.shape}")

Tensor Core Optimization
-------------------------

For maximum performance on modern GPUs, ensure dimensions are multiples of 16:

.. code-block:: python

   # Optimized for Tensor Cores (all dimensions are multiples of 16)
   model = pr.nn.SKLinear(
       in_features=1024,    # Multiple of 16 ✓
       out_features=512,    # Multiple of 16 ✓  
       num_terms=8,
       low_rank=64          # Multiple of 16 ✓
   )
   
   # Batch size should also be multiple of 16 for best performance
   x = torch.randn(128, 1024)  # batch_size=128 (multiple of 16) ✓
   output = model(x)

Working with Different Data Types
----------------------------------

Panther supports various PyTorch data types:

.. code-block:: python

   # Float32 (default, good balance of speed and precision)
   model_fp32 = pr.nn.SKLinear(512, 256, num_terms=4, low_rank=32, 
                               dtype=torch.float32)
   
   # Float16 (faster on modern GPUs, uses less memory)
   model_fp16 = pr.nn.SKLinear(512, 256, num_terms=4, low_rank=32,
                               dtype=torch.float16)
   
   # Double precision (slower but more accurate)
   model_fp64 = pr.nn.SKLinear(512, 256, num_terms=4, low_rank=32,
                               dtype=torch.float64)

Common Patterns and Best Practices
-----------------------------------

**1. Choosing Sketching Parameters**

.. code-block:: python

   # Rule of thumb for choosing parameters:
   # - num_terms: Start with 1-3, increase for better accuracy
   # - low_rank: Should be much smaller than min(in_features, out_features)
   # - Total parameters: 2 * num_terms * low_rank * (in_features + out_features)
   #   should be less than in_features * out_features
   
   in_features, out_features = 1024, 512
   
   # Conservative choice (fewer parameters)
   conservative = pr.nn.SKLinear(in_features, out_features, 
                                num_terms=4, low_rank=32)
   
   # Aggressive choice (more parameters but better approximation)  
   aggressive = pr.nn.SKLinear(in_features, out_features,
                              num_terms=8, low_rank=64)

**2. Memory Monitoring**

.. code-block:: python

   import torch
   
   def print_memory_usage():
       if torch.cuda.is_available():
           allocated = torch.cuda.memory_allocated() / 1024**3  # GB
           cached = torch.cuda.memory_reserved() / 1024**3     # GB
           print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
   
   # Monitor memory usage
   print_memory_usage()
   
   # Create large model
   model = pr.nn.SKLinear(4096, 4096, num_terms=16, low_rank=128)
   model = model.cuda()
   
   print_memory_usage()

Next Steps
----------

Now that you understand the basics, explore these advanced topics:

* :doc:`tutorials/index` - Detailed tutorials and examples
* :doc:`api/nn` - Complete neural network API reference  
* :doc:`examples/resnet_sketching` - Real-world example with ResNet
* :doc:`examples/autotuner_guide` - Automatic hyperparameter tuning
* :doc:`advanced/cuda_kernels` - Custom CUDA kernel development

Need Help?
----------

* Check the API documentation: :doc:`api/linalg`, :doc:`api/nn`, :doc:`api/sketch`, :doc:`api/tuner`, :doc:`api/utils`
* Visit our `GitHub repository <https://github.com/FahdSeddik/panther>`_ for issues and discussions
* See :doc:`examples/index` for more comprehensive examples
