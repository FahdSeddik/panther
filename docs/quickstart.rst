Quickstart Guide
================

This guide will get you up and running with Panther in just a few minutes.

Installation
------------

Install Panther via pip (Windows with CUDA 12.4 GPU only):

.. code-block:: bash

   pip install panther-ml==0.1.2 --extra-index-url https://download.pytorch.org/whl/cu124

.. note::
   **CPU-only support requires building from source.** Panther fully supports CPU-only systems, 
   but the PyPI package currently includes CUDA dependencies. Use the source installation below 
   to automatically build a CPU-only version.

Build from source (for CPU-only systems or custom builds):

.. code-block:: bash

   git clone https://github.com/FahdSeddik/panther.git
   cd panther
   .\\install.ps1  # Windows
   # OR
   make install   # Linux/macOS

Using Docker (all dependencies included) for GPU systems:

.. code-block:: bash

   docker pull fahdseddik/panther-dev

Your First Panther Program
---------------------------

Let's start with a simple example that demonstrates Panther's core functionality:

.. code-block:: python

   import torch
   import panther as pr
   
   # Create a random matrix
   A = torch.randn(1000, 800, dtype=torch.float32)
   
   # Perform randomized QR decomposition with column pivoting
   Q, R, P = pr.linalg.cqrrpt(A, gamma=1.25)
   
   print(f"Original matrix: {A.shape}")
   print(f"Q (orthogonal): {Q.shape}")
   print(f"R (upper triangular): {R.shape}")
   print(f"P (permutation indices): {P.shape}")
   
   # Verify the decomposition
   A_permuted = A[:, P]
   A_reconstructed = Q @ R
   error = torch.norm(A_permuted - A_reconstructed)
   print(f"Reconstruction error: {error.item():.6f}")

Core Concepts
-------------

**1. Sketched Linear Layers**

Replace standard linear layers with memory-efficient sketched versions:

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   
   # Standard linear layer
   standard_layer = nn.Linear(in_features=512, out_features=256)
   
   # Sketched linear layer (uses less memory)
   sketched_layer = pr.nn.SKLinear(
       in_features=8192,
       out_features=8192,
       num_terms=2,      # Number of sketching terms
       low_rank=64       # Rank of each sketch
   )
   
   # Both layers work identically
   x = torch.randn(32, 8192)  # batch_size=32
   
   y1 = standard_layer(x)
   y2 = sketched_layer(x)
   
   print(f"Standard output: {y1.shape}")
   print(f"Sketched output: {y2.shape}")

**2. Randomized SVD**

Compute approximate singular value decomposition efficiently:

.. code-block:: python

   import torch
   import panther as pr
   
   # Create a large matrix
   A = torch.randn(2000, 1500)
   
   # Compute top 100 singular values/vectors
   U, S, V = pr.linalg.randomized_svd(A, k=100, tol=1e-6)
   
   print(f"U: {U.shape}, S: {S.shape}, V: {V.shape}")
   
   # Reconstruct low-rank approximation
   A_approx = U @ torch.diag(S) @ V.T
   print(f"Approximation error: {torch.norm(A - A_approx):.6f}")

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
               num_terms=3,
               low_rank=32
           )
           
           self.layer2 = pr.nn.SKLinear(
               in_features=hidden_dim,
               out_features=hidden_dim,
               num_terms=3,
               low_rank=32
           )
           
           self.layer3 = pr.nn.SKLinear(
               in_features=hidden_dim,
               out_features=output_dim,
               num_terms=2,
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
   model = SketchedMLP(input_dim=8192, hidden_dim=1024, output_dim=10)
   
   # Test with sample data
   x = torch.randn(64, 8192)  # batch of 64 samples
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
       in_features=8192,
       out_features=1024,
       num_terms=1,
       low_rank=128,
       device=device
   )
   
   # Create input on GPU
   x = torch.randn(128, 8192, device=device)
   
   # Forward pass (automatically uses CUDA kernels)
   output = model(x)
   
   print(f"GPU computation completed. Output shape: {output.shape}")

Tensor Core Optimization
-------------------------

For maximum performance on modern GPUs with Tensor Cores, ensure dimensions are multiples of 16:

.. code-block:: python

   from panther.nn import SKLinear
   
   # Optimized for Tensor Cores (all dimensions are multiples of 16)
   model = SKLinear(
       in_features=512,    # Multiple of 16 ✓
       out_features=256,    # Multiple of 16 ✓  
       num_terms=2,
       low_rank=32          # Multiple of 16 ✓
   )
   
   # Batch size should also be multiple of 16 for best performance
   x = torch.randn(128, 512)  # batch_size=128 (multiple of 16) ✓
   output = model(x)

Working with Different Data Types
----------------------------------

Panther's sketched layers work with standard PyTorch data types. Usage is similar to PyTorch's ``nn.Linear``:

.. code-block:: python

   from panther.nn import SKLinear
   import torch
   
   # Create layer with specific dtype
   model_fp32 = SKLinear(8192, 512, num_terms=1, low_rank=128, 
                         dtype=torch.float32)
   model_fp16 = SKLinear(8192, 512, num_terms=1, low_rank=128,
                         dtype=torch.float16)

Common Patterns and Best Practices
-----------------------------------

**1. Choosing Sketching Parameters**

.. code-block:: python

   from panther.nn import SKLinear
   
   # Rule of thumb for choosing parameters:
   # - num_terms: Start with 1-3, increase for better accuracy
   # - low_rank: Should be much smaller than min(in_features, out_features)
   # - Total parameters: 2 * num_terms * low_rank * (in_features + out_features)
   #   should be less than in_features * out_features
   
   in_features, out_features = 8192, 8192
   
   # Conservative choice (fewer parameters, faster)
   conservative = SKLinear(in_features, out_features, 
                          num_terms=1, low_rank=32)
   
   # Balanced choice (good accuracy-speed tradeoff)
   balanced = SKLinear(in_features, out_features,
                      num_terms=1, low_rank=64)
   
   # Aggressive choice (more parameters but better approximation)  
   aggressive = SKLinear(in_features, out_features,
                        num_terms=2, low_rank=128)

**2. Memory Monitoring**

Use standard PyTorch memory monitoring:

.. code-block:: python

   import torch
   from panther.nn import SKLinear
   
   if torch.cuda.is_available():
       device = torch.device('cuda')
       model = SKLinear(8192, 8192, num_terms=2, low_rank=128, device=device)
       
       allocated = torch.cuda.memory_allocated() / 1024**3
       print(f"GPU Memory Allocated: {allocated:.2f}GB")

Next Steps
----------

Now that you understand the basics, explore these advanced topics:

* :doc:`tutorials/index` - Detailed tutorials and examples
* :doc:`api/nn` - Complete neural network API reference  

Need Help?
----------

* Check the API documentation: :doc:`api/linalg`, :doc:`api/nn`, :doc:`api/sketch`, :doc:`api/tuner`
* Visit our `GitHub repository <https://github.com/FahdSeddik/panther>`_ for issues and discussions
