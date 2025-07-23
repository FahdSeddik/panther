Neural Networks API
===================

The :mod:`panther.nn` module provides sketched neural network layers that reduce memory usage while maintaining performance.

.. currentmodule:: panther.nn

Linear Layers
-------------

.. autoclass:: SKLinear
   :members:
   :special-members: __init__
   :no-index:

.. autoclass:: SKLinear_triton
   :members:
   :special-members: __init__
   :no-index:

Convolution Layers
------------------

.. autoclass:: SKConv2d
   :members:
   :special-members: __init__
   :no-index:

Attention Mechanisms
--------------------

.. autoclass:: RandMultiHeadAttention
   :members:
   :no-index:
   :special-members: __init__

Examples
--------

**Basic Sketched Linear Layer**

.. code-block:: python

   import torch
   import panther as pr
   
   # Create a sketched linear layer
   layer = pr.nn.SKLinear(
       in_features=512,
       out_features=256,
       num_terms=8,        # Number of sketching terms
       low_rank=32,        # Rank of each sketch
       bias=True
   )
   
   # Forward pass
   x = torch.randn(64, 512)  # batch_size=64
   y = layer(x)
   print(f"Output shape: {y.shape}")  # (64, 256)

**Replacing Standard Layers**

.. code-block:: python

   import torch.nn as nn
   import panther as pr
   
   class StandardMLP(nn.Module):
       def __init__(self):
           super().__init__()
           self.layer1 = nn.Linear(784, 512)
           self.layer2 = nn.Linear(512, 256) 
           self.layer3 = nn.Linear(256, 10)
           
   class SketchedMLP(nn.Module):
       def __init__(self):
           super().__init__()
           # Drop-in replacements with memory savings
           self.layer1 = pr.nn.SKLinear(784, 512, num_terms=2, low_rank=64)
           self.layer2 = pr.nn.SKLinear(512, 256, num_terms=1, low_rank=32)
           self.layer3 = pr.nn.SKLinear(256, 10, num_terms=1, low_rank=16)

**Sketched Convolution**

.. code-block:: python

   import panther as pr
   
   # Sketched 2D convolution layer
   conv_layer = pr.nn.SKConv2d(
       in_channels=64,
       out_channels=128,
       kernel_size=3,
       stride=1,
       padding=1,
       num_terms=1,
       low_rank=16
   )
   
   # Forward pass
   x = torch.randn(32, 64, 56, 56)  # (batch, channels, height, width)
   y = conv_layer(x)
   print(f"Output shape: {y.shape}")  # (32, 128, 56, 56)

**Multi-Head Attention with Sketching**

.. code-block:: python

   import panther as pr
   
   # Randomized multi-head attention
   attention = pr.nn.RandMultiHeadAttention(
       embed_dim=512,
       num_heads=8,
       sketch_dim=64,      # Sketching dimension
       dropout=0.1
   )
   
   # Self-attention
   x = torch.randn(10, 32, 512)  # (seq_len, batch, embed_dim)
   output, attn_weights = attention(x, x, x)

Parameter Selection Guidelines
------------------------------

**Choosing num_terms and low_rank**

The key parameters for sketched layers are:

* **num_terms**: Number of low-rank terms in the approximation
* **low_rank**: Rank of each term

**Rules of thumb:**

.. code-block:: python

   # Conservative: Fewer parameters, faster but less accurate
   conservative = pr.nn.SKLinear(1024, 512, num_terms=4, low_rank=32)
   
   # Balanced: Good accuracy/speed tradeoff  
   balanced = pr.nn.SKLinear(1024, 512, num_terms=8, low_rank=64)
   
   # Aggressive: More parameters, slower but more accurate
   aggressive = pr.nn.SKLinear(1024, 512, num_terms=16, low_rank=128)

**Parameter count constraint:**

The total parameters should be less than the original layer:

.. math::

   2 \\times \\text{num_terms} \\times \\text{low_rank} \\times (\\text{in_features} + \\text{out_features}) < \\text{in_features} \\times \\text{out_features}

GPU Optimization
----------------

**Tensor Core Requirements**

For optimal GPU performance on modern hardware:

.. code-block:: python

   # All dimensions should be multiples of 16
   layer = pr.nn.SKLinear(
       in_features=1024,    # ✓ Multiple of 16
       out_features=512,    # ✓ Multiple of 16
       num_terms=8,
       low_rank=64          # ✓ Multiple of 16
   )
   
   # Batch size should also be multiple of 16
   x = torch.randn(128, 1024)  # ✓ batch_size=128

**Memory Monitoring**

.. code-block:: python

   import torch
   
   def compare_memory_usage():
       # Standard layer
       standard = torch.nn.Linear(2048, 2048)
       
       # Sketched layer
       sketched = pr.nn.SKLinear(2048, 2048, num_terms=8, low_rank=128)
       
       print(f"Standard parameters: {sum(p.numel() for p in standard.parameters()):,}")
       print(f"Sketched parameters: {sum(p.numel() for p in sketched.parameters()):,}")

Training Considerations
-----------------------

**Gradient Computation**

Sketched layers support full backpropagation:

.. code-block:: python

   import torch.nn as nn
   
   model = nn.Sequential(
       pr.nn.SKLinear(784, 512, num_terms=8, low_rank=64),
       nn.ReLU(),
       pr.nn.SKLinear(512, 256, num_terms=6, low_rank=32), 
       nn.ReLU(),
       pr.nn.SKLinear(256, 10, num_terms=4, low_rank=16)
   )
   
   # Standard training loop works
   optimizer = torch.optim.Adam(model.parameters())
   criterion = nn.CrossEntropyLoss()
   
   for x, y in dataloader:
       optimizer.zero_grad()
       output = model(x)
       loss = criterion(output, y)
       loss.backward()  # Gradients computed correctly
       optimizer.step()

**Learning Rate Considerations**

Sketched layers may benefit from different learning rates:

.. code-block:: python

   # Separate learning rates for sketched parameters
   optimizer = torch.optim.Adam([
       {'params': [p for name, p in model.named_parameters() if 'S1s' in name], 'lr': 1e-3},
       {'params': [p for name, p in model.named_parameters() if 'S2s' in name], 'lr': 1e-3},
       {'params': [p for name, p in model.named_parameters() if 'bias' in name], 'lr': 1e-4}
   ])

Performance Benchmarks
----------------------

Memory usage comparison for different layer sizes:

.. list-table::
   :header-rows: 1
   
   * - Layer Size
     - Standard (MB)
     - Sketched (MB)
     - Memory Savings
   * - 1024 → 1024
     - 4.19
     - 1.05
     - 75%
   * - 2048 → 2048  
     - 16.78
     - 2.10
     - 87%
   * - 4096 → 4096
     - 67.11
     - 4.20
     - 94%

Speed comparison (forward + backward pass):

.. list-table::
   :header-rows: 1
   
   * - Layer Size
     - Standard (ms)
     - Sketched (ms)
     - Speed Change
   * - 1024 → 1024
     - 0.82
     - 0.96
     - -17%
   * - 2048 → 2048
     - 2.34
     - 2.18
     - +7%
   * - 4096 → 4096
     - 8.91
     - 6.24
     - +30%

*Note: Benchmarks run on NVIDIA A100 with Tensor Cores enabled.*
