Neural Networks API
===================

The :mod:`panther.nn` module provides sketched neural network layers that reduce memory usage while maintaining performance.

.. note::
   The neural network classes below require compiled C++ extensions (pawX) which may not be available in all environments.

Linear Layers
-------------

SKLinear
~~~~~~~~

.. py:class:: SKLinear(in_features, out_features, num_terms, low_rank, W_init=None, bias=True, dtype=None, device=None)

   SKLinear is a custom linear (fully connected) layer with sketching and optional low-rank approximation, designed for efficient computation and potential GPU Tensor Core acceleration.

   :param int in_features: Number of input features.
   :param int out_features: Number of output features.
   :param int num_terms: Number of sketching terms (controls the number of low-rank approximations).
   :param int low_rank: Rank of the low-rank approximation for each term.
   :param torch.Tensor W_init: Optional initial weight matrix. If None, weights are initialized using Kaiming uniform initialization.
   :param bool bias: If True, adds a learnable bias to the output. Default: True.
   :param torch.dtype dtype: Data type of the parameters.
   :param torch.device device: Device to store the parameters.

SKLinear_triton
~~~~~~~~~~~~~~~

.. py:class:: SKLinear_triton(in_features, out_features, num_terms, low_rank, W_init=None, bias=True, dtype=None, device=None)

   Triton-accelerated version of SKLinear for enhanced GPU performance.

   :param int in_features: Number of input features.
   :param int out_features: Number of output features.
   :param int num_terms: Number of sketching terms.
   :param int low_rank: Rank of the low-rank approximation for each term.
   :param torch.Tensor W_init: Optional initial weight matrix.
   :param bool bias: If True, adds a learnable bias to the output.
   :param torch.dtype dtype: Data type of the parameters.
   :param torch.device device: Device to store the parameters.

Convolution Layers
------------------

SKConv2d
~~~~~~~~

.. py:class:: SKConv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, num_terms=1, low_rank=64, W_init=None, dtype=None, device=None)

   Sketched 2D convolution layer for memory-efficient convolution operations.

   :param int in_channels: Number of input channels.
   :param int out_channels: Number of output channels.
   :param int kernel_size: Size of the convolution kernel.
   :param int stride: Stride of the convolution operation.
   :param int padding: Padding applied to the input.
   :param bool bias: If True, adds a learnable bias.
   :param int num_terms: Number of sketching terms.
   :param int low_rank: Rank of the low-rank approximation.
   :param torch.Tensor W_init: Optional initial weight matrix.
   :param torch.dtype dtype: Data type of the parameters.
   :param torch.device device: Device to store the parameters.

Attention Mechanisms
--------------------

RandMultiHeadAttention
~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: RandMultiHeadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None, SRPE=None)

   Randomized Multi-Head Attention mechanism with optional sketching for memory efficiency.

   :param int embed_dim: Total dimension of the model.
   :param int num_heads: Number of parallel attention heads.
   :param float dropout: Dropout probability. Default: 0.0.
   :param bool bias: If True, adds bias to input/output projections.
   :param bool add_bias_kv: If True, adds bias to the key and value sequences.
   :param bool add_zero_attn: If True, adds a new batch of zeros to the key and value sequences.
   :param int kdim: Total number of features for keys. Default: None (uses embed_dim).
   :param int vdim: Total number of features for values. Default: None (uses embed_dim).
   :param bool batch_first: If True, batch dimension comes first.
   :param torch.device device: Device to store the parameters.
   :param torch.dtype dtype: Data type of the parameters.
   :param SRPE: Sketched Random Positional Encoding. Default: None.

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
