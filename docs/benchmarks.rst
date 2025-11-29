Performance Benchmarks
======================

This section presents performance benchmarks for Panther's randomized linear algebra operations and sketched neural network layers.

.. note::
   All benchmarks should be conducted on your specific hardware. The placeholder images below need to be replaced with actual benchmark results. Benchmark notebooks are available in ``tests/notebooks/`` including ``plot_linear_bench.ipynb``, ``plot_attention_bench.ipynb``, and ``plot_conv_bench.ipynb``.

Linear Layer Benchmarks
------------------------

SKLinear Performance
~~~~~~~~~~~~~~~~~~~~

Benchmarks compare standard PyTorch ``nn.Linear`` layers against Panther's ``SKLinear`` layers across various dimensions and configurations.

**Forward Pass Time**

.. image:: _static/benchmarks/linear_forward_t4.png
   :alt: SKLinear Forward Pass Performance
   :align: center
   :width: 100%

*Placeholder: Replace with forward pass benchmark comparing PyTorch Linear vs SKLinear for different input×output dimensions (e.g., 256×256 to 65536×65536) with various num_terms and low_rank configurations.*

**Backward Pass Time**

.. image:: _static/benchmarks/linear_backward_t4.png
   :alt: SKLinear Backward Pass Performance
   :align: center
   :width: 100%

*Placeholder: Replace with backward pass benchmark data.*

.. image:: _static/benchmarks/linear_forward_p100.png
   :alt: SKLinear Forward Pass on Different Hardware
   :align: center
   :width: 100%

.. image:: _static/benchmarks/linear_backward_p100.png
   :alt: SKLinear Backward Pass on Different Hardware
   :align: center
   :width: 100%

*Placeholders: Add benchmarks from different GPU architectures as needed.*

Randomized Attention Benchmarks
--------------------------------

RandMultiHeadAttention benchmarks across multiple configurations.

*Note: These are placeholder images. Replace with actual benchmarking results from your experiments varying embedding dimensions, kernel functions (ReLU/Softmax), number of random features, attention heads, and sequence lengths.*

Forward Pass Time - Various Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/benchmarks/attention_forward_time_128_relu.png
   :alt: Attention Forward Time
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_forward_time_128_softmax.png
   :alt: Attention Forward Time
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_forward_time_256_relu.png
   :alt: Attention Forward Time
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_forward_time_256_softmax.png
   :alt: Attention Forward Time
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_forward_time_512_relu.png
   :alt: Attention Forward Time
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_forward_time_512_softmax.png
   :alt: Attention Forward Time  
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_forward_time_1024_relu.png
   :alt: Attention Forward Time
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_forward_time_1024_softmax.png
   :alt: Attention Forward Time
   :align: center
   :width: 100%

*Placeholders: Add benchmark results comparing standard attention vs randomized attention for different embedding dimensions and kernel types.*

Backward Pass Time
~~~~~~~~~~~~~~~~~~

.. image:: _static/benchmarks/attention_backward_time_128_relu.png
   :alt: Attention Backward Time
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_backward_time_128_softmax.png
   :alt: Attention Backward Time
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_backward_time_256_relu.png
   :alt: Attention Backward Time
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_backward_time_256_softmax.png
   :alt: Attention Backward Time
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_backward_time_512_relu.png
   :alt: Attention Backward Time
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_backward_time_512_softmax.png
   :alt: Attention Backward Time
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_backward_time_1024_relu.png
   :alt: Attention Backward Time
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_backward_time_1024_softmax.png
   :alt: Attention Backward Time
   :align: center
   :width: 100%

Memory Usage
~~~~~~~~~~~~

.. image:: _static/benchmarks/attention_memory_128_relu.png
   :alt: Attention Memory Usage
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_memory_128_softmax.png
   :alt: Attention Memory Usage
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_memory_256_relu.png
   :alt: Attention Memory Usage
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_memory_256_softmax.png
   :alt: Attention Memory Usage
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_memory_512_relu.png
   :alt: Attention Memory Usage
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_memory_512_softmax.png
   :alt: Attention Memory Usage
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_memory_1024_relu.png
   :alt: Attention Memory Usage
   :align: center
   :width: 100%

.. image:: _static/benchmarks/attention_memory_1024_softmax.png
   :alt: Attention Memory Usage
   :align: center
   :width: 100%

*Placeholders: Add memory usage benchmarks for different configurations.*

Convolution Layer Benchmarks
-----------------------------

SKConv2d performance across different kernel sizes and image dimensions.

*Note: Placeholder images - replace with actual benchmarking results from your experiments.*

Kernel Sizes and Image Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: _static/benchmarks/conv_forward_k3_im64.png
   :alt: Conv Forward - Kernel 3, Image 64
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_forward_k3_im128.png
   :alt: Conv Forward - Kernel 3, Image 128
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_forward_k3_im256.png
   :alt: Conv Forward - Kernel 3, Image 256
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_backward_k3_im64.png
   :alt: Conv Backward - Kernel 3, Image 64
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_backward_k3_im128.png
   :alt: Conv Backward - Kernel 3, Image 128
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_backward_k3_im256.png
   :alt: Conv Backward - Kernel 3, Image 256
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_forward_k5_im64.png
   :alt: Conv Forward - Kernel 5, Image 64
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_forward_k5_im128.png
   :alt: Conv Forward - Kernel 5, Image 128
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_forward_k5_im256.png
   :alt: Conv Forward - Kernel 5, Image 256
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_backward_k5_im64.png
   :alt: Conv Backward - Kernel 5, Image 64
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_backward_k5_im128.png
   :alt: Conv Backward - Kernel 5, Image 128
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_backward_k5_im256.png
   :alt: Conv Backward - Kernel 5, Image 256
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_forward_k9_im64.png
   :alt: Conv Forward - Kernel 9, Image 64
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_forward_k9_im128.png
   :alt: Conv Forward - Kernel 9, Image 128
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_forward_k9_im256.png
   :alt: Conv Forward - Kernel 9, Image 256
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_backward_k9_im64.png
   :alt: Conv Backward - Kernel 9, Image 64
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_backward_k9_im128.png
   :alt: Conv Backward - Kernel 9, Image 128
   :align: center
   :width: 45%

.. image:: _static/benchmarks/conv_backward_k9_im256.png
   :alt: Conv Backward - Kernel 9, Image 256
   :align: center
   :width: 45%

Matrix Decomposition Benchmarks
--------------------------------

Randomized SVD and CQRRPT performance.

*Note: Placeholder images - replace with actual benchmarking results.*

.. image:: _static/benchmarks/rsvd_runtime.png
   :alt: RSVD Runtime Comparison
   :align: center
   :width: 45%

.. image:: _static/benchmarks/rsvd_error.png
   :alt: RSVD Approximation Error
   :align: center
   :width: 45%

.. image:: _static/benchmarks/rsvd_memory.png
   :alt: RSVD Memory Usage
   :align: center
   :width: 45%

**Test Configuration:**

- Matrix sizes: 1000×1000 to 16384×16384
- Ranks: 50, 100, 200, 500
- Tolerance: 1e-6

CholeskyQR with Pivoting (CQRRPT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Performance benchmarks for CQRRPT across different matrix sizes.

**Matrix Size m=8192**

.. image:: _static/benchmarks/cqrrpt_runtime_m8192.png
   :alt: CQRRPT Runtime - m=8192
   :align: center
   :width: 45%

.. image:: _static/benchmarks/cqrrpt_error_m8192.png
   :alt: CQRRPT Error - m=8192
   :align: center
   :width: 45%

**Matrix Size m=16384**

.. image:: _static/benchmarks/cqrrpt_runtime_m16384.png
   :alt: CQRRPT Runtime - m=16384
   :align: center
   :width: 45%

.. image:: _static/benchmarks/cqrrpt_error_m16384.png
   :alt: CQRRPT Error - m=16384
   :align: center
   :width: 45%

**Key Observations:**

- RSVD provides significant speedups for rank-k approximations (k << min(m,n))
- CQRRPT maintains numerical stability better than standard QR for tall matrices
- Both algorithms scale well to large matrices

Running Your Own Benchmarks
----------------------------

Use Panther's benchmark utilities to evaluate performance on your hardware:

.. code-block:: python

   import torch
   from panther.nn import SKLinear
   import time

   def benchmark_linear_layer(in_features, out_features, num_terms, low_rank, num_runs=100):
       \"\"\"Benchmark SKLinear layer.\"\"\"
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
       # Create layers
       standard = torch.nn.Linear(in_features, out_features).to(device)
       sketched = SKLinear(in_features, out_features, num_terms, low_rank).to(device)
       
       # Test data
       x = torch.randn(128, in_features, device=device)
       
       # Warmup
       for _ in range(10):
           _ = standard(x)
           _ = sketched(x)
       
       # Benchmark standard
       torch.cuda.synchronize() if device.type == 'cuda' else None
       start = time.time()
       for _ in range(num_runs):
           _ = standard(x)
           torch.cuda.synchronize() if device.type == 'cuda' else None
       standard_time = (time.time() - start) / num_runs
       
       # Benchmark sketched
       torch.cuda.synchronize() if device.type == 'cuda' else None
       start = time.time()
       for _ in range(num_runs):
           _ = sketched(x)
           torch.cuda.synchronize() if device.type == 'cuda' else None
       sketched_time = (time.time() - start) / num_runs
       
       print(f"Standard: {standard_time*1000:.2f}ms")
       print(f"Sketched: {sketched_time*1000:.2f}ms")
       print(f"Speedup: {standard_time/sketched_time:.2f}x")

   # Run benchmark
   benchmark_linear_layer(2048, 2048, num_terms=4, low_rank=128)

.. image:: _static/benchmarks/rsvd_runtime.png
   :alt: RSVD Runtime
   :align: center
   :width: 45%

.. image:: _static/benchmarks/rsvd_error.png
   :alt: RSVD Error
   :align: center
   :width: 45%

.. image:: _static/benchmarks/rsvd_memory.png
   :alt: RSVD Memory
   :align: center
   :width: 45%

.. image:: _static/benchmarks/cqrrpt_runtime_m8192.png
   :alt: CQRRPT Runtime
   :align: center
   :width: 45%

.. image:: _static/benchmarks/cqrrpt_error_m8192.png
   :alt: CQRRPT Error
   :align: center
   :width: 45%

.. image:: _static/benchmarks/cqrrpt_runtime_m16384.png
   :alt: CQRRPT Runtime
   :align: center
   :width: 45%

.. image:: _static/benchmarks/cqrrpt_error_m16384.png
   :alt: CQRRPT Error
   :align: center
   :width: 45%

Benchmarking Your Own Models
-----------------------------

To generate benchmark data for your specific use case, use the plotting notebooks in ``tests/notebooks/``:

- ``plot_linear_bench.ipynb`` - Benchmark SKLinear layers
- ``plot_attention_bench.ipynb`` - Benchmark RandMultiHeadAttention  
- ``plot_conv_bench.ipynb`` - Benchmark SKConv2d layers

Replace the placeholder images above with your generated plots.

For comprehensive benchmarking tools, see :doc:`examples/performance_benchmarks`.

.. note::
   Benchmark results depend heavily on hardware configuration. Run your own benchmarks to get accurate performance data for your specific setup.
