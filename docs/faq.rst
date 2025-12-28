Frequently Asked Questions
===========================

This page answers common questions about Panther and its sketching algorithms.

General Questions
-----------------

**What is Panther?**

Panther is a high-performance Python library for randomized numerical linear algebra (RandNLA) and sketching algorithms. It provides memory-efficient alternatives to standard neural network operations while maintaining accuracy and enabling faster training.

**What does "sketching" mean in this context?**

Sketching refers to randomized dimensionality reduction techniques that compress data while preserving important properties. In neural networks, sketching can reduce the memory footprint of linear layers by using smaller, compressed representations of weight matrices.

**How does Panther relate to PyTorch?**

Panther is built on top of PyTorch and integrates seamlessly with existing PyTorch workflows. You can replace standard PyTorch layers (like ``torch.nn.Linear``) with Panther's sketched equivalents (like ``panther.nn.SKLinear``) without changing your training code.

**Is Panther only for research, or can it be used in production?**

Panther is designed for both research and production use. While it includes cutting-edge research algorithms, it also provides stable, well-tested implementations suitable for production deployments.

Installation and Setup
-----------------------

**What are the system requirements?**

- Python 3.12 or later
- PyTorch 2.6.0 or later
- CUDA 12.4 or later (for GPU acceleration)
- C++17 compatible compiler
- 8GB+ RAM recommended

**Do I need a GPU to use Panther?**

No, Panther works on both CPU and GPU. However, GPU acceleration provides significant performance improvements, especially for large models and datasets.

**How do I install Panther?**

.. code-block:: bash

   # Install from PyPI (recommended)
   pip install panther-ml
   
   # Or install from source
   git clone https://github.com/FahdSeddik/panther.git
   cd panther
   pip install -e .

**I'm getting compilation errors during installation. What should I do?**

1. Ensure you have a C++17 compatible compiler installed
2. Check that CUDA toolkit version matches your PyTorch installation
3. Try installing without CUDA support: ``PANTHER_BUILD_CUDA=0 pip install panther-ml``
4. See our :doc:`installation` guide for detailed troubleshooting

Usage Questions
---------------

**How do I replace a standard Linear layer with SKLinear?**

.. code-block:: python

   # Before
   import torch.nn as nn
   layer = nn.Linear(1024, 512)
   
   # After
   import panther as pr
   layer = pr.nn.SKLinear(1024, 512, num_terms=4, low_rank=64)

**What num_terms and low_rank should I use?**

A good starting point is num_terms=4 and low_rank to be 10-20% of min(in_features, out_features). You can use the AutoTuner to find optimal parameters:

.. code-block:: python

   from panther.tuner import SKAutoTuner, LayerConfig, TuningConfigs
   
   config = LayerConfig(
       layer_names=["layer1", "layer2"],
       params={
           "num_terms": [1, 2, 4, 8],
           "low_rank": [32, 64, 128]
       }
   )
   tuner = SKAutoTuner(model, TuningConfigs([config]), accuracy_eval_func)
   best_params = tuner.tune()

**Can I use Panther with existing pre-trained models?**

Yes, but you'll need to fine-tune the model after replacing layers with sketched versions. The sketched layers are not directly compatible with pre-trained weights.

**How much memory reduction can I expect?**

Memory reduction depends on the sketch ratio:
- 50% sketch ratio: ~25-40% memory reduction
- 30% sketch ratio: ~40-60% memory reduction
- Trade-off between memory savings and accuracy loss

**Does sketching affect model accuracy?**

Sketching introduces a small approximation error. With proper sketch size selection:
- Accuracy loss is typically <1-3%
- Sometimes accuracy can even improve due to regularization effects
- Use the AutoTuner to find the best accuracy-memory trade-off

Performance Questions
---------------------

**Is Panther always faster than standard PyTorch?**

Not always. Panther provides benefits for:
- Large linear layers (>1000 features)
- Memory-constrained environments
- Models with many parameters

For small models or layers, the overhead might outweigh benefits.

**How does Panther compare to other compression methods?**

.. list-table:: Comparison with Other Methods
   :header-rows: 1
   :widths: 25 25 25 25

   * - Method
     - Memory Reduction
     - Speed
     - Accuracy Preservation
   * - Pruning
     - 50-90%
     - Moderate
     - Good
   * - Quantization
     - 50-75%
     - Fast
     - Good
   * - Knowledge Distillation
     - Variable
     - Fast
     - Excellent
   * - **Panther Sketching**
     - **25-60%**
     - **Fast**
     - **Very Good**

**Can I combine Panther with other optimization techniques?**

Yes! Panther works well with:
- Mixed precision training (AMP)
- Gradient checkpointing
- Model pruning (apply after sketching)
- Quantization (on sketched models)

Technical Questions
-------------------

**What algorithms does Panther implement?**

- **CQRRPT**: CholeskyQR with Randomized column Pivoting for tall matrices
- **RSVD**: Randomized singular value decomposition
- **Gaussian sketching**: Random Gaussian projection matrices
- **SRHT**: Subsampled randomized Hadamard transform
- **CountSketch**: Sparse sketching with random hashing
- **SJLT**: Sparse Johnson-Lindenstrauss Transform

**How does the sketching work mathematically?**

For an input :math:`x \in \mathbb{R}^n` and weight matrix :math:`W \in \mathbb{R}^{m \times n}`:

1. Standard linear layer: :math:`y = Wx`
2. Sketched layer: :math:`y = W'S x` where :math:`S \in \mathbb{R}^{k \times n}` is the sketch matrix and :math:`W' \in \mathbb{R}^{m \times k}`

The sketch preserves important properties while using less memory.

**Are the random matrices truly random every time?**

No, sketch matrices are initialized once and remain fixed during training. This ensures:
- Consistent gradients during backpropagation
- Reproducible results with fixed random seeds
- Stable training dynamics

**Can I inspect or modify the sketch matrices?**

Yes, you can access sketch parameters:

.. code-block:: python

   layer = pr.nn.SKLinear(1024, 512, num_terms=4, low_rank=64)
   
   # Access sketch parameters (S1s and S2s)
   print(f"S1s shape: {layer.S1s.shape}")
   print(f"S2s shape: {layer.S2s.shape}")
   print(f"U1s shape: {layer.U1s.shape}")
   print(f"U2s shape: {layer.U2s.shape}")

Troubleshooting
---------------

**My model's accuracy dropped significantly after using sketching. What should I do?**

1. Increase sketch sizes (try 70-80% of input dimension)
2. Use the AutoTuner to find optimal parameters
3. Apply sketching gradually (start with later layers)
4. Check if your learning rate needs adjustment
5. Consider using different sketch types for different layers

**Training is slower with Panther. Why?**

Possible causes:
- Sketch sizes too large (reduce sketch ratio)
- Inefficient GPU utilization (check batch sizes)
- CPU bottlenecks (ensure data loading is optimized)
- Mixed CPU/GPU operations (keep everything on GPU)

**I'm getting out-of-memory errors even with sketching.**

1. Reduce batch size
2. Use gradient checkpointing
3. Enable mixed precision training
4. Consider using smaller sketch ratios
5. Use data-parallel training across multiple GPUs

**Results are not reproducible between runs.**

Ensure you set random seeds:

.. code-block:: python

   import torch
   import numpy as np
   import random
   
   # Set all random seeds
   torch.manual_seed(42)
   np.random.seed(42)
   random.seed(42)
   
   if torch.cuda.is_available():
       torch.cuda.manual_seed(42)
       torch.backends.cudnn.deterministic = True

Advanced Questions
------------------

**How do I contribute to Panther?**

Contributions are welcome! To contribute:
- Fork the repository on GitHub
- Create a feature branch with your changes
- Add tests for new functionality
- Submit a pull request

Check the repository for detailed development setup instructions.

**How do I cite Panther in academic work?**

.. code-block:: bibtex

   @software{panther2025,
     title={Panther: High-Performance Randomized Numerical Linear Algebra for PyTorch},
     author={Panther Development Team},
     year={2025},
     url={https://github.com/FahdSeddik/panther},
     version={0.1.2}
   }

Still Have Questions?
---------------------

If your question isn't answered here:

1. **Search GitHub Issues**: Check if someone has asked the same question
2. **Open a GitHub Issue**: Report bugs or request features
3. **Check the Documentation**: Browse the full documentation for detailed information

**Community Resources**

- GitHub Repository: https://github.com/FahdSeddik/panther
- GitHub Issues: For bug reports and feature requests

We appreciate feedback and contributions from the community!
