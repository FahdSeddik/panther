Changelog
=========

All notable changes to Panther will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_, and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

**Added**

- Comprehensive documentation with tutorials and examples
- API reference documentation for all modules
- Installation guide with multiple installation methods
- Quickstart guide for new users

**Changed**

- Improved documentation structure and navigation
- Enhanced code examples throughout documentation

**Fixed**

- Documentation formatting and cross-references

[0.1.2] - 2024-12-05
--------------------

**Added**

- SKLinear: Sketched linear layers with GPU acceleration
- SKConv2d: Sketched 2D convolution layers
- RandMultiHeadAttention: Randomized multi-head attention
- CQRRPT: Randomized QR decomposition with column pivoting
- Randomized SVD: Fast approximate singular value decomposition
- Dense and sparse sketching operators
- SRHT (Subsampled Randomized Hadamard Transform)
- AutoTuner: Bayesian optimization for hyperparameter tuning
- CUDA kernels with Tensor Core support
- Mixed precision training support
- Triton kernel implementations (experimental)

**Core Features**

- Memory-efficient neural network layers
- GPU acceleration with CUDA
- Automatic parameter optimization
- PyTorch integration
- Windows and Linux support

**Performance**

- Up to 90% memory reduction for large linear layers
- Competitive or improved speed on large matrices
- Tensor Core optimization for modern GPUs

**Documentation**

- Basic API documentation
- README with installation instructions
- Example notebooks

[0.1.1] - 2024-11-15
--------------------

**Added**

- Initial implementation of sketched linear layers
- Basic CUDA kernels for linear operations
- Python packaging with Poetry
- Core sketching algorithms

**Fixed**

- Build system improvements
- Memory leaks in CUDA kernels

[0.1.0] - 2024-10-30
--------------------

**Added**

- Initial project structure
- Basic linear algebra operations
- C++/CUDA backend (pawX)
- Python bindings
- Build system setup

**Project Structure**

- Core Panther Python package
- Native pawX backend
- Test suite
- Documentation framework

Migration Guide
===============

From 0.1.1 to 0.1.2
-------------------

**API Changes**

No breaking changes. All existing code should work without modification.

**New Features**

If upgrading from 0.1.1, you can now use:

.. code-block:: python

   # New sketched convolution layers
   import panther as pr
   conv = pr.nn.SKConv2d(64, 128, kernel_size=3, num_terms=4, low_rank=16)
   
   # New attention mechanisms
   attention = pr.nn.RandMultiHeadAttention(512, num_heads=8, sketch_dim=64)
   
   # New AutoTuner
   from panther.tuner import SkAutoTuner
   tuner = SkAutoTuner(parameter_bounds={'num_terms': (2, 16)}, ...)

**Installation Changes**

The installation process has been simplified:

.. code-block:: bash

   # Old method (still works)
   git clone ... && cd panther && poetry install && cd pawX && make all
   
   # New method (recommended)
   pip install panther-ml==0.1.2 --extra-index-url https://download.pytorch.org/whl/cu124

Known Issues
============

Current Limitations
------------------

**CUDA Support**

- Requires CUDA 12.4+ for optimal performance
- Tensor Core optimizations require dimensions to be multiples of 16
- Some operations fall back to CPU on older GPUs

**Memory Usage**

- Initial memory allocation may be higher than expected due to CUDA context
- Very large sketching operations may exceed GPU memory limits

**Platform Support**

- macOS support is experimental (CPU only)
- ARM64 support is not yet available
- Some Windows configurations may require manual CUDA setup

**Performance**

- Small layers (< 512 parameters) may be slower than standard layers
- First run may be slower due to CUDA kernel compilation
- Memory benefits are most apparent for large layers

Planned Features
===============

**Version 0.2.0 (Planned Q1 2025)**

- Support for additional PyTorch layers (LSTM, GRU, Transformer)
- Improved AutoTuner with multi-objective optimization
- Better memory management and garbage collection
- ARM64 and Apple Silicon support
- Distributed training support

**Version 0.3.0 (Planned Q2 2025)**

- Dynamic sketching parameters during training
- Integration with Hugging Face Transformers
- Advanced sketching methods (CountSketch, FJLT)
- Compression utilities for model deployment
- Quantization support

**Version 1.0.0 (Planned Q3 2025)**

- Stable API with backward compatibility guarantees
- Production-ready performance optimizations
- Comprehensive benchmarking suite
- Integration with major ML frameworks
- Enterprise support options

Contributing
============

We welcome contributions! See our `Contributing Guide <contributing.html>`_ for details on:

- Setting up a development environment
- Code style and testing requirements
- Submitting pull requests
- Reporting issues

**Priority Areas for Contribution**

1. **Performance Optimization**: CUDA kernel improvements, memory optimization
2. **New Algorithms**: Additional sketching methods, randomized algorithms
3. **Framework Integration**: TensorFlow, JAX, Hugging Face compatibility
4. **Documentation**: Tutorials, examples, API improvements
5. **Testing**: Cross-platform testing, performance benchmarks

Acknowledgments
===============

**Core Development Team**

- The Panther Team

**Special Thanks**

- PyTorch team for the excellent framework
- Contributors to randomized numerical linear algebra research
- Open source community for feedback and contributions

**Research Citations**

Panther implements algorithms from several research papers. See our `References <references.html>`_ section for detailed citations.

**Third-Party Libraries**

Panther builds upon these excellent open source projects:

- PyTorch: Deep learning framework
- OpenBLAS: Optimized linear algebra routines  
- CUDA: GPU computing platform
- Triton: GPU kernel language
- BoTorch: Bayesian optimization
- Poetry: Python dependency management
- Sphinx: Documentation generation

Support
=======

**Getting Help**

- `GitHub Issues <https://github.com/FahdSeddik/panther/issues>`_: Bug reports and feature requests
- `GitHub Discussions <https://github.com/FahdSeddik/panther/discussions>`_: Questions and community support
- `Documentation <https://panther.readthedocs.io>`_: Comprehensive guides and API reference

**Commercial Support**

For enterprise users requiring:

- Priority support and bug fixes
- Custom feature development
- Performance optimization consulting
- Training and integration assistance

Please contact the development team through GitHub.

License
=======

Panther is released under the MIT License. See `LICENSE <https://github.com/FahdSeddik/panther/blob/master/LICENSE>`_ for details.
