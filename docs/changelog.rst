Changelog
=========

This is the initial release of Panther. As the library evolves, this document will track changes between versions.

Current Version (0.1.2)
-----------------------

**Core Features**

- **SKLinear**: Sketched linear layers with GPU acceleration
- **SKConv2d**: Sketched 2D convolution layers  
- **RandMultiHeadAttention**: Randomized multi-head attention mechanism
- **CQRRPT**: Randomized QR decomposition with column pivoting
- **Randomized SVD**: Fast approximate singular value decomposition
- **SKAutoTuner**: Hyperparameter tuning with multiple search algorithms (Grid Search, Random Search, Bayesian Optimization, etc.)
- **CUDA Backend**: GPU-accelerated operations via pawX C++/CUDA extension
- **PyTorch Integration**: Drop-in replacements for standard PyTorch layers

**Platform Support**

- Windows and Linux support (CPU and GPU)
- CPU-only operation fully supported (requires building from source)
- CUDA 12.4+ for optional GPU acceleration
- Python 3.12+

Current Limitations
===================

**CUDA Support**

- CUDA 12.4+ required for GPU operations (optional)
- All core features available in CPU-only mode
- Tensor Core optimizations work best with dimensions that are multiples of 16

**Performance**

- Small layers (< 512 parameters) may be slower than standard layers
- First run may be slower due to CUDA kernel compilation
- Memory benefits are most noticeable for large layers

Contributing
============

Contributions are welcome! To contribute:

1. Fork the repository on GitHub
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

See the repository for more details on development setup and guidelines.

Acknowledgments
===============

Panther builds upon excellent open source projects:

- **PyTorch**: Deep learning framework
- **OpenBLAS**: Optimized linear algebra routines  
- **CUDA**: GPU computing platform
- **Optuna**: Bayesian optimization
- **Poetry**: Python dependency management

License
=======

Panther is released under the MIT License. See the LICENSE file in the repository for details.
