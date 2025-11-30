Roadmap
=======

This document outlines potential future directions for Panther development. These are aspirational goals and not commitments to specific timelines or features.

Current Status (v0.1.2)
-----------------------

**Available Features**

- ✅ Core sketching algorithms (CQRRPT, Randomized SVD)
- ✅ SKLinear layer implementation
- ✅ SKConv2d layer implementation
- ✅ RandMultiHeadAttention mechanism
- ✅ CUDA kernel support
- ✅ PyTorch integration
- ✅ C++/CUDA backend (pawX)
- ✅ SKAutoTuner with multiple search algorithms
- ✅ Documentation and examples

**Current Limitations**

- Single-node training only
- Limited to implemented layer types
- CUDA 12.4+ required for GPU acceleration (CPU-only mode fully supported)
- Windows and Linux fully supported (macOS experimental, CPU-only)

Potential Future Directions
---------------------------

The following are areas being considered for future development. Community contributions and feedback will help shape priorities.

**Enhanced Neural Network Support**

- Additional sketched layer types (LSTM, GRU, etc.)
- More attention mechanism variants
- Additional convolution operation types
- Transformer block optimizations

**Performance Optimizations**

- Enhanced CUDA kernel implementations
- Dynamic sketching with adaptive parameters
- Memory usage optimizations
- Support for additional GPU architectures

**Platform and Framework Support**

- Improved macOS support
- ARM64 architecture support  
- Potential integration with other frameworks
- Multi-GPU training support

**Developer Tools**

- Enhanced debugging and profiling utilities
- Additional AutoTuner optimization strategies
- Better performance analysis tools
- Expanded documentation and examples

**Algorithm Improvements**

- Additional sketching methods
- Hierarchical sketching approaches
- Task-specific optimization strategies
- Improved accuracy-performance tradeoffs

Contributing
============

Community contributions are welcome! If you're interested in working on any of these areas or have other ideas, please:

- Open an issue on GitHub to discuss your proposal
- Check existing issues for areas where help is needed
- Submit pull requests with improvements

The actual development path will depend on:

- Community needs and feedback
- Contributor interests and availability
- Technical feasibility
- Resource availability

For questions or suggestions about future directions, please open an issue on the GitHub repository.
