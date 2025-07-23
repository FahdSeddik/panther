API Reference: Utilities
========================

This page documents Panther's utility functions and helper classes.

.. note::
   Utility functions are currently under development. Detailed documentation will be available in future releases.

.. currentmodule:: panther.utils

Compatibility Utilities
------------------------

These functions help check system compatibility and requirements.

* ``check_pytorch_version()`` - Verify PyTorch version compatibility
* ``check_cuda_availability()`` - Check if CUDA is available
* ``get_device_capability()`` - Get GPU compute capability
* ``ensure_contiguous()`` - Ensure tensor memory layout

Tensor Utilities
----------------

High-performance tensor operations and utilities.

* ``safe_matmul()`` - Memory-safe matrix multiplication
* ``batch_trace()`` - Efficient batch trace computation
* ``fast_norm()`` - Optimized norm calculations
* ``orthogonalize()`` - Tensor orthogonalization

Memory Management
-----------------

Tools for monitoring and optimizing memory usage.

* ``get_memory_usage()`` - Get current memory statistics
* ``clear_cache()`` - Clear GPU memory cache
* ``estimate_memory_requirement()`` - Estimate operation memory needs

Performance Profiling
----------------------

Profiling tools for performance analysis.

* ``PerformanceProfiler`` - Profile computation performance
* ``MemoryProfiler`` - Profile memory usage patterns

Numerical Utilities
-------------------

Mathematical utilities for numerical analysis.

* ``condition_number()`` - Compute matrix condition number
* ``numerical_rank()`` - Estimate numerical rank
* ``relative_error()`` - Calculate relative error metrics
* ``spectral_norm()`` - Compute spectral norm

Random Number Generation
-------------------------

Utilities for managing random number generation.

* ``set_random_seed()`` - Set global random seed
* ``get_random_state()`` - Get current random state
* ``with_random_seed()`` - Context manager for random seeds

Type Checking
--------------

Type checking and validation utilities.

* ``is_tensor()`` - Check if object is a tensor
* ``is_complex()`` - Check for complex data types
* ``is_cuda_tensor()`` - Check if tensor is on GPU
* ``check_shape_compatibility()`` - Validate tensor shapes

Configuration
-------------

Configuration management utilities.

* ``Config`` - Configuration management class
* ``get_config()`` - Get current configuration
* ``set_config()`` - Update configuration
* ``reset_config()`` - Reset to default configuration
