API Reference: Utilities
=======================

This page documents Panther's utility functions and helper classes.

.. currentmodule:: panther.utils

Compatibility Utilities
------------------------

.. autofunction:: check_pytorch_version

.. autofunction:: check_cuda_availability

.. autofunction:: get_device_capability

.. autofunction:: ensure_contiguous

Tensor Utilities
----------------

.. autofunction:: safe_matmul

.. autofunction:: batch_trace

.. autofunction:: fast_norm

.. autofunction:: orthogonalize

Memory Management
-----------------

.. autofunction:: get_memory_usage

.. autofunction:: clear_cache

.. autofunction:: estimate_memory_requirement

Performance Profiling
----------------------

.. autoclass:: PerformanceProfiler
   :members:

.. autoclass:: MemoryProfiler
   :members:

Numerical Utilities
-------------------

.. autofunction:: condition_number

.. autofunction:: numerical_rank

.. autofunction:: relative_error

.. autofunction:: spectral_norm

Random Number Generation
-------------------------

.. autofunction:: set_random_seed

.. autofunction:: get_random_state

.. autofunction:: with_random_seed

Type Checking
--------------

.. autofunction:: is_tensor

.. autofunction:: is_complex

.. autofunction:: is_cuda_tensor

.. autofunction:: check_shape_compatibility

Configuration
-------------

.. autoclass:: Config
   :members:

.. autofunction:: get_config

.. autofunction:: set_config

.. autofunction:: reset_config
