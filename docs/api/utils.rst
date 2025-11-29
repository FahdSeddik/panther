API Reference: Utilities
========================

This page documents Panther's utility functions and helper classes.

.. currentmodule:: panther.utils

Compatibility Utilities
------------------------

The :mod:`panther.utils.compatibility` module provides utilities to check system compatibility.

.. py:function:: has_tensor_core_support()

   Checks if the current CUDA device supports Tensor Cores.

   :return: True if the device has Tensor Core support (compute capability 7.0 or higher), False otherwise.
   :rtype: bool

   Example:

   .. code-block:: python

      from panther.utils.compatibility import has_tensor_core_support
      
      if has_tensor_core_support():
          print("Tensor Cores available - optimizations enabled")
      else:
          print("Tensor Cores not available")

.. note::
   Additional utility functions are under development and will be documented in future releases.
