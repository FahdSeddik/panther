from typing import Optional, Type

from .attention import RandMultiHeadAttention
from .conv2d import SKConv2d
from .linear import SKLinear

# Try to import Triton-based implementations
# This will fail on CPU-only systems or systems without CUDA drivers
SKLinear_triton: Optional[Type] = None
try:
    import triton  # type: ignore # noqa: F401

    # Check if triton can actually be used (has active drivers)
    # This prevents runtime errors on CPU-only systems
    if hasattr(triton.runtime, "driver") and hasattr(triton.runtime.driver, "active"):
        # Try to access the driver to trigger any initialization errors
        _ = triton.runtime.driver.active
        from .linear_tr import SKLinear_triton  # type: ignore
except (ImportError, RuntimeError, AttributeError):
    # ImportError: triton not installed
    # RuntimeError: triton installed but no CUDA drivers (0 active drivers)
    # AttributeError: triton API changed
    pass

__all__ = ["SKLinear", "RandMultiHeadAttention", "SKConv2d", "SKLinear_triton"]
