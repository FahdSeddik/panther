import os

from .base import BaseBackend

# from .jax_backend import JaxBackend
from .numpy_backend import NumpyBackend

# Import available backends
from .torch_backend import TorchBackend

Backend: BaseBackend

# Detect preferred backend
PREFERRED_BACKEND = os.getenv("PANTHER_BACKEND", "auto").lower()

if PREFERRED_BACKEND == "torch":
    Backend = TorchBackend()
# elif PREFERRED_BACKEND == "jax":
#     Backend = JaxBackend()
elif PREFERRED_BACKEND == "numpy":
    Backend = NumpyBackend()
elif PREFERRED_BACKEND == "auto":
    # Auto-detect based on availability (Torch > JAX > NumPy)
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print("CUDA is available.")
        Backend = TorchBackend()
    except ImportError:
        Backend = NumpyBackend()

else:
    raise ValueError(f"Unknown backend: {PREFERRED_BACKEND}")

print(f"Using Panther Backend: {Backend}")
