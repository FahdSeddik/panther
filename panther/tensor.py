from panther.backends import Backend  # Auto-selects backend


class Tensor:
    """Unified tensor abstraction for different backends (NumPy, Torch, JAX)."""

    def __init__(self, data, dtype=None, device=None):
        """
        Initializes a PantherTensor.

        Args:
            data: List, NumPy array, Torch tensor, or JAX array.
            dtype: Desired dtype (optional).
            device: Desired device ('cpu', 'cuda', etc.).
        """
        self.backend = Backend  # Use auto-selected backend
        self.tensor = self.backend.as_tensor(data)

        if dtype:
            self.tensor = self.backend.astype(self.tensor, dtype)
        if device:
            self.tensor = self.backend.to_device(self.tensor, device)

    def to_numpy(self):
        """Converts to NumPy array."""
        return self.backend.to_numpy(self.tensor)

    def to_device(self, device):
        """Moves tensor to a different device (if supported)."""
        self.tensor = self.backend.to_device(self.tensor, device)
        return self

    def matmul(self, other):
        """Performs matrix multiplication."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(self.backend.matmul(self.tensor, other.tensor))

    def __repr__(self):
        return f"PantherTensor({self.tensor})"

    def __getitem__(self, key):
        return Tensor(self.tensor[key])

    def __setitem__(self, key, value):
        self.tensor[key] = value

    def __add__(self, other):
        return Tensor(self.tensor + other.tensor)

    def __sub__(self, other):
        return Tensor(self.tensor - other.tensor)

    def __mul__(self, other):
        return Tensor(self.tensor * other.tensor)

    def __truediv__(self, other):
        return Tensor(self.tensor / other.tensor)
