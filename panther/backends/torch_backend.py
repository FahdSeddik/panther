import torch

from .base import BaseBackend


class TorchBackend(BaseBackend):
    def __init__(self):
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"

    def as_tensor(self, data):
        """Ensures data is a PyTorch tensor on the correct device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device_type)

        if hasattr(data, "tolist"):  # Works for NumPy, JAX, lists
            data = data.tolist()

        return torch.tensor(data, device=self.device_type)

    def to_device(self, tensor, device):
        """Moves a tensor to the specified device."""
        return tensor.to(device)

    def to_numpy(self, tensor):
        """Converts a PyTorch tensor to a NumPy array (moves to CPU first)."""
        return tensor.cpu().detach().numpy()

    # --- Core Tensor Operations ---
    def matmul(self, A, B):
        return torch.matmul(A, B)

    def dot(self, A, B):
        return torch.dot(A, B)

    def tensordot(self, A, B, axes):
        return torch.tensordot(A, B, dims=axes)

    def einsum(self, eq, *tensors):
        return torch.einsum(eq, *tensors)

    # --- Linear Algebra ---
    def cholesky(self, A):
        return torch.linalg.cholesky(A)

    def qr(self, A):
        return torch.linalg.qr(A)

    def svd(self, A):
        return torch.linalg.svd(A)

    def eig(self, A):
        return torch.linalg.eig(A)

    def solve(self, A, b):
        return torch.linalg.solve(A, b)

    def lstsq(self, A, b):
        return torch.linalg.lstsq(A, b).solution

    def inv(self, A):
        return torch.linalg.inv(A)

    def pinv(self, A):
        return torch.linalg.pinv(A)

    # --- Random Matrices ---
    def randn(self, *shape):
        return torch.randn(*shape, device=self.device_type)

    def uniform(self, *shape):
        return torch.rand(*shape, device=self.device_type)

    # --- Special Constants ---
    @property
    def pi(self):
        return torch.pi

    @property
    def e(self):
        return torch.e

    # --- Tensor Manipulation ---
    def reshape(self, A, shape):
        return A.reshape(shape)

    def ravel(self, A):
        return A.view(-1)

    def pad(self, A, pad_width):
        return torch.nn.functional.pad(A, pad_width)

    # --- Neural Network Operations ---
    def relu(self, x):
        return torch.relu(x)

    def softmax(self, x):
        return torch.softmax(x, dim=-1)

    # --- JIT & AutoDiff ---
    def grad(self, func):
        return torch.autograd.functional.jacobian(func)

    def jit(self, func):
        return torch.jit.script(func)

    # --- Device Info ---
    def device(self):
        return self.device_type


Backend = TorchBackend()
