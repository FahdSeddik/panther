from abc import ABC, abstractmethod


class BaseBackend(ABC):
    """
    Abstract base class for Panther backends.
    Ensures a unified API for NumPy, PyTorch, JAX, and TensorFlow.
    """

    @abstractmethod
    def as_tensor(self, data):
        pass

    @abstractmethod
    def to_numpy(self, tensor):
        pass

    @abstractmethod
    def to_device(self, tensor, device):
        pass

    # --- Core Tensor Operations ---
    @abstractmethod
    def matmul(self, A, B):
        pass

    @abstractmethod
    def dot(self, A, B):
        pass

    @abstractmethod
    def tensordot(self, A, B, axes):
        pass

    @abstractmethod
    def einsum(self, eq, *tensors):
        pass

    # --- Linear Algebra ---
    @abstractmethod
    def cholesky(self, A):
        pass

    @abstractmethod
    def qr(self, A):
        pass

    @abstractmethod
    def svd(self, A):
        pass

    @abstractmethod
    def eig(self, A):
        pass

    @abstractmethod
    def solve(self, A, b):
        pass

    @abstractmethod
    def lstsq(self, A, b):
        pass

    @abstractmethod
    def inv(self, A):
        pass

    @abstractmethod
    def pinv(self, A):
        pass

    # --- Random Matrices ---
    @abstractmethod
    def randn(self, *shape):
        pass

    @abstractmethod
    def uniform(self, *shape):
        pass

    # --- Special Constants ---
    @property
    @abstractmethod
    def pi(self):
        pass

    @property
    @abstractmethod
    def e(self):
        pass

    # --- Tensor Manipulation ---
    @abstractmethod
    def reshape(self, A, shape):
        pass

    @abstractmethod
    def ravel(self, A):
        pass

    @abstractmethod
    def pad(self, A, pad_width):
        pass

    # --- Neural Network Operations (For Keras-like API) ---
    @abstractmethod
    def relu(self, x):
        pass

    @abstractmethod
    def softmax(self, x):
        pass

    # --- JIT & AutoDiff (JAX & Torch-specific) ---
    @abstractmethod
    def grad(self, func):
        pass

    @abstractmethod
    def jit(self, func):
        pass

    # --- Device Info ---
    @abstractmethod
    def device(self):
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.device()}>"
