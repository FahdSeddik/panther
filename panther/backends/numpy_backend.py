import numpy as np

from .base import BaseBackend


class NumpyBackend(BaseBackend):
    def as_tensor(self, data):
        return np.array(data)

    def to_numpy(self, tensor):
        return np.asarray(tensor)

    def to_device(self, tensor, device):
        return tensor  # NumPy does not have explicit device management

    def matmul(self, A, B):
        return np.matmul(A, B)

    def dot(self, A, B):
        return np.dot(A, B)

    def tensordot(self, A, B, axes):
        return np.tensordot(A, B, axes)

    def einsum(self, eq, *tensors):
        return np.einsum(eq, *tensors)

    def cholesky(self, A):
        return np.linalg.cholesky(A)

    def qr(self, A):
        return np.linalg.qr(A)

    def svd(self, A):
        return np.linalg.svd(A, full_matrices=False)

    def eig(self, A):
        return np.linalg.eig(A)

    def solve(self, A, b):
        return np.linalg.solve(A, b)

    def lstsq(self, A, b):
        return np.linalg.lstsq(A, b, rcond=None)[0]

    def inv(self, A):
        return np.linalg.inv(A)

    def pinv(self, A):
        return np.linalg.pinv(A)

    def randn(self, *shape):
        return np.random.randn(*shape)

    def uniform(self, *shape):
        return np.random.uniform(size=shape)

    @property
    def pi(self):
        return np.pi

    @property
    def e(self):
        return np.e

    def reshape(self, A, shape):
        return np.reshape(A, shape)

    def ravel(self, A):
        return np.ravel(A)

    def pad(self, A, pad_width):
        return np.pad(A, pad_width)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def grad(self, func):
        raise NotImplementedError("NumPy does not support automatic differentiation.")

    def jit(self, func):
        return func  # No JIT in NumPy, return function as is

    def device(self):
        return "CPU"


Backend = NumpyBackend()
