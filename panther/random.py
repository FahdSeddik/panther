import torch

# DISCLAIMER: THIS FILE NEEDS TO BE CHECKED FOR CORRECTNESS


def uniform_dense_sketch(m, n, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    return torch.empty(m, n, **factory_kwargs).uniform_(-1, 1)


def gaussian_dense_sketch(m, n, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    return torch.randn(m, n, **factory_kwargs)


def hadamard_sketch(m, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    if m & (m - 1) != 0:
        raise ValueError("m must be a power of 2")

    H = torch.tensor([[1.0]])
    while H.shape[0] < m:
        H = torch.cat((torch.cat((H, H), dim=1), torch.cat((H, -H), dim=1)), dim=0)

    return H / torch.sqrt(torch.tensor(m, **factory_kwargs))


def gaussian_orthonormal_sketch(m, n, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    return torch.qr(torch.randn(m, n, **factory_kwargs))[0]


def clarkson_woodruff_sketch(m, n, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    indices = torch.randint(0, m, (n,), **factory_kwargs)
    signs = torch.randint(0, 2, (n,), **factory_kwargs) * 2 - 1
    sketch = torch.zeros(m, n, **factory_kwargs)
    sketch[indices, torch.arange(n)] = signs
    return sketch


def sparse_sign_embeddings_sketch(m, n, sparsity=0.1):
    mask = torch.rand(m, n) < sparsity
    signs = torch.randint(0, 2, (m, n)) * 2 - 1
    return mask.float() * signs.float()
