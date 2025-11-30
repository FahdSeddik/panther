import pytest
import torch

try:
    from pawX import test_tensor_accessor as tacc

    HAS_CUDA_ACCESSOR = tacc is not None
except (ImportError, AttributeError):
    HAS_CUDA_ACCESSOR = False
    tacc = None  # type: ignore

CONFIG = [
    ((3, 4), torch.float32),
    ((3, 4), torch.float16),
]


@pytest.fixture(params=CONFIG)
def input_tensor(request):
    """Create a random tensor with the given shape and dtype."""
    torch.manual_seed(42)
    shape, dtype = request.param
    return torch.randn(shape, dtype=dtype)


@pytest.mark.skipif(
    not HAS_CUDA_ACCESSOR, reason="test_tensor_accessor requires CUDA build"
)
def test_accessor(input_tensor):
    """Test the tensor accessor functionality with different tensor types and shapes."""
    tacc(input_tensor)


if __name__ == "__main__":
    pytest.main([__file__])
