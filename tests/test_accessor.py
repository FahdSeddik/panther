import pytest
import torch

from pawX import test_tensor_accessor as tacc

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


def test_accessor(input_tensor):
    """Test the tensor accessor functionality with different tensor types and shapes."""
    tacc(input_tensor)


if __name__ == "__main__":
    pytest.main([__file__])
