from torch.cuda import get_device_capability, is_available


def has_tensor_core_support() -> bool:
    """
    Checks if the current CUDA device supports Tensor Cores.

    Returns:
        bool: True if the device has Tensor Core support (compute capability 7.0 or higher), False otherwise.
    """
    if not is_available():
        return False
    major, minor = get_device_capability()
    return (major > 7) or (major == 7 and minor >= 0)
