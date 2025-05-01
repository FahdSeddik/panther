from torch.cuda import get_device_capability, is_available


def has_tensor_core_support() -> bool:
    if not is_available():
        return False
    major, minor = get_device_capability()
    return (major > 7) or (major == 7 and minor >= 0)
