from torch.cuda import get_device_capability, is_available


class PadMixin:
    PAD_SIZE = 16

    def __init__(self, *args, **kwargs):
        # Determine once whether we want to pad matrices
        super().__init__(*args, **kwargs)
        if is_available():
            major, minor = get_device_capability()
            self.pad_matrices = (major > 7) or (major == 7 and minor >= 0)
        else:
            self.pad_matrices = False

    def pad_dim(self, dim: int) -> int:
        """
        If padding is enabled, round `dim` up to the nearest multiple of PAD_SIZE.
        Otherwise, return `dim` unchanged.
        """
        if not self.pad_matrices:
            return dim
        return ((dim + self.PAD_SIZE - 1) // self.PAD_SIZE) * self.PAD_SIZE
