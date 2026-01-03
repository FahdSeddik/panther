from typing import Callable, Iterator, List

from .LayerConfig import LayerConfig


class TuningConfigs:
    """
    Collection of LayerConfig objects for tuning multiple layer groups.
    """

    def __init__(self, configs: List[LayerConfig]):
        """
        Initialize with a list of layer configurations.

        Args:
            configs: List of LayerConfig objects
        """
        self.configs = configs

    def __repr__(self):
        return f"TuningConfigs(configs={self.configs})"

    def __len__(self) -> int:
        return len(self.configs)

    def __getitem__(self, index: int) -> LayerConfig:
        return self.configs[index]

    def __iter__(self) -> Iterator[LayerConfig]:
        return iter(self.configs)

    def add(self, config: LayerConfig) -> "TuningConfigs":
        """Add a LayerConfig, returning a new TuningConfigs."""
        return TuningConfigs(self.configs + [config])

    def remove(self, index: int) -> "TuningConfigs":
        """Remove a LayerConfig by index, returning a new TuningConfigs."""
        if index < 0 or index >= len(self.configs):
            raise IndexError(f"Config index {index} out of range")
        new_configs = self.configs.copy()
        new_configs.pop(index)
        return TuningConfigs(new_configs)

    def replace(self, index: int, config: LayerConfig) -> "TuningConfigs":
        """Replace a LayerConfig at index, returning a new TuningConfigs."""
        if index < 0 or index >= len(self.configs):
            raise IndexError(f"Config index {index} out of range")
        new_configs = self.configs.copy()
        new_configs[index] = config
        return TuningConfigs(new_configs)

    def clone(self) -> "TuningConfigs":
        """Create a deep copy of this TuningConfigs."""
        return TuningConfigs([config.clone() for config in self.configs])

    def merge(self, other: "TuningConfigs") -> "TuningConfigs":
        """Merge with another TuningConfigs, returning a new TuningConfigs."""
        return TuningConfigs(self.configs + other.configs)

    def filter(self, predicate: Callable[[LayerConfig], bool]) -> "TuningConfigs":
        """Filter configs by predicate, returning a new TuningConfigs."""
        return TuningConfigs([config for config in self.configs if predicate(config)])

    def map(self, transform: Callable[[LayerConfig], LayerConfig]) -> "TuningConfigs":
        """Apply a transformation to each config, returning a new TuningConfigs."""
        return TuningConfigs([transform(config) for config in self.configs])
