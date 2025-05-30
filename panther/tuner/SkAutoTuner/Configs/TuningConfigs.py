from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List

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
        """
        Get the number of layer configurations.

        Returns:
            Number of layer configs in this collection
        """
        return len(self.configs)

    def __getitem__(self, index: int) -> LayerConfig:
        """
        Get a specific layer configuration by index.

        Args:
            index: Index of the layer config to retrieve

        Returns:
            The LayerConfig at the specified index
        """
        return self.configs[index]

    def __iter__(self) -> Iterator[LayerConfig]:
        """
        Iterate over all layer configurations.

        Returns:
            Iterator over LayerConfig objects
        """
        return iter(self.configs)

    def add(self, config: LayerConfig) -> "TuningConfigs":
        """
        Create a new TuningConfigs with an additional LayerConfig.

        Args:
            config: LayerConfig to add

        Returns:
            A new TuningConfigs instance with the added config
        """
        return TuningConfigs(self.configs + [config])

    def remove(self, index: int) -> "TuningConfigs":
        """
        Create a new TuningConfigs with a LayerConfig removed.

        Args:
            index: Index of the config to remove

        Returns:
            A new TuningConfigs instance without the specified config

        Raises:
            IndexError: If the index is out of range
        """
        if index < 0 or index >= len(self.configs):
            raise IndexError(f"Config index {index} out of range")

        new_configs = self.configs.copy()
        new_configs.pop(index)
        return TuningConfigs(new_configs)

    def replace(self, index: int, config: LayerConfig) -> "TuningConfigs":
        """
        Create a new TuningConfigs with a LayerConfig replaced.

        Args:
            index: Index of the config to replace
            config: New LayerConfig to use

        Returns:
            A new TuningConfigs with the replaced config

        Raises:
            IndexError: If the index is out of range
        """
        if index < 0 or index >= len(self.configs):
            raise IndexError(f"Config index {index} out of range")

        new_configs = self.configs.copy()
        new_configs[index] = config
        return TuningConfigs(new_configs)

    def clone(self) -> "TuningConfigs":
        """
        Create a deep copy of this TuningConfigs.

        Returns:
            A new TuningConfigs instance with copies of all LayerConfig objects
        """
        return TuningConfigs([config.clone() for config in self.configs])

    def filter(self, predicate: Callable[[LayerConfig], bool]) -> "TuningConfigs":
        """
        Create a new TuningConfigs containing only configs that match a predicate.

        Args:
            predicate: Function that takes a LayerConfig and returns True if it should be included

        Returns:
            A new TuningConfigs with only the matching configs
        """
        return TuningConfigs([config for config in self.configs if predicate(config)])

    def map(self, transform: Callable[[LayerConfig], LayerConfig]) -> "TuningConfigs":
        """
        Apply a transformation to each LayerConfig.

        Args:
            transform: Function that takes a LayerConfig and returns a modified LayerConfig

        Returns:
            A new TuningConfigs with transformed configs
        """
        return TuningConfigs([transform(config) for config in self.configs])

    def merge(self, other: "TuningConfigs") -> "TuningConfigs":
        """
        Merge this TuningConfigs with another one.

        Args:
            other: Another TuningConfigs to merge with

        Returns:
            A new TuningConfigs with configs from both
        """
        return TuningConfigs(self.configs + other.configs)

    def get_total_param_space_size(self) -> int:
        """
        Calculate the total parameter space size across all configurations.

        Returns:
            The sum of parameter space sizes for all configs
        """
        return sum(config.get_param_space_size() for config in self.configs)

    def get_param_usage_stats(self) -> Dict[str, int]:
        """
        Get statistics on how frequently each parameter is used across configs.

        Returns:
            Dictionary mapping parameter names to their frequency of use
        """
        param_counts = defaultdict(int)

        for config in self.configs:
            for param_name in config.params:
                param_counts[param_name] += 1

        return dict(param_counts)

    def get_layers_stats(self) -> Dict[str, int]:
        """
        Analyze how many layer patterns or names are used across configs.

        Returns:
            Dictionary with statistics about layer selection methods
        """
        stats = {
            "string_patterns": 0,
            "list_patterns": 0,
            "dict_selectors": 0,
            "total_layers": 0,
        }

        for config in self.configs:
            layer_names = config.layer_names

            if isinstance(layer_names, str):
                stats["string_patterns"] += 1
                stats["total_layers"] += 1
            elif isinstance(layer_names, list):
                stats["list_patterns"] += 1
                stats["total_layers"] += len(layer_names)
            elif isinstance(layer_names, dict):
                stats["dict_selectors"] += 1
                stats["total_layers"] += 1  # Approximate as 1 per dict selector

        return stats

    def get_configs_by_param(self, param_name: str) -> "TuningConfigs":
        """
        Get all configs that tune a specific parameter.

        Args:
            param_name: Name of the parameter to filter by

        Returns:
            A new TuningConfigs with only configs that tune the specified parameter
        """
        return self.filter(lambda config: param_name in config.params)

    def get_separate_configs(self) -> "TuningConfigs":
        """
        Get all configs where separate=True.

        Returns:
            A new TuningConfigs with only configs that have separate=True
        """
        return self.filter(lambda config: config.separate)

    def get_non_separate_configs(self) -> "TuningConfigs":
        """
        Get all configs where separate=False.

        Returns:
            A new TuningConfigs with only configs that have separate=False
        """
        return self.filter(lambda config: not config.separate)

    def set_all_separate(self, separate: bool) -> "TuningConfigs":
        """
        Create a new TuningConfigs with all configs' separate flag set to a value.

        Args:
            separate: Value to set for the separate flag

        Returns:
            A new TuningConfigs with modified configs
        """

        def transform(config: LayerConfig) -> LayerConfig:
            new_config = config.clone()
            new_config.separate = separate
            return new_config

        return self.map(transform)

    def set_all_copy_weights(self, copy_weights: bool) -> "TuningConfigs":
        """
        Create a new TuningConfigs with all configs' copy_weights flag set to a value.

        Args:
            copy_weights: Value to set for the copy_weights flag

        Returns:
            A new TuningConfigs with modified configs
        """

        def transform(config: LayerConfig) -> LayerConfig:
            new_config = config.clone()
            new_config.copy_weights = copy_weights
            return new_config

        return self.map(transform)

    def add_param_to_all(
        self, param_name: str, param_values: List[Any]
    ) -> "TuningConfigs":
        """
        Add a parameter to all configs.

        Args:
            param_name: Name of the parameter to add
            param_values: Values to try for the parameter

        Returns:
            A new TuningConfigs with the parameter added to all configs
        """
        return self.map(lambda config: config.with_param(param_name, param_values))

    def remove_param_from_all(self, param_name: str) -> "TuningConfigs":
        """
        Remove a parameter from all configs.

        Args:
            param_name: Name of the parameter to remove

        Returns:
            A new TuningConfigs with the parameter removed from all configs
        """
        return self.map(lambda config: config.without_param(param_name))

    def to_dict(self) -> List[Dict[str, Any]]:
        """
        Convert the TuningConfigs to a list of dictionaries for serialization.

        Returns:
            List of dictionaries representing the configs
        """
        result = []

        for config in self.configs:
            config_dict = {
                "layer_names": config.layer_names,
                "params": config.params,
                "separate": config.separate,
                "copy_weights": config.copy_weights,
            }
            result.append(config_dict)

        return result

    @classmethod
    def from_dict(cls, config_dicts: List[Dict[str, Any]]) -> "TuningConfigs":
        """
        Create a TuningConfigs from a list of dictionaries.

        Args:
            config_dicts: List of dictionaries representing configs

        Returns:
            A new TuningConfigs instance
        """
        configs = []

        for config_dict in config_dicts:
            config = LayerConfig(
                layer_names=config_dict["layer_names"],
                params=config_dict["params"],
                separate=config_dict.get("separate", True),
                copy_weights=config_dict.get("copy_weights", True),
            )
            configs.append(config)

        return cls(configs)
