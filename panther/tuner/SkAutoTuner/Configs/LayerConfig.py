import copy as copy_module
from typing import Dict, List, Union

from .ParamSpec import ParamSpec


class LayerConfig:
    """
    Configuration for a single layer or group of layers to tune.

    Attributes:
        layer_names: Layer selector (string, list, or dict with criteria)
        params: Dictionary of parameter names to ParamSpec or list of values
        separate: Whether to tune each layer independently
        copy_weights: Whether to copy weights when replacing layers
    """

    def __init__(
        self,
        layer_names: Union[
            str, List[str], Dict[str, Union[str, List[str], int, List[int]]]
        ],
        params: Dict[str, ParamSpec],
        separate: bool = True,
        copy_weights: bool = True,
    ):
        """
        Initialize a layer configuration.

        Args:
            layer_names: Layer selector, can be:
                - A string: Regex pattern or substring
                - A list of strings: Multiple patterns or exact names
                - A dictionary with selection criteria (pattern, type, contains, indices, range)
            params: Dictionary of parameter names and their possible values
            separate: Whether to tune layers separately or jointly
            copy_weights: Whether to copy weights when replacing layers
        """
        self.layer_names = layer_names
        self.params = params
        self.separate = separate
        self.copy_weights = copy_weights

    def __repr__(self):
        return f"LayerConfig(layer_names={self.layer_names}, params={self.params}, separate={self.separate}, copy_weights={self.copy_weights})"

    def clone(self) -> "LayerConfig":
        """
        Create a deep copy of this LayerConfig.

        Returns:
            A new LayerConfig instance with the same properties
        """
        layer_names_copy: Union[
            str, List[str], Dict[str, Union[str, List[str], int, List[int]]]
        ]
        if isinstance(self.layer_names, dict):
            layer_names_copy = {
                k: v.copy() if isinstance(v, list) else v
                for k, v in self.layer_names.items()
            }
        elif isinstance(self.layer_names, list):
            layer_names_copy = self.layer_names.copy()
        else:
            layer_names_copy = self.layer_names

        params_copy: Dict[str, ParamSpec] = {}
        for key, param_spec in self.params.items():
            if isinstance(param_spec, list):
                params_copy[key] = param_spec.copy()
            else:
                params_copy[key] = copy_module.deepcopy(param_spec)

        return LayerConfig(
            layer_names=layer_names_copy,
            params=params_copy,
            separate=self.separate,
            copy_weights=self.copy_weights,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LayerConfig):
            return False
        return (
            self.layer_names == other.layer_names
            and self.params == other.params
            and self.separate == other.separate
            and self.copy_weights == other.copy_weights
        )
