import copy as copy_module
from typing import Any, Dict, List, Optional, Union

from .ParamSpec import ParamSpec, get_param_choices


class LayerConfig:
    """
    Configuration object for a single layer or group of layers.
    Contains the layer names, parameters to tune and whether these layers should be tuned separately.
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

                - A string: Regex pattern or substring (e.g., "encoder", "layer1.*conv")
                - A list of strings: Multiple patterns or exact layer names (e.g., ["encoder", "decoder"])
                - A dictionary with selection criteria:

                    - "pattern": String or list of regex patterns (e.g., "encoder.*")
                    - "type": Layer type or list of types (e.g., "Linear", ["Conv2d", "ConvTranspose2d"])
                    - "contains": String that layer name must contain (e.g., "attention")
                    - "indices": Specific indices to select from matched layers (e.g., [0, 2, 4])
                    - "range": Range of indices as [start, end, step] (e.g., [0, 6] or [0, 12, 2])
                    - Multiple criteria can be combined (e.g., {"pattern": "encoder.*", "type": "Linear"})

            params: Dictionary of parameter names and their possible values to try
            separate: Whether these layers should be tuned separately or together
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
        # Deep copy layer_names based on its type
        layer_names_copy: Union[
            str, List[str], Dict[str, Union[str, List[str], int, List[int]]]
        ]
        if isinstance(self.layer_names, dict):
            layer_names_copy_dict: Dict[str, Union[str, List[str], int, List[int]]] = {}
            for key, layer_val in self.layer_names.items():
                if isinstance(layer_val, list):
                    layer_names_copy_dict[key] = layer_val.copy()
                else:
                    layer_names_copy_dict[key] = layer_val
            layer_names_copy = layer_names_copy_dict
        elif isinstance(self.layer_names, list):
            layer_names_copy = self.layer_names.copy()
        else:
            layer_names_copy = self.layer_names

        # Deep copy params
        params_copy: Dict[str, ParamSpec] = {}
        for key, param_spec in self.params.items():
            if isinstance(param_spec, list):
                params_copy[key] = param_spec.copy()
            else:
                # ParamSpec types (Categorical, Int, Float) are dataclasses
                params_copy[key] = copy_module.deepcopy(param_spec)

        return LayerConfig(
            layer_names=layer_names_copy,
            params=params_copy,
            separate=self.separate,
            copy_weights=self.copy_weights,
        )

    def merge(self, other: "LayerConfig") -> "LayerConfig":
        """
        Merge this LayerConfig with another one.

        The merged config will have:
        - Combined layer_names (if they are lists)
        - Combined params dictionaries
        - 'separate' and 'copy_weights' values from this config

        Args:
            other: Another LayerConfig to merge with

        Returns:
            A new LayerConfig with merged properties

        Raises:
            TypeError: If layer_names are not compatible for merging
        """
        merged_layer_names: Union[
            str, List[str], Dict[str, Union[str, List[str], int, List[int]]]
        ]

        # Handle layer_names merging based on type
        if isinstance(self.layer_names, list) and isinstance(other.layer_names, list):
            merged_layer_names = self.layer_names + other.layer_names
        elif isinstance(self.layer_names, str) and isinstance(other.layer_names, str):
            merged_layer_names = [self.layer_names, other.layer_names]
        elif isinstance(self.layer_names, dict) and isinstance(other.layer_names, dict):
            merged_layer_names_dict: Dict[
                str, Union[str, List[str], int, List[int]]
            ] = {}
            for key, layer_val in self.layer_names.items():
                merged_layer_names_dict[key] = layer_val

            for key, layer_val in other.layer_names.items():
                if key in merged_layer_names_dict:
                    existing = merged_layer_names_dict[key]
                    if isinstance(existing, list) and isinstance(layer_val, list):
                        # Properly handle list types
                        if isinstance(
                            existing[0] if existing else None, str
                        ) and isinstance(layer_val[0] if layer_val else None, str):
                            result: List[str] = existing + layer_val  # type: ignore
                            merged_layer_names_dict[key] = result
                        else:
                            result_int: List[int] = existing + layer_val  # type: ignore
                            merged_layer_names_dict[key] = result_int
                    elif isinstance(existing, list):
                        if isinstance(layer_val, (str, int)):
                            existing.append(layer_val)  # type: ignore
                    elif isinstance(layer_val, list):
                        merged_layer_names_dict[key] = [existing] + layer_val  # type: ignore
                    else:
                        merged_layer_names_dict[key] = [existing, layer_val]  # type: ignore
                else:
                    merged_layer_names_dict[key] = layer_val
            merged_layer_names = merged_layer_names_dict
        else:
            # If types are different, convert to list and combine
            first_list: List[
                Union[str, Dict[str, Union[str, List[str], int, List[int]]]]
            ]
            if isinstance(self.layer_names, (str, dict)):
                first_list = [self.layer_names]  # type: ignore
            else:
                first_list = self.layer_names  # type: ignore

            second_list: List[
                Union[str, Dict[str, Union[str, List[str], int, List[int]]]]
            ]
            if isinstance(other.layer_names, (str, dict)):
                second_list = [other.layer_names]  # type: ignore
            else:
                second_list = other.layer_names  # type: ignore

            merged_layer_names = first_list + second_list  # type: ignore

        # Merge params
        merged_params: Dict[str, ParamSpec] = self.params.copy()
        for key, param_spec in other.params.items():
            if key in merged_params:
                # Combine parameter values, removing duplicates
                # Convert both to lists for merging
                existing_choices = get_param_choices(merged_params[key]) or []
                new_choices = get_param_choices(param_spec) or []
                merged_params[key] = list(set(existing_choices + new_choices))
            else:
                merged_params[key] = param_spec

        return LayerConfig(
            layer_names=merged_layer_names,
            params=merged_params,
            separate=self.separate,
            copy_weights=self.copy_weights,
        )

    def with_param(self, param_name: str, param_values: List[Any]) -> "LayerConfig":
        """
        Create a new LayerConfig with an additional parameter to tune.

        Args:
            param_name: Name of the parameter
            param_values: List of values to try for this parameter

        Returns:
            A new LayerConfig with the additional parameter
        """
        new_config = self.clone()
        new_config.params[param_name] = param_values
        return new_config

    def without_param(self, param_name: str) -> "LayerConfig":
        """
        Create a new LayerConfig with a parameter removed.

        Args:
            param_name: Name of the parameter to remove

        Returns:
            A new LayerConfig without the specified parameter
        """
        new_config = self.clone()
        if param_name in new_config.params:
            del new_config.params[param_name]
        return new_config

    def with_layer_names(
        self,
        layer_names: Union[
            str, List[str], Dict[str, Union[str, List[str], int, List[int]]]
        ],
    ) -> "LayerConfig":
        """
        Create a new LayerConfig with different layer names.

        Args:
            layer_names: New layer names specification

        Returns:
            A new LayerConfig with the specified layer names
        """
        return LayerConfig(
            layer_names=layer_names,
            params=self.params.copy(),
            separate=self.separate,
            copy_weights=self.copy_weights,
        )

    def toggle_separate(self) -> "LayerConfig":
        """
        Create a new LayerConfig with the 'separate' flag toggled.

        Returns:
            A new LayerConfig with 'separate' set to the opposite of its current value
        """
        new_config = self.clone()
        new_config.separate = not self.separate
        return new_config

    def toggle_copy_weights(self) -> "LayerConfig":
        """
        Create a new LayerConfig with the 'copy_weights' flag toggled.

        Returns:
            A new LayerConfig with 'copy_weights' set to the opposite of its current value
        """
        new_config = self.clone()
        new_config.copy_weights = not self.copy_weights
        return new_config

    def get_param_space_size(self) -> int:
        """
        Calculate the total number of parameter combinations to try.

        Returns:
            The product of the number of values for each parameter
        """
        if not self.params:
            return 0

        space_size = 1
        for spec in self.params.values():
            choices = get_param_choices(spec)
            if choices is not None:
                space_size *= len(choices)
            else:
                # Continuous range - return -1 to indicate infinite space
                return -1
        return space_size

    def has_param(self, param_name: str) -> bool:
        """
        Check if this config includes a specific parameter.

        Args:
            param_name: The parameter name to check for

        Returns:
            True if the parameter exists in this config, False otherwise
        """
        return param_name in self.params

    def get_param_values(self, param_name: str) -> Optional[List[Any]]:
        """
        Get the list of values for a specific parameter.

        Args:
            param_name: The parameter name to get values for

        Returns:
            List of values for the parameter, or None if the parameter doesn't exist
        """
        spec = self.params.get(param_name)
        if spec is None:
            return None
        return get_param_choices(spec)

    def param_count(self) -> int:
        """
        Get the number of parameters being tuned in this config.

        Returns:
            The number of parameters in the params dictionary
        """
        return len(self.params)

    def __eq__(self, other: object) -> bool:
        """
        Check if this LayerConfig is equal to another.

        Args:
            other: Another object to compare with

        Returns:
            True if the configs are equal, False otherwise
        """
        if not isinstance(other, LayerConfig):
            return False

        return (
            self.layer_names == other.layer_names
            and self.params == other.params
            and self.separate == other.separate
            and self.copy_weights == other.copy_weights
        )
