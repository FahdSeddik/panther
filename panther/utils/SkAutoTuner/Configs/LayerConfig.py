from typing import Any, Dict, List, Optional, Union


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
        params: Dict[str, List],
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
        if isinstance(self.layer_names, dict):
            layer_names_copy = {}
            for key, value in self.layer_names.items():
                if isinstance(value, list):
                    layer_names_copy[key] = value.copy()
                else:
                    layer_names_copy[key] = value
        elif isinstance(self.layer_names, list):
            layer_names_copy = self.layer_names.copy()
        else:
            layer_names_copy = self.layer_names

        # Deep copy params
        params_copy = {}
        for key, value in self.params.items():
            params_copy[key] = value.copy()

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
        # Handle layer_names merging based on type
        if isinstance(self.layer_names, list) and isinstance(other.layer_names, list):
            merged_layer_names = self.layer_names + other.layer_names
        elif isinstance(self.layer_names, str) and isinstance(other.layer_names, str):
            merged_layer_names = [self.layer_names, other.layer_names]
        elif isinstance(self.layer_names, dict) and isinstance(other.layer_names, dict):
            merged_layer_names = self.layer_names.copy()
            for key, value in other.layer_names.items():
                if key in merged_layer_names:
                    if isinstance(merged_layer_names[key], list) and isinstance(
                        value, list
                    ):
                        merged_layer_names[key].extend(value)
                    elif isinstance(merged_layer_names[key], list):
                        merged_layer_names[key].append(value)
                    elif isinstance(value, list):
                        merged_layer_names[key] = [merged_layer_names[key]] + value
                    else:
                        merged_layer_names[key] = [merged_layer_names[key], value]
                else:
                    merged_layer_names[key] = value
        else:
            # If types are different, convert to list and combine
            if isinstance(self.layer_names, (str, dict)):
                first = [self.layer_names]
            else:
                first = self.layer_names

            if isinstance(other.layer_names, (str, dict)):
                second = [other.layer_names]
            else:
                second = other.layer_names

            merged_layer_names = first + second

        # Merge params
        merged_params = self.params.copy()
        for key, value in other.params.items():
            if key in merged_params:
                # Combine parameter values, removing duplicates
                merged_params[key] = list(set(merged_params[key] + value))
            else:
                merged_params[key] = value

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
        for values in self.params.values():
            space_size *= len(values)
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
        return self.params.get(param_name)

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
