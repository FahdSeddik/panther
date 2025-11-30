import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch.nn as nn


class LayerNameResolver:
    """
    Provides intuitive ways to select layers for tuning in large models.

    This resolver allows users to specify layers using patterns, types, or indices
    rather than requiring exact layer names.
    """

    def __init__(
        self, model: nn.Module, layer_map: Optional[Dict[str, nn.Module]] = None
    ):
        """
        Initialize the resolver with a model.

        Args:
            model: The neural network model to analyze
        """
        self.model = model
        self.layer_map = layer_map if layer_map is not None else {}

    def _check_selectors(
        self, selectors: Dict[str, Union[str, List[str], int, List[int]]]
    ) -> None:
        """
        Validate selector configurations before processing.

        Args:
            selectors: Dictionary with selection criteria

        Raises:
            ValueError: If conflicting or invalid selectors are detected
        """
        # Check if both 'indices' and 'range' are used together
        if "indices" in selectors and "range" in selectors:
            raise ValueError("Cannot use both 'indices' and 'range' selectors together")

        # Validate range format if present
        if "range" in selectors:
            range_val = selectors["range"]
            if not isinstance(range_val, list):
                raise ValueError(
                    "'range' must be a list of [start, end] or [start, end, step]"
                )

            if len(range_val) < 2 or len(range_val) > 3:
                raise ValueError(
                    "'range' must contain 2 or 3 elements: [start, end] or [start, end, step]"
                )

            if not all(isinstance(v, int) for v in range_val):
                raise ValueError("All 'range' values must be integers")

            # Type narrowing: we know all elements are int now
            start: int = range_val[0]  # type: ignore
            end: int = range_val[1]  # type: ignore

            if start < 0:
                raise ValueError("'range' start value must be non-negative")
            if end <= start:
                raise ValueError("'range' end value must be greater than start value")

        # Validate indices format if present
        if "indices" in selectors:
            indices = selectors["indices"]
            if isinstance(indices, int):
                if indices < 0:
                    raise ValueError("Index values must be non-negative")
            elif isinstance(indices, list):
                if not all(isinstance(idx, int) for idx in indices):
                    raise ValueError("All 'indices' must be integers")
                # Type narrowing: all elements are int now
                int_indices: List[int] = [
                    idx for idx in indices if isinstance(idx, int)
                ]
                if any(idx < 0 for idx in int_indices):
                    raise ValueError("All 'indices' must be non-negative")
            else:
                raise ValueError("'indices' must be an integer or list of integers")

    def resolve(
        self,
        selectors: Union[
            str, List[str], Dict[str, Union[str, List[str], int, List[int]]]
        ],
    ) -> List[str]:
        """
        Resolve layer name selectors to actual layer names in the model.

        Args:
            selectors: One or more selectors to match layers. Can be:

                - A single string pattern (e.g., "encoder.*attention")
                - A list of string patterns
                - A dictionary with keys:

                    - 'pattern': String or list of regex patterns
                    - 'type': Layer type or list of types (e.g., 'Linear', 'Conv2d')
                    - 'contains': String that layer name must contain
                    - 'indices': Indices to select from matched layers (e.g., [0, 2, 4] for first, third, fifth)
                    - 'range': Range of indices as [start, end, step]

        Returns:
            List of resolved layer names that match the selectors
        """
        # Handle string case
        if isinstance(selectors, str):
            return self._resolve_pattern(selectors)

        # Handle list of strings case
        elif isinstance(selectors, list) and all(isinstance(s, str) for s in selectors):
            matched_names = []
            for selector in selectors:
                matched_names.extend(self._resolve_pattern(selector))
            return list(set(matched_names))  # Remove duplicates

        # Handle dictionary case with advanced options
        elif isinstance(selectors, dict):
            # Validate selectors before processing
            self._check_selectors(selectors)

            matched_layers = set(self.layer_map.keys())

            # Filter by pattern
            if "pattern" in selectors:
                patterns_val = selectors["pattern"]
                patterns: List[str] = []
                if isinstance(patterns_val, str):
                    patterns = [patterns_val]
                elif isinstance(patterns_val, list):
                    # Filter to only string elements
                    patterns = [p for p in patterns_val if isinstance(p, str)]

                pattern_matches = set()
                for pattern in patterns:
                    pattern_matches.update(self._resolve_pattern(pattern))

                matched_layers = matched_layers.intersection(pattern_matches)

            # Filter by containing string
            if "contains" in selectors:
                contains_val = selectors["contains"]
                contains: List[str] = []
                if isinstance(contains_val, str):
                    contains = [contains_val]
                elif isinstance(contains_val, list):
                    contains = [c for c in contains_val if isinstance(c, str)]
                matched_layers = {
                    name for name in matched_layers if any(c in name for c in contains)
                }

            # Filter by type
            if "type" in selectors:
                types_val = selectors["type"]
                types: List[str] = []
                if isinstance(types_val, str):
                    types = [types_val]
                elif isinstance(types_val, list):
                    types = [t for t in types_val if isinstance(t, str)]

                type_matches = set()
                for layer_name in matched_layers:
                    layer = self.layer_map[layer_name]
                    layer_type = type(layer).__name__
                    if layer_type in types:
                        type_matches.add(layer_name)

                matched_layers = type_matches

            # Convert to list for indexing operations
            matched_list = sorted(list(matched_layers))

            if not matched_list:
                return []

            # Apply indices filter
            if "indices" in selectors:
                indices_val = selectors["indices"]
                if isinstance(indices_val, int):
                    indices = [indices_val]
                elif isinstance(indices_val, list):
                    indices = [i for i in indices_val if isinstance(i, int)]
                else:
                    indices = []

                # Validate indices are within bounds
                if indices and max(indices) >= len(matched_list):
                    raise ValueError(
                        f"Index {max(indices)} is out of bounds for {len(matched_list)} matched layers"
                    )

                selected_layers = [matched_list[i] for i in indices]
                return selected_layers

            # Apply range filter
            if "range" in selectors:
                range_val = selectors["range"]
                if isinstance(range_val, list) and len(range_val) >= 2:
                    start_val = range_val[0]
                    end_val = range_val[1]
                    step_val = range_val[2] if len(range_val) > 2 else 1

                    if not isinstance(start_val, int) or not isinstance(end_val, int):
                        raise ValueError("Range start and end must be integers")
                    if not isinstance(step_val, int):
                        raise ValueError("Range step must be an integer")

                    # Type narrowed now
                    start: int = start_val
                    end: int = end_val
                    step: int = step_val

                    # Validate end is within bounds
                    if end > len(matched_list):
                        raise ValueError(
                            f"Range end {end} exceeds the number of matched layers ({len(matched_list)})"
                        )

                    selected_layers = matched_list[start:end:step]
                    return selected_layers

            if len(matched_list) == 0:
                raise ValueError("No layers matched the provided selectors")

            return matched_list

        else:
            raise ValueError(
                "Selectors must be a string, list of strings, or a dictionary with selection criteria"
            )

    def _resolve_pattern(self, pattern: str) -> List[str]:
        """
        Resolve a regex pattern to matching layer names.

        Args:
            pattern: Regex pattern to match layer names

        Returns:
            List of layer names that match the pattern
        """
        try:
            regex = re.compile(pattern)
            return [name for name in self.layer_map.keys() if regex.search(name)]
        except re.error:
            # If not a valid regex, try simple substring matching
            return [name for name in self.layer_map.keys() if pattern in name]

    def get_layers_by_depth(self, depth: int) -> List[str]:
        """
        Get layer names at a specific depth in the model hierarchy.

        Depth is determined by the number of '.' in the layer name.
        For example, 'model.encoder.layer0' has depth 2.

        Args:
            depth: The depth level to retrieve layers from

        Returns:
            List of layer names at the specified depth
        """
        return [name for name in self.layer_map.keys() if name.count(".") == depth]

    def get_layer_types(self) -> Dict[str, List[str]]:
        """
        Get a mapping of layer types to layer names.

        Returns:
            Dictionary where keys are layer types (e.g., 'Linear', 'Conv2d')
            and values are lists of layer names of that type
        """
        type_map: Dict[str, List[str]] = {}
        for name, layer in self.layer_map.items():
            layer_type = type(layer).__name__
            if layer_type not in type_map:
                type_map[layer_type] = []
            type_map[layer_type].append(name)

        return type_map

    def get_layers_by_parent(self, parent_name: str) -> List[str]:
        """
        Get all direct child layers of a parent layer.

        Args:
            parent_name: Name of the parent layer

        Returns:
            List of layer names that are direct children of the parent
        """
        if not parent_name.endswith("."):
            parent_name = parent_name + "."

        return [
            name
            for name in self.layer_map.keys()
            if name.startswith(parent_name) and name[len(parent_name) :].count(".") == 0
        ]

    def filter_layers_by_attribute(
        self, attribute_name: str, attribute_value: Any
    ) -> List[str]:
        """
        Filter layers based on a specific attribute value.

        Args:
            attribute_name: Name of the attribute to check
            attribute_value: Expected value of the attribute

        Returns:
            List of layer names with matching attribute value
        """
        matched_layers = []

        for name, layer in self.layer_map.items():
            if hasattr(layer, attribute_name):
                if getattr(layer, attribute_name) == attribute_value:
                    matched_layers.append(name)

        return matched_layers

    def filter_layers_by_function(
        self, filter_fn: Callable[[nn.Module], bool]
    ) -> List[str]:
        """
        Filter layers using a custom function.

        Args:
            filter_fn: Function that takes a layer and returns True if it should be included

        Returns:
            List of layer names that pass the filter function
        """
        return [name for name, layer in self.layer_map.items() if filter_fn(layer)]

    def get_layer_parameter_count(self) -> Dict[str, int]:
        """
        Count parameters for each layer in the model.

        Returns:
            Dictionary mapping layer names to their parameter counts
        """
        param_counts = {}

        for name, layer in self.layer_map.items():
            param_count = sum(p.numel() for p in layer.parameters())
            param_counts[name] = param_count

        return param_counts

    def get_top_n_layers_by_parameters(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Get the top N layers with the most parameters.

        Args:
            n: Number of top layers to return

        Returns:
            List of tuples (layer_name, parameter_count) sorted by parameter count
        """
        param_counts = self.get_layer_parameter_count()
        sorted_layers = sorted(param_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_layers[:n]

    def find_common_prefix(self) -> str:
        """
        Find the common prefix shared by all layer names.

        This is useful for identifying the root module name.

        Returns:
            The common prefix string
        """
        if not self.layer_map:
            return ""

        names = list(self.layer_map.keys())
        prefix = names[0]

        for name in names[1:]:
            while not name.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""

        return prefix
