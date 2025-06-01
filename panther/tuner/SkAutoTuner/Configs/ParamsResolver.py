import warnings
from typing import Any, Dict, List, Optional, Tuple, Type

import torch.nn as nn

from ..layer_type_mapping import LAYER_TYPE_MAPPING
from .LayerConfig import LayerConfig
from .TuningConfigs import TuningConfigs


class ParamsResolver:
    """
    A class to resolve parameters for the SkAutoTuner.
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

    def _recommend_params(
        self, layer_type: Type[nn.Module], layer_sizes: Tuple
    ) -> Dict[str, List[Any]]:
        recommended_params: Dict[str, List[Any]] = {}
        num_terms_values = [1, 2, 3]  # Common default for num_terms

        if layer_type == nn.Linear:
            if not (
                len(layer_sizes) >= 2
                and isinstance(layer_sizes[0], int)
                and isinstance(layer_sizes[1], int)
            ):
                warnings.warn(
                    f"Linear layer_sizes tuple {layer_sizes} is not as expected. Using fallback param recommendation."
                )
                return {"num_terms": num_terms_values, "low_rank": [4, 8, 16, 32]}

            in_features, out_features = layer_sizes[0], layer_sizes[1]
            D1, D2 = out_features, in_features

            if D1 <= 0 or D2 <= 0:
                warnings.warn(
                    f"Linear layer dimensions D1={D1}, D2={D2} are not positive. Using fallback param recommendation."
                )
                return {"num_terms": num_terms_values, "low_rank": [4, 8, 16, 32]}

            start_k_iter = 16
            linear_all_low_rank_values = set()
            recommended_params["num_terms"] = list(num_terms_values)

            for L in num_terms_values:
                current_k = start_k_iter

                while current_k > 0:
                    if (D1 + D2) == 0:
                        is_efficient = False
                    else:
                        is_efficient = (D1 * D2) > (2 * L * current_k * (D1 + D2))

                    if is_efficient:
                        linear_all_low_rank_values.add(current_k)
                        current_k *= 2
                    else:
                        break

            if not linear_all_low_rank_values:
                warnings.warn(
                    f"Iterative low_rank generation for nn.Linear with sizes {layer_sizes} "
                    f"and num_terms {num_terms_values} yielded no efficient ranks starting from k={start_k_iter}. "
                    "Applying fallback: [1, 2, 4, 8]."
                )
                linear_all_low_rank_values = {1, 2, 4, 8}

            recommended_params["low_rank"] = [
                item for item in linear_all_low_rank_values
            ]

            return recommended_params

        elif layer_type == nn.Conv2d:
            conv_all_low_rank_values = set()
            if not (
                len(layer_sizes) >= 2
                and isinstance(layer_sizes[0], int)
                and isinstance(layer_sizes[1], int)
            ):
                warnings.warn(
                    f"Conv2d layer_sizes tuple {layer_sizes} is not as expected. Using fallback param recommendation."
                )
                return {"num_terms": num_terms_values, "low_rank": [4, 8, 16, 32]}

            # For Conv2d, layer_sizes is (in_channels, out_channels)
            # Let D1_c = out_channels and D2_c = in_channels for the formula
            D2_c, D1_c = (
                layer_sizes[0],
                layer_sizes[1],
            )  # D2_c = in_channels, D1_c = out_channels

            if D1_c <= 0 or D2_c <= 0:
                warnings.warn(
                    f"Conv2d dimensions out_channels(D1_c)={D1_c}, in_channels(D2_c)={D2_c} are not positive. Using fallback param recommendation."
                )
                return {"num_terms": num_terms_values, "low_rank": [4, 8, 16, 32]}

            start_k_iter = 8
            recommended_params["num_terms"] = list(num_terms_values)

            for L in num_terms_values:
                if L <= 0:  # Should not happen with current num_terms_values
                    continue
                current_k = start_k_iter

                while current_k > 0:
                    # Efficiency equation: (D1_c * D2_c) > (2 * L * K * (D1_c + D2_c))
                    # This is analogous to the Linear layer's efficiency condition.
                    if (
                        D1_c + D2_c
                    ) == 0:  # Avoid division by zero or meaningless comparison
                        is_efficient = False
                    else:
                        is_efficient = (D1_c * D2_c) > (
                            2 * L * current_k * (D1_c + D2_c)
                        )

                    if is_efficient:
                        conv_all_low_rank_values.add(current_k)
                        if current_k > 512:  # Safety break for Conv2d ranks
                            break
                        current_k *= 2
                    else:
                        break

            # Fallback if no ranks generated by the loop
            if not conv_all_low_rank_values:
                warnings.warn(
                    f"Iterative low_rank generation for nn.Conv2d with sizes {layer_sizes} "
                    f"and num_terms {num_terms_values} yielded no efficient ranks starting from k={start_k_iter} "
                    f"using formula (D1*D2) > (2*L*K*(D1+D2)). Applying fallback: [1, 2, 4, 8]."
                )
                conv_all_low_rank_values = set([1, 2, 4, 8])

            recommended_params["low_rank"] = list(conv_all_low_rank_values)

            return recommended_params

        else:
            warnings.warn(
                f"Automatic parameter value recommendation for layer type {layer_type.__name__} "
                f"is not implemented. Using generic fallbacks."
            )
            recommended_params["num_terms"] = [1, 2]
            recommended_params["low_rank"] = [4, 8, 16, 32]
            return recommended_params

    def resolve(self, config: LayerConfig) -> TuningConfigs:
        """
        Resolves and returns the parameters.

        If config.params is a dictionary, it's returned as is, wrapped in TuningConfigs.
        If config.params is "auto", it automatically determines parameters based on layer types and sizes.
        Otherwise, raises a ValueError.

        :return: The resolved parameters as TuningConfigs.
        """
        if isinstance(config.params, dict):
            return TuningConfigs([config])

        if config.params != "auto":
            raise ValueError(
                f"config.params must be a dictionary or the string 'auto', got {config.params}"
            )

        # Logic for "auto"
        resolved_layer_configs: List[LayerConfig] = []
        # Structure: {layer_type: {size_key: [layer_name_1, layer_name_2, ...]}}
        layers_by_type_and_size: Dict[Type[nn.Module], Dict[Tuple, List[str]]] = {}

        for layer_name in config.layer_names:
            if layer_name not in self.layer_map:
                warnings.warn(
                    f"Layer {layer_name} specified in config not found in model's layer_map. Skipping."
                )
                continue

            layer_obj = self.layer_map[layer_name]
            layer_type = type(layer_obj)

            if layer_type not in LAYER_TYPE_MAPPING and config.params == "auto":
                warnings.warn(
                    f"Layer type {layer_type.__name__} for layer {layer_name} is not supported "
                    f"for automatic parameter generation via LAYER_TYPE_MAPPING. Skipping."
                )
                continue

            size_key: Tuple
            if isinstance(layer_obj, nn.Linear):
                size_key = (layer_obj.in_features, layer_obj.out_features)
            elif isinstance(layer_obj, nn.Conv2d):
                # Simplified size_key for Conv2d
                size_key = (layer_obj.in_channels, layer_obj.out_channels)
            else:
                warnings.warn(
                    f"Automatic parameter resolution for layer type {layer_type.__name__} "
                    f"(layer: {layer_name}) is not implemented. Skipping."
                )
                continue

            if layer_type not in layers_by_type_and_size:
                layers_by_type_and_size[layer_type] = {}

            group = layers_by_type_and_size[layer_type]
            if size_key not in group:
                group[size_key] = []
            group[size_key].append(layer_name)

        for layer_type, groups_by_size in layers_by_type_and_size.items():
            for size_key, layer_names_list in groups_by_size.items():
                try:
                    recommended_params = self._recommend_params(layer_type, size_key)
                    new_lc = LayerConfig(
                        layer_names=layer_names_list,
                        params=recommended_params,
                        separate=config.separate,
                        copy_weights=config.copy_weights,
                    )
                    resolved_layer_configs.append(new_lc)
                except ValueError as e:
                    warnings.warn(
                        f"Could not recommend params for group (type: {layer_type.__name__}, size_key: {size_key}): {e}. Skipping group."
                    )

        if (
            not resolved_layer_configs
            and config.params == "auto"
            and config.layer_names
        ):
            warnings.warn(
                f"Auto parameter resolution for layers {config.layer_names} resulted in no valid tuning configurations. "
                "This might be due to unsupported layer types or issues with LAYER_TYPE_MAPPING."
            )
            return TuningConfigs([])

        return TuningConfigs(resolved_layer_configs)
