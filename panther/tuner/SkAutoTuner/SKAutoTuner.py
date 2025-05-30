import copy
import os
import pickle
from typing import Any, Callable, Dict, Optional, Tuple, Type

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from .Configs import LayerConfig, LayerNameResolver, TuningConfigs
from .layer_type_mapping import LAYER_TYPE_MAPPING
from .Searching import GridSearch, SearchAlgorithm


class SKAutoTuner:
    """
    Auto-tuner for sketched neural network layers.
    """

    def __init__(
        self,
        model: nn.Module,
        configs: TuningConfigs,
        accuracy_eval_func: Callable[[nn.Module], float],
        search_algorithm: Optional[SearchAlgorithm] = None,
        verbose: bool = False,
        accuracy_threshold: Optional[float] = None,
        optmization_eval_func: Optional[Callable[[nn.Module], float]] = None,
        num_runs_per_param=1,
    ):
        """
        Initialize the autotuner.

        Args:
            model: The neural network model to tune
            configs: Configuration for the layers to tune
            accuracy_eval_func: Evaluation function that takes a model and returns an accuracy score (higher is better)
            search_algorithm: Search algorithm to use for finding optimal parameters
            verbose: Whether to print progress during tuning
            accuracy_threshold: Minimum acceptable accuracy (if None, will use accuracy_eval_func for optimization)
            optmization_eval_func: Function to maxmimize (e.g., speed) after reaching the accuracy threshold
        """
        self.model = model
        self.accuracy_eval_func = accuracy_eval_func
        self.search_algorithm = search_algorithm or GridSearch()
        self.verbose = verbose
        self.num_runs_per_param = num_runs_per_param
        self.accuracy_threshold = accuracy_threshold
        self.optmization_eval_func = optmization_eval_func
        # Dictionary to store results for each layer
        self.results: Dict[str, Any] = {}
        # Dictionary to store best parameters for each layer
        self.best_params: Dict[str, Any] = {}
        # Map from layer name to layer object
        self.layer_map: Dict[str, nn.Module] = {}
        # Initialize the layer name resolver
        self._build_layer_map(model)
        self.resolver = LayerNameResolver(model, self.layer_map)
        # Resolve layer names in configs
        self.configs = self._resolve_configs(configs)

        self._check_configs()

    def _resolve_configs(self, configs: TuningConfigs) -> TuningConfigs:
        """
        Resolve layer names in configs to actual layer names in the model.

        Args:
            configs: Configuration for the layers to tune

        Returns:
            Updated configuration with resolved layer names
        """
        resolved_configs = TuningConfigs([])

        for config in configs.configs:
            # Make a deep copy of the config to avoid modifying the original
            resolved_config = LayerConfig(
                layer_names=self.resolver.resolve(config.layer_names),
                params=config.params,
                separate=config.separate,
                copy_weights=config.copy_weights,
            )
            resolved_configs.configs.append(resolved_config)

        return resolved_configs

    def _check_configs(self):
        """
        Validate configuration settings before tuning.
        Checks:
        1. No empty layer groups
        2. No empty parameter spaces
        3. Parameters are valid for layer types
        4. Grouped layers (if not separate) have compatible parameters

        Raises:
            ValueError: If any configuration issue is found
        """
        for i, config in enumerate(self.configs.configs):
            # Check for empty layer groups
            if not config.layer_names:
                raise ValueError(f"Config {i} has no layer names specified")

            # Check for empty parameter spaces
            if not config.params:
                raise ValueError(
                    f"Config {i} for layers {config.layer_names} has no parameters to tune"
                )

            # Get actual layer objects and check compatibility
            layers = []
            for layer_name in config.layer_names:
                try:
                    layer = self._get_layer_by_name(layer_name)
                    layers.append(layer)
                except ValueError as e:
                    raise ValueError(
                        f"Layer '{layer_name}' in config {i} not found in model: {e}"
                    )

            # Check each parameter is valid for the layer types
            for layer in layers:
                layer_type = type(layer)
                if layer_type not in LAYER_TYPE_MAPPING:
                    raise ValueError(
                        f"Layer '{layer_name}' is of type {layer_type.__name__}, which is not supported for sketching"
                    )

                valid_params = LAYER_TYPE_MAPPING[layer_type]["params"]
                for param in config.params:
                    if param not in valid_params:
                        raise ValueError(
                            f"Parameter '{param}' is not valid for layer type {layer_type.__name__}. Valid parameters are: {valid_params}"
                        )

            # Check parameter compatibility across layers in the group
            if len(config.layer_names) > 1:
                param_sets = [
                    set(LAYER_TYPE_MAPPING[type(layer)]["params"]) for layer in layers
                ]
                common_params = set.intersection(*param_sets)

                # Check that all parameters in the config are valid for all layers
                for param in config.params:
                    if param not in common_params:
                        raise ValueError(
                            f"Parameter '{param}' in config {i} is not compatible with all layers in the group. "
                            f"For joint tuning, all layers must support the same parameters."
                        )

    def _build_layer_map(self, model: nn.Module, prefix: str = ""):
        """Build a mapping from layer names to layer objects."""
        for name, module in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            self.layer_map[full_name] = module
            self._build_layer_map(module, full_name)

    def _get_layer_by_name(self, name: str) -> nn.Module:
        """Get a layer by its name."""
        if name not in self.layer_map:
            raise ValueError(f"Layer {name} not found in the model")
        return self.layer_map[name]

    def _get_parent_module_and_name(self, layer_name: str) -> Tuple[nn.Module, str]:
        """Get the parent module and attribute name for a layer."""
        if layer_name not in self.layer_map:
            raise ValueError(f"Layer {layer_name} not found in the model")

        if "." not in layer_name:
            return self.model, layer_name

        parent_name, child_name = layer_name.rsplit(".", 1)

        try:
            parent = self._get_layer_by_name(parent_name)
        except ValueError:
            parent = self.model

        return parent, child_name

    def _replace_layer(self, layer_name: str, new_layer: nn.Module) -> None:
        """
        Swap a layer in the model with a new layer.

        Args:
            layer_name: Name of the layer to swap
            new_layer: New layer to replace the old one

        Returns:
            None
        """
        parent, name = self._get_parent_module_and_name(layer_name)
        setattr(parent, name, new_layer)

        if self.verbose:
            print(f"replaced {layer_name} with {new_layer.__class__.__name__}")

    def _sketch_layer(
        self,
        layer_name: str,
        layer_type: Type[nn.Module],
        params: Dict[str, Any],
        copy_weights: bool = True,
    ) -> Any | None:
        """
        Replace a layer with its sketched version.

        Args:
            layer_name: Name of the layer to replace
            layer_type: Type of the layer
            params: Parameters for the sketched layer
            copy_weights: Whether to copy the learnable parameters from the original

        Returns:
            True if layer was replaced, False otherwise
        """
        parent, name = self._get_parent_module_and_name(layer_name)
        original_layer = getattr(parent, name)
        sketched_layer = None  # Initialize sketched_layer

        # Check if the layer type is supported
        if type(original_layer) not in LAYER_TYPE_MAPPING:
            if self.verbose:
                print(
                    f"Layer type {type(original_layer).__name__} is not supported for sketching"
                )
            return False

        # Get the sketched layer class and its parameters
        from .layer_type_mapping import get_sketched_class

        sketched_class = get_sketched_class(type(original_layer))

        # Create the sketched layer with original parameters and new sketching parameters
        if isinstance(original_layer, nn.Linear):
            sketched_layer = sketched_class(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                bias=original_layer.bias is not None,
                W_init=original_layer.weight if copy_weights else None,
                device=original_layer.weight.device,
                dtype=original_layer.weight.dtype,
                **params,
            )

            # Copy bias if needed
            if (
                copy_weights
                and original_layer.bias is not None
                and sketched_layer.bias is not None
            ):
                sketched_layer.bias.data.copy_(original_layer.bias.data)
            # for the weights its handled in the SKLinear constructor

        elif isinstance(original_layer, nn.Conv2d):
            # Create a new sketched convolution layer
            from panther.nn.conv2d import SKConv2d

            if copy_weights:
                sketched_layer = SKConv2d.fromTorch(original_layer, **params)
            else:
                sketched_layer = sketched_class(
                    in_channels=original_layer.in_channels,
                    out_channels=original_layer.out_channels,
                    kernel_size=original_layer.kernel_size,
                    stride=original_layer.stride,
                    padding=original_layer.padding,
                    device=original_layer.weight.device,
                    dtype=original_layer.weight.dtype,
                    **params,
                )

        elif isinstance(original_layer, nn.MultiheadAttention):
            # Create a new sketched attention layer
            sketched_layer = sketched_class(
                embed_dim=original_layer.embed_dim,
                num_heads=original_layer.num_heads,
                dropout=original_layer.dropout,
                bias=original_layer.in_proj_bias is not None,
                device=original_layer.in_proj_weight.device,
                dtype=original_layer.in_proj_weight.dtype,
                **params,
            )

            if copy_weights:
                # Transfer learnable parameters from the original attention layer
                sketched_layer.Wq.data.copy_(
                    original_layer.in_proj_weight[: original_layer.embed_dim, :]
                )
                sketched_layer.Wk.data.copy_(
                    original_layer.in_proj_weight[
                        original_layer.embed_dim : 2 * original_layer.embed_dim, :
                    ]
                )
                sketched_layer.Wv.data.copy_(
                    original_layer.in_proj_weight[
                        2 * original_layer.embed_dim : 3 * original_layer.embed_dim, :
                    ]
                )
                sketched_layer.W0.data.copy_(original_layer.out_proj.weight)

                if original_layer.in_proj_bias is not None and sketched_layer.bias:
                    sketched_layer.bq.data.copy_(
                        original_layer.in_proj_bias[: original_layer.embed_dim]
                    )
                    sketched_layer.bk.data.copy_(
                        original_layer.in_proj_bias[
                            original_layer.embed_dim : 2 * original_layer.embed_dim
                        ]
                    )
                    sketched_layer.bv.data.copy_(
                        original_layer.in_proj_bias[
                            2 * original_layer.embed_dim : 3 * original_layer.embed_dim
                        ]
                    )
                    sketched_layer.b0.data.copy_(original_layer.out_proj.bias)
        else:
            # If the layer type is not nn.Linear, nn.Conv2d, or nn.MultiheadAttention,
            # sketched_layer remains None, and we return it as is.
            return sketched_layer

        # Replace the layer
        setattr(parent, name, sketched_layer)

        if self.verbose:
            print(
                f"Replaced {layer_name} with sketched version using parameters: {params}"
            )

        return sketched_layer

    def _evaluate_model(self, accuracy_score):
        """
        Evaluate the model and calculate the optimization score.

        Args:
            accuracy_score: Accuracy score of the model

        Returns:
            Tuple of (final_score, speed_score)
        """
        speed_score = None
        if (
            self.accuracy_threshold is not None
            and self.optmization_eval_func is not None
        ):
            if accuracy_score >= self.accuracy_threshold:
                speed_score = self.optmization_eval_func(self.model)
                score = speed_score
            else:
                score = float("-inf")
        else:
            score = accuracy_score

        return score, speed_score

    def _try_parameters(self, layer_name, params, copy_weights=True):
        """
        Apply parameters to a layer, evaluate the model, and restore the original layer.

        Args:
            layer_name: Name of the layer to tune
            params: Parameters to try
            copy_weights: Whether to copy weights from original layer

        Returns:
            Tuple of (score, accuracy_score, speed_score, original_layer)
        """
        # Get and store original layer
        layer = self._get_layer_by_name(layer_name)

        # Apply parameters
        new_layer = self._sketch_layer(
            layer_name, type(layer), params, copy_weights=copy_weights
        )

        # Evaluate
        accuracy_score = self.accuracy_eval_func(self.model)
        score, speed_score = self._evaluate_model(accuracy_score)

        return score, accuracy_score, speed_score, layer, new_layer

    def _handle_tune_layer_group_seperate(
        self, config: LayerConfig
    ) -> Dict[str, Dict[str, Any]]:
        layer_params: Dict[str, Dict[str, Any]] = {}

        for layer_name in config.layer_names:
            if self.verbose:
                print(f"Tuning layer: {layer_name}")

            # Initialize search algorithm for this layer
            self.search_algorithm.initialize(config.params)

            best_score = float("-inf")
            best_params = None
            layer_results = []
            best_layer = None

            while True:
                params = self.search_algorithm.get_next_params()
                if params is None:
                    break  # No more parameter combinations to try

                best_score_across_param_runs = float("-inf")
                best_accuracy_score_across_param_runs = float("-inf")
                best_speed_score_across_param_runs = float("-inf")
                best_layer_across_param_runs = None

                if self.verbose:
                    print(f"Trying parameters: {params}")

                for run_n in range(self.num_runs_per_param):
                    # Apply parameters and evaluate
                    score, accuracy_score, speed_score, original_layer, new_layer = (
                        self._try_parameters(
                            layer_name, params, copy_weights=config.copy_weights
                        )
                    )

                    # Restore original layer for next run
                    parent, name = self._get_parent_module_and_name(layer_name)
                    setattr(parent, name, original_layer)

                    # update local things
                    if score > best_score_across_param_runs:
                        best_score_across_param_runs = score
                        best_accuracy_score_across_param_runs = accuracy_score
                        best_speed_score_across_param_runs = speed_score
                        best_layer_across_param_runs = new_layer

                    if self.verbose:
                        print(
                            f"run: {run_n + 1}/{self.num_runs_per_param} - accuracy_score: {accuracy_score}, speed_score: {speed_score}, final score: {score}"
                        )

                # Update search algorithm with average score
                self.search_algorithm.update(params, best_score_across_param_runs)

                # Save result
                result = {"params": params, "score": best_score_across_param_runs}
                layer_results.append(result)

                if best_score_across_param_runs > best_score:
                    best_score = best_score_across_param_runs
                    best_params = params
                    best_layer = best_layer_across_param_runs

                if self.verbose:
                    print(
                        f"Tried parameters: {params}, accuracy_score: {best_accuracy_score_across_param_runs}, speed_score: {best_speed_score_across_param_runs}, final score: {best_score_across_param_runs}"
                    )

            # Store results for this layer
            self.results[layer_name] = layer_results

            # Save best parameters for this layer
            layer_params[layer_name] = {
                "params": best_params,
                "copy_weights": config.copy_weights,
                "best_layer": best_layer,
            }

            if self.verbose:
                print(
                    f"  Best parameters for {layer_name}: {best_params}, score: {best_score}"
                )

        return layer_params

    def _handle_tune_layer_group_joint(
        self, config: LayerConfig
    ) -> Dict[str, Dict[str, Any]]:
        # Initialize search algorithm
        self.search_algorithm.initialize(config.params)

        best_score = float("-inf")
        best_params = None
        all_results = []
        best_layers = None

        while True:
            params = self.search_algorithm.get_next_params()
            if params is None:
                break  # No more parameter combinations to try

            best_score_across_param_runs = float("-inf")
            best_accuracy_score_across_param_runs = float("-inf")
            best_speed_score_across_param_runs = float("-inf")
            best_layers_across_param_runs = None

            if self.verbose:
                print(f"Trying parameters: {params}")

            for run_n in range(self.num_runs_per_param):
                original_layers = []
                new_layers = []
                # Apply parameters to all layers
                for layer_name in config.layer_names:
                    layer = self._get_layer_by_name(layer_name)
                    original_layers.append(layer)
                    new_layer = self._sketch_layer(
                        layer_name,
                        type(layer),
                        params,
                        copy_weights=config.copy_weights,
                    )
                    new_layers.append(new_layer)

                # Evaluate
                accuracy_score = self.accuracy_eval_func(self.model)
                score, speed_score = self._evaluate_model(accuracy_score)

                # Restore original layers
                for i, layer_name in enumerate(config.layer_names):
                    parent, name = self._get_parent_module_and_name(layer_name)
                    original_layer = original_layers[i]
                    setattr(parent, name, original_layer)

                # Check if this run is the best for the current parameter set
                if score > best_score_across_param_runs:
                    best_score_across_param_runs = score
                    best_accuracy_score_across_param_runs = accuracy_score
                    best_speed_score_across_param_runs = speed_score
                    best_layers_across_param_runs = copy.deepcopy(new_layers)

                if self.verbose:
                    print(
                        f"run: {run_n + 1}/{self.num_runs_per_param} - accuracy_score: {accuracy_score}, speed_score: {speed_score}, final score: {score}"
                    )

            # Update search algorithm with best score
            self.search_algorithm.update(params, best_score_across_param_runs)

            # Save result
            result = {"params": params, "score": best_score_across_param_runs}
            all_results.append(result)

            if best_score_across_param_runs > best_score:
                best_score = best_score_across_param_runs
                best_params = params
                best_layers = best_layers_across_param_runs

            if self.verbose:
                print(
                    f"Tried parameters: {params}, accuracy_score: {best_accuracy_score_across_param_runs}, speed_score: {best_speed_score_across_param_runs}, final score: {best_score_across_param_runs}"
                )

        # Store results
        group_key = "_".join(config.layer_names)
        self.results[group_key] = all_results

        # Return best parameters for all layers in the group
        result_dict = {}
        for i, layer_name in enumerate(config.layer_names):
            result_dict[layer_name] = {
                "params": best_params,
                "copy_weights": config.copy_weights,
                "best_layer": best_layers[i] if best_layers else None,
            }

        return result_dict

    def _tune_layer_group(self, config: LayerConfig) -> Dict[str, Dict[str, Any]]:
        """
        Tune a group of layers according to the configuration.

        Args:
            config: Layer configuration

        Returns:
            Dictionary of best parameters for each layer
        """
        # If separate is False, tune all layers together with the same parameters
        if not config.separate:
            return self._handle_tune_layer_group_joint(config)
        # If separate is True, tune each layer individually
        else:
            return self._handle_tune_layer_group_seperate(config)

    def tune(self) -> Dict[str, Dict[str, Any]]:
        """
        Tune all layer groups according to their configurations.

        Returns:
            Dictionary of best parameters for each layer
        """

        # Tune each layer group
        for config in self.configs.configs:
            # tune the layer group
            layer_params = self._tune_layer_group(config)
            self.best_params.update(layer_params)

        return self.best_params

    def apply_best_params(self) -> nn.Module:
        """
        Apply the best parameters to the model.

        Returns:
            The model with the best parameters applied
        """

        # Apply best parameters to each layer
        for layer_name, data in self.best_params.items():
            if data is not None:
                if data["best_layer"] is None:
                    # If no parameters were found, skip this layer
                    if self.verbose:
                        print(f"No parameters found for layer {layer_name}. Skipping.")
                    continue
                layer = self._get_layer_by_name(layer_name)  # noqa: F841
                self._replace_layer(layer_name, data["best_layer"])

        return self.model

    def replace_without_tuning(self) -> nn.Module:
        """
        Replace layers with their sketched versions using the first parameter values without tuning.

        Returns:
            The model with layers replaced
        """

        # Replace each layer with the first parameter values
        for config in self.configs.configs:
            default_params = {
                param: values[0] for param, values in config.params.items()
            }

            for layer_name in config.layer_names:
                layer = self._get_layer_by_name(layer_name)
                self._sketch_layer(
                    layer_name,
                    type(layer),
                    default_params,
                    copy_weights=config.copy_weights,
                )

        return self.model

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get the results of the tuning process as a pandas DataFrame.

        Returns:
            DataFrame with columns for layer name, parameters, and scores
        """
        rows = []

        for layer_name, results in self.results.items():
            for result in results:
                row = {"layer_name": layer_name}
                row.update(result["params"])
                row["score"] = result["score"]
                rows.append(row)

        return pd.DataFrame(rows)

    def get_best_params(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the best parameters for each layer.

        Returns:
            Dictionary of best parameters for each layer
        """
        return self.best_params

    def visualize_tuning_results(
        self,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
        max_cols: int = 2,
    ) -> None:
        """
        Visualize tuning results with plots showing the relationship between parameters and scores.

        Args:
            save_path: Path to save the visualization. If None, the plot won't be saved.
            show_plot: Whether to display the plot.
            figsize: Figure size as (width, height). If None, it will be calculated automatically.
            max_cols: Maximum number of columns in the subplot grid.

        Returns:
            None
        """
        # Get results dataframe
        results_df = self.get_results_dataframe()

        if results_df.empty:
            print("No results to visualize. Run tune() first.")
            return

        # Get unique layer names
        layer_names = results_df["layer_name"].unique()

        # Create dynamic subplots based on number of layers and parameters
        param_columns = [
            col for col in results_df.columns if col not in ["layer_name", "score"]
        ]
        total_plots = 0

        # Calculate total plots needed
        for layer in layer_names:
            layer_df = results_df[results_df["layer_name"] == layer]
            layer_params = [
                p
                for p in param_columns
                if p in layer_df.columns and len(layer_df[p].unique()) > 1
            ]
            total_plots += len(layer_params)

        if total_plots == 0:
            print("No variable parameters found to visualize.")
            return

        # Calculate grid dimensions
        cols = min(max_cols, total_plots)
        rows = (total_plots + cols - 1) // cols

        # Create figure with appropriate size
        if figsize is None:
            figsize = (7 * cols, 4 * rows)

        plt.figure(figsize=figsize)

        # Plot counter
        plot_num = 1

        # For each layer, plot the relationship between each parameter and score
        for layer in layer_names:
            layer_df = results_df[results_df["layer_name"] == layer]
            if layer_df.empty:
                continue

            # Find parameters with multiple values (worth plotting)
            layer_params = [
                p
                for p in param_columns
                if p in layer_df.columns and len(layer_df[p].unique()) > 1
            ]

            for param in layer_params:
                plt.subplot(rows, cols, plot_num)

                # Check if parameter is categorical or numeric
                unique_values = layer_df[param].unique()
                is_categorical = any(
                    isinstance(val, str) for val in unique_values if not pd.isna(val)
                )

                if is_categorical:
                    # Bar plot for categorical parameters
                    grouped = layer_df.groupby(param)["score"].mean()
                    grouped.plot(kind="bar", color="skyblue")
                else:
                    # Line plot for numeric parameters
                    grouped = layer_df.groupby(param)["score"].mean()
                    plt.plot(grouped.index, grouped.values, "o-", color="blue")

                plt.xlabel(param)
                plt.ylabel("Score (higher is better)")
                plt.title(f"{layer}: Effect of {param}")
                plt.grid(True)
                plt.tight_layout()

                plot_num += 1

        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

    def save_tuning_results(self, file_path: str) -> None:
        """
        Save tuning results to a file.

        Args:
            file_path: Path to save the results

        Returns:
            None
        """
        data = {
            "results": self.results,
            "best_params": self.best_params,
            "config": self.configs,
        }

        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, "wb") as f:
            pickle.dump(data, f)

        if self.verbose:
            print(f"Tuning results saved to {file_path}")

    def load_tuning_results(self, file_path: str) -> None:
        """
        Load tuning results from a file.

        Args:
            file_path: Path to load the results from

        Returns:
            None

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file contains invalid data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Results file {file_path} not found")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        if (
            not isinstance(data, dict)
            or "results" not in data
            or "best_params" not in data
        ):
            raise ValueError(f"Invalid data format in {file_path}")

        self.results = data["results"]
        self.best_params = data["best_params"]

        if self.verbose:
            print(f"Tuning results loaded from {file_path}")

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model architecture, including sketched layers.

        Returns:
            Dictionary with model summary information
        """
        total_params = 0
        sketched_layers = 0
        layer_info = []

        # Traverse the model hierarchy
        for name, module in self.model.named_modules():
            if name == "":  # Skip the root module
                continue

            # Calculate parameters
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)

            # Check if it's a sketched layer
            is_sketched = False
            from panther.nn.attention import (
                RandMultiHeadAttention as SKMultiheadAttention,
            )
            from panther.nn.conv2d import SKConv2d
            from panther.nn.linear import SKLinear

            if isinstance(module, (SKLinear, SKConv2d, SKMultiheadAttention)):
                is_sketched = True
                sketched_layers += 1

            # Add layer info
            layer_info.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "params": params,
                    "is_sketched": is_sketched,
                }
            )

            total_params += params

        return {
            "total_params": total_params,
            "sketched_layers": sketched_layers,
            "layers": layer_info,
        }

    def print_model_summary(self) -> None:
        """
        Print a summary of the model architecture, highlighting sketched layers.

        Returns:
            None
        """
        summary = self.get_model_summary()

        print("=" * 80)
        print(
            f"Model Summary (Total trainable parameters: {summary['total_params']:,})"
        )
        print(f"Sketched layers: {summary['sketched_layers']}")
        print("-" * 80)
        print(f"{'Layer Name':<40} {'Layer Type':<25} {'Parameters':<15} {'Sketched'}")
        print("-" * 80)

        for layer in summary["layers"]:
            print(
                f"{layer['name']:<40} {layer['type']:<25} {layer['params']:<15,} {'✓' if layer['is_sketched'] else ''}"
            )
        print("=" * 80)

    def compare_models(self, original_model: nn.Module) -> Dict[str, Any]:
        """
        Compare the tuned model with the original model to analyze differences.

        Args:
            original_model: The original model before tuning

        Returns:
            Dictionary with comparison metrics
        """
        # Collect information about both models
        original_summary = SKAutoTuner(
            original_model, self.configs, self.accuracy_eval_func
        ).get_model_summary()

        tuned_summary = self.get_model_summary()

        # Calculate parameter reduction
        original_params = original_summary["total_params"]
        tuned_params = tuned_summary["total_params"]
        param_reduction = original_params - tuned_params
        param_reduction_percent = (
            (param_reduction / original_params) * 100 if original_params > 0 else 0
        )

        # Collect layer-specific comparisons
        layer_comparisons = []

        # Build a map of layer names to layer info for both models
        original_layers = {layer["name"]: layer for layer in original_summary["layers"]}
        tuned_layers = {layer["name"]: layer for layer in tuned_summary["layers"]}

        # Compare each layer
        all_layer_names = set(original_layers.keys()) | set(tuned_layers.keys())
        for name in all_layer_names:
            if name in original_layers and name in tuned_layers:
                orig = original_layers[name]
                tuned = tuned_layers[name]

                # Only add detailed comparison for layers that changed
                if orig["type"] != tuned["type"] or orig["params"] != tuned["params"]:
                    layer_comparisons.append(
                        {
                            "name": name,
                            "original_type": orig["type"],
                            "tuned_type": tuned["type"],
                            "original_params": orig["params"],
                            "tuned_params": tuned["params"],
                            "param_reduction": orig["params"] - tuned["params"],
                            "param_reduction_percent": (
                                orig["params"] - tuned["params"]
                            )
                            / orig["params"]
                            * 100
                            if orig["params"] > 0
                            else 0,
                        }
                    )

        return {
            "original_params": original_params,
            "tuned_params": tuned_params,
            "param_reduction": param_reduction,
            "param_reduction_percent": param_reduction_percent,
            "layer_comparisons": layer_comparisons,
        }

    def print_comparison_summary(self, original_model: nn.Module) -> None:
        """
        Print a summary comparing the tuned model with the original model.

        Args:
            original_model: The original model before tuning

        Returns:
            None
        """
        comparison = self.compare_models(original_model)

        print("=" * 80)
        print("Model Comparison Summary")
        print("-" * 80)
        print(f"Original model parameters: {comparison['original_params']:,}")
        print(f"Tuned model parameters:    {comparison['tuned_params']:,}")
        print(
            f"Parameter reduction:       {comparison['param_reduction']:,} ({comparison['param_reduction_percent']:.2f}%)"
        )
        print("-" * 80)
        print("Modified Layers:")
        print(
            f"{'Layer Name':<40} {'Original Type':<20} {'Tuned Type':<20} {'Param Reduction'}"
        )
        print("-" * 80)

        for layer in comparison["layer_comparisons"]:
            print(
                f"{layer['name']:<40} {layer['original_type']:<20} {layer['tuned_type']:<20} {layer['param_reduction']:,} ({layer['param_reduction_percent']:.2f}%)"
            )
        print("=" * 80)

    def export_model(self, file_path: str) -> None:
        """
        Export the tuned model to a file.

        Args:
            file_path: Path to save the model

        Returns:
            None
        """
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.model.state_dict(), file_path)

        if self.verbose:
            print(f"Model exported to {file_path}")

    def visualize_parameter_distribution(
        self, save_path: Optional[str] = None, show_plot: bool = True
    ) -> None:
        """
        Visualize the distribution of parameters across different layer types in the model.

        Args:
            save_path: Path to save the visualization. If None, the plot won't be saved.
            show_plot: Whether to display the plot.

        Returns:
            None
        """
        summary = self.get_model_summary()

        # Group layers by type
        layer_types = {}
        for layer in summary["layers"]:
            layer_type = layer["type"]
            if layer_type not in layer_types:
                layer_types[layer_type] = 0
            layer_types[layer_type] += layer["params"]

        # Create visualization
        plt.figure(figsize=(10, 6))

        # Pie chart of parameter distribution
        plt.subplot(1, 2, 1)
        labels = list(layer_types.keys())
        sizes = list(layer_types.values())

        # Filter small values for legibility
        threshold = sum(sizes) * 0.01
        other_size = sum(size for i, size in enumerate(sizes) if size < threshold)
        filtered_labels = [
            label for i, label in enumerate(labels) if sizes[i] >= threshold
        ]
        filtered_sizes = [size for size in sizes if size >= threshold]

        if other_size > 0:
            filtered_labels.append("Other")
            filtered_sizes.append(other_size)

        plt.pie(
            filtered_sizes, labels=filtered_labels, autopct="%1.1f%%", startangle=90
        )
        plt.axis("equal")
        plt.title("Parameter Distribution by Layer Type")

        # Bar chart comparing sketched vs non-sketched
        plt.subplot(1, 2, 2)
        sketched_params = sum(
            layer["params"] for layer in summary["layers"] if layer["is_sketched"]
        )
        non_sketched_params = sum(
            layer["params"] for layer in summary["layers"] if not layer["is_sketched"]
        )

        plt.bar(["Non-Sketched", "Sketched"], [non_sketched_params, sketched_params])
        plt.ylabel("Number of Parameters")
        plt.title("Parameters in Sketched vs Non-Sketched Layers")

        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path)
            print(f"Parameter distribution visualization saved to {save_path}")

        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()

    def get_inference_benchmark(
        self, input_data: torch.Tensor, num_runs: int = 100, warm_up: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference speed of the model.

        Args:
            input_data: Input tensor to use for benchmarking
            num_runs: Number of inference runs to perform
            warm_up: Number of warm-up runs before timing

        Returns:
            Dictionary with benchmark results
        """
        import time

        import numpy as np

        # Move to the same device as the model
        device = next(self.model.parameters()).device
        input_data = input_data.to(device)

        # Warm-up runs
        with torch.no_grad():
            for _ in range(warm_up):
                self.model(input_data)

        # Actual benchmark runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                self.model(input_data)
                # Ensure synchronization if using GPU
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)

        # Calculate statistics
        mean_time = np.mean(times)
        median_time = np.median(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        return {
            "mean_time": float(mean_time),
            "median_time": float(median_time),
            "std_time": float(std_time),
            "min_time": float(min_time),
            "max_time": float(max_time),
            "fps": float(1.0 / mean_time),
        }
