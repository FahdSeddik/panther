import copy
import pickle
from typing import Any, Callable, Dict, Tuple, Type

import pandas as pd
import torch
import torch.nn as nn

from .Searching import SearchAlgorithm, GridSearch
from .layer_type_mapping import LAYER_TYPE_MAPPING
from .Configs import TuningConfigs, LayerConfig

class SKAutoTuner:
    """
    Auto-tuner for sketched neural network layers.
    """
    def __init__(
        self, 
        model: nn.Module, 
        configs: TuningConfigs,
        eval_func: Callable[[nn.Module], float],
        search_algorithm: SearchAlgorithm = None,
        verbose: bool = False
    ):
        """
        Initialize the autotuner.
        
        Args:
            model: The neural network model to tune
            configs: Configuration for the layers to tune
            eval_func: Evaluation function that takes a model and returns a score (higher is better)
            search_algorithm: Search algorithm to use for finding optimal parameters
            verbose: Whether to print progress during tuning
        """
        self.model = model
        self.configs = configs
        self.eval_func = eval_func
        self.search_algorithm = search_algorithm or GridSearch()
        self.verbose = verbose
        
        # Store original model state
        self.original_model_state = copy.deepcopy(model.state_dict())
        
        # Dictionary to store results for each layer
        self.results = {}
        
        # Dictionary to store best parameters for each layer
        self.best_params = {}
        
        # Map from layer name to layer object
        self.layer_map = {}
        self._build_layer_map(model)
        self._check_configs()

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
                raise ValueError(f"Config {i} for layers {config.layer_names} has no parameters to tune")
            
            # Get actual layer objects and check compatibility
            layers = []
            for layer_name in config.layer_names:
                try:
                    layer = self._get_layer_by_name(layer_name)
                    layers.append(layer)
                except ValueError as e:
                    raise ValueError(f"Layer '{layer_name}' in config {i} not found in model: {e}")
            
            # Check each parameter is valid for the layer types
            for layer in layers:
                layer_type = type(layer)
                if layer_type not in LAYER_TYPE_MAPPING:
                    raise ValueError(f"Layer '{layer_name}' is of type {layer_type.__name__}, which is not supported for sketching")
                
                valid_params = LAYER_TYPE_MAPPING[layer_type]["params"]
                for param in config.params:
                    if param not in valid_params:
                        raise ValueError(f"Parameter '{param}' is not valid for layer type {layer_type.__name__}. Valid parameters are: {valid_params}")
            
            # Check parameter compatibility across layers in the group
            if len(config.layer_names) > 1:
                param_sets = [set(LAYER_TYPE_MAPPING[type(layer)]["params"]) for layer in layers]
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
        if "." not in layer_name:
            return self.model, layer_name
        
        parent_name, child_name = layer_name.rsplit(".", 1)
        parent = self._get_module_by_name(parent_name)
        return parent, child_name
    
    def _get_module_by_name(self, name: str) -> nn.Module:
        """Get a module by its name."""
        if not name:
            return self.model
        
        for n, m in self.model.named_modules():
            if n == name:
                return m
        
        raise ValueError(f"Module {name} not found in the model")
    
    def replace_layer(self, layer_name: str, layer_type: Type[nn.Module], params: Dict[str, Any], copy_weights: bool = False) -> bool:
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
        
        # Check if the layer type is supported
        if type(original_layer) not in LAYER_TYPE_MAPPING:
            if self.verbose:
                print(f"Layer type {type(original_layer).__name__} is not supported for sketching")
            return False
        
        # Get the sketched layer class and its parameters
        sketched_class = LAYER_TYPE_MAPPING[type(original_layer)]["class"]
        
        # Create the sketched layer with original parameters and new sketching parameters
        if isinstance(original_layer, nn.Linear):
            sketched_layer = sketched_class(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                bias=original_layer.bias is not None,
                W_init=original_layer.weight if copy_weights else None,
                device=original_layer.weight.device,
                dtype=original_layer.weight.dtype,
                **params
            )
    
            # Copy bias if needed
            if copy_weights and original_layer.bias is not None and sketched_layer.bias is not None:
                sketched_layer.bias.data.copy_(original_layer.bias.data)
            # for the weights its handled in the SKLinear constructor

        elif isinstance(original_layer, nn.Conv2d):
            # Create a new sketched convolution layer
            sketched_layer = sketched_class(
                in_channels=original_layer.in_channels,
                out_channels=original_layer.out_channels,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=original_layer.padding,
                device=original_layer.weight.device,
                dtype=original_layer.weight.dtype,
                **params
            )
            
            if copy_weights:
                # Initialize S1s and S2s based on original weights
                kernels = original_layer.weight.detach().clone()
                kernels = kernels.permute(1, 2, 3, 0)  # Convert to expected shape
                
                for i in range(sketched_layer.num_terms):
                    sketched_layer.S1s.data[i] = torch.matmul(
                        kernels.reshape(-1, kernels.shape[-1]), 
                        sketched_layer.U1s[i].T
                    )
                    
                    K_mat4 = kernels.reshape(-1, kernels.shape[-1])
                    sketched_layer.S2s.data[i] = torch.matmul(
                        sketched_layer.U2s[i], 
                        K_mat4
                    ).reshape(sketched_layer.low_rank, -1)
                
                # Copy bias
                if original_layer.bias is not None:
                    sketched_layer.bias.data.copy_(original_layer.bias.data)
                
        elif isinstance(original_layer, nn.MultiheadAttention):
            # Create a new sketched attention layer
            sketched_layer = sketched_class(
                embed_dim=original_layer.embed_dim,
                num_heads=original_layer.num_heads,
                dropout=original_layer.dropout,
                bias=original_layer.in_proj_bias is not None,
                device=original_layer.in_proj_weight.device,
                dtype=original_layer.in_proj_weight.dtype,
                **params
            )
            
            if copy_weights:
                # Transfer learnable parameters from the original attention layer
                sketched_layer.Wq.data.copy_(original_layer.in_proj_weight[:original_layer.embed_dim, :])
                sketched_layer.Wk.data.copy_(original_layer.in_proj_weight[original_layer.embed_dim:2*original_layer.embed_dim, :])
                sketched_layer.Wv.data.copy_(original_layer.in_proj_weight[2*original_layer.embed_dim:3*original_layer.embed_dim, :])
                sketched_layer.W0.data.copy_(original_layer.out_proj.weight)
                
                if original_layer.in_proj_bias is not None and sketched_layer.bias:
                    sketched_layer.bq.data.copy_(original_layer.in_proj_bias[:original_layer.embed_dim])
                    sketched_layer.bk.data.copy_(original_layer.in_proj_bias[original_layer.embed_dim:2*original_layer.embed_dim])
                    sketched_layer.bv.data.copy_(original_layer.in_proj_bias[2*original_layer.embed_dim:3*original_layer.embed_dim])
                    sketched_layer.b0.data.copy_(original_layer.out_proj.bias)
        else:
            return False
        
        # Replace the layer
        setattr(parent, name, sketched_layer)
        
        if self.verbose:
            print(f"Replaced {layer_name} with sketched version using parameters: {params}")
        
        return True
    
    def tune_layer_group(self, config: LayerConfig) -> Dict[str, Dict[str, Any]]:
        """
        Tune a group of layers according to the configuration.
        
        Args:
            config: Layer configuration
            
        Returns:
            Dictionary of best parameters for each layer
        """
        # If separate is False, tune all layers together with the same parameters
        if not config.separate:
            # Initialize search algorithm
            self.search_algorithm.initialize(config.params)
            
            best_score = float('-inf')
            best_params = None
            all_results = []
            
            while True:
                params = self.search_algorithm.get_next_params()
                if params is None:
                    break  # No more parameter combinations to try
                
                # Apply parameters to all layers
                for layer_name in config.layer_names:
                    layer = self._get_layer_by_name(layer_name)
                    self.replace_layer(layer_name, type(layer), params)
                
                # Evaluate
                score = self.eval_func(self.model)
                
                # Update search algorithm
                self.search_algorithm.update(params, score)
                
                # Save result
                result = {"params": params, "score": score}
                all_results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                if self.verbose:
                    print(f"Tried parameters: {params}, score: {score}")
            
            # Store results
            group_key = "_".join(config.layer_names)
            self.results[group_key] = all_results
            
            # Return best parameters for all layers in the group
            return {
                layer_name: best_params for layer_name in config.layer_names
            }
        
        # If separate is True, tune each layer individually
        else:
            layer_params = {}
            
            for layer_name in config.layer_names:
                if self.verbose:
                    print(f"Tuning layer: {layer_name}")
                
                # Initialize search algorithm for this layer
                self.search_algorithm.initialize(config.params)
                
                best_score = float('-inf')
                best_params = None
                layer_results = []
                
                while True:
                    params = self.search_algorithm.get_next_params()
                    if params is None:
                        break  # No more parameter combinations to try
                    
                    # Reset model to original state before applying new parameters
                    self.model.load_state_dict(self.original_model_state)
                    
                    # Apply parameters to this layer
                    layer = self._get_layer_by_name(layer_name)
                    self.replace_layer(layer_name, type(layer), params)
                    
                    # Evaluate
                    score = self.eval_func(self.model)
                    
                    # Update search algorithm
                    self.search_algorithm.update(params, score)
                    
                    # Save result
                    result = {"params": params, "score": score}
                    layer_results.append(result)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                    
                    if self.verbose:
                        print(f"  Tried parameters: {params}, score: {score}")
                
                # Store results for this layer
                self.results[layer_name] = layer_results
                
                # Save best parameters for this layer
                layer_params[layer_name] = best_params
                
                if self.verbose:
                    print(f"  Best parameters for {layer_name}: {best_params}, score: {best_score}")
            
            return layer_params
    
    def tune(self) -> Dict[str, Dict[str, Any]]:
        """
        Tune all layer groups according to their configurations.
        
        Returns:
            Dictionary of best parameters for each layer
        """
        # Store original model state
        self.original_model_state = copy.deepcopy(self.model.state_dict())
        
        # Tune each layer group
        for config in self.configs.configs:
            layer_params = self.tune_layer_group(config)
            self.best_params.update(layer_params)
        
        # Reset model to original state
        self.model.load_state_dict(self.original_model_state)
        
        return self.best_params
    
    def apply_best_params(self) -> nn.Module:
        """
        Apply the best parameters to the model.
        
        Returns:
            The model with the best parameters applied
        """
        # Reset model to original state
        self.model.load_state_dict(self.original_model_state)
        
        # Apply best parameters to each layer
        for layer_name, params in self.best_params.items():
            if params is not None:
                layer = self._get_layer_by_name(layer_name)
                self.replace_layer(layer_name, type(layer), params)
        
        return self.model
    
    def replace_without_tuning(self) -> nn.Module:
        """
        Replace layers with their sketched versions using the first parameter values without tuning.
        
        Returns:
            The model with layers replaced
        """
        # Reset model to original state
        self.model.load_state_dict(self.original_model_state)
        
        # Replace each layer with the first parameter values
        for config in self.configs.configs:
            default_params = {
                param: values[0] 
                for param, values in config.params.items()
            }
            
            for layer_name in config.layer_names:
                layer = self._get_layer_by_name(layer_name)
                self.replace_layer(layer_name, type(layer), default_params)
        
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
    
    def save(self, filepath: str):
        """
        Save the model and best parameters to a file.
        
        Args:
            filepath: Path to save the model and parameters
        """
        save_data = {
            "model_state": self.model.state_dict(),
            "best_params": self.best_params,
            "results": self.results
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)
        
        if self.verbose:
            print(f"Model and parameters saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, model: nn.Module, eval_func: Callable = None, verbose: bool = False) -> 'SKAutoTuner':
        """
        Load a model and best parameters from a file.
        
        Args:
            filepath: Path to load the model and parameters from
            model: Model to apply the parameters to
            eval_func: Evaluation function (optional)
            verbose: Whether to print progress
            
        Returns:
            SKAutoTuner instance with the loaded model and parameters
        """
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)
        
        # Create an empty configs object (not needed for loading)
        configs = TuningConfigs([])
        
        # Create a new SKAutoTuner instance
        autotuner = cls(model, configs, eval_func or (lambda m: 0), verbose=verbose)
        
        # Load the model state and best parameters
        model.load_state_dict(save_data["model_state"])
        autotuner.best_params = save_data["best_params"]
        autotuner.results = save_data["results"]
        
        if verbose:
            print(f"Model and parameters loaded from {filepath}")
        
        return autotuner