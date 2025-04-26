import copy
import json
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from panther.nn.linear import SKLinear
from panther.nn.conv2d import SKConv2d
from panther.nn.attention import RandMultiHeadAttention


# Layer type mapping to their sketched versions and parameters
LAYER_TYPE_MAPPING = {
    nn.Linear: {
        "class": SKLinear,
        "params": ["num_terms", "low_rank"],
    },
    nn.Conv2d: {
        "class": SKConv2d,
        "params": ["num_terms", "low_rank"],
    },
    nn.MultiheadAttention: {
        "class": RandMultiHeadAttention,
        "params": ["num_random_features", "kernel_fn"],
    }
}


class LayerConfig:
    """
    Configuration object for a single layer or group of layers.
    Contains the layer names, parameters to tune and whether these layers should be tuned separately.
    """
    def __init__(
        self, 
        layer_names: List[str], 
        params: Dict[str, List], 
        separate: bool = True
    ):
        """
        Initialize a layer configuration.
        
        Args:
            layer_names: List of names of layers to configure (dot notation for nested modules)
            params: Dictionary of parameter names and their possible values to try
            separate: Whether these layers should be tuned separately or together
        """
        self.layer_names = layer_names
        self.params = params
        self.separate = separate
    
    def __repr__(self):
        return f"LayerConfig(layer_names={self.layer_names}, params={self.params}, separate={self.separate})"


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


class SearchAlgorithm(ABC):
    """
    Abstract base class for search algorithms to use in autotuning.
    """
    @abstractmethod
    def initialize(self, param_space: Dict[str, List]):
        """
        Initialize the search algorithm with the parameter space.
        
        Args:
            param_space: Dictionary of parameter names and their possible values
        """
        pass
    
    @abstractmethod
    def get_next_params(self) -> Dict[str, Any]:
        """
        Get the next set of parameters to try.
        
        Returns:
            Dictionary of parameter names and values to try
        """
        pass
    
    @abstractmethod
    def update(self, params: Dict[str, Any], score: float):
        """
        Update the search algorithm with the results of the latest trial.
        
        Args:
            params: Dictionary of parameter names and values that were tried
            score: The evaluation score for the parameters
        """
        pass
    
    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best parameters found so far.
        
        Returns:
            Dictionary of parameter names and their best values
        """
        pass


class GridSearch(SearchAlgorithm):
    """
    Grid search algorithm that tries all combinations of parameters.
    """
    def __init__(self):
        self.param_space = {}
        self.param_combinations = []
        self.current_idx = 0
        self.results = []
        self.best_score = float('-inf')
        self.best_params = {}
    
    def initialize(self, param_space: Dict[str, List]):
        self.param_space = param_space
        self._generate_combinations()
    
    def _generate_combinations(self):
        """Generate all combinations of parameters."""
        from itertools import product
        
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())
        
        for combination in product(*values):
            self.param_combinations.append(dict(zip(keys, combination)))
    
    def get_next_params(self) -> Dict[str, Any]:
        if self.current_idx >= len(self.param_combinations):
            return None  # All combinations tried
        
        params = self.param_combinations[self.current_idx]
        self.current_idx += 1
        return params
    
    def update(self, params: Dict[str, Any], score: float):
        self.results.append((params, score))
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
    
    def get_best_params(self) -> Dict[str, Any]:
        return self.best_params


class RandomSearch(SearchAlgorithm):
    """
    Random search algorithm that randomly samples from the parameter space.
    """
    def __init__(self, max_trials: int = 20):
        self.param_space = {}
        self.max_trials = max_trials
        self.current_trial = 0
        self.results = []
        self.best_score = float('-inf')
        self.best_params = {}
    
    def initialize(self, param_space: Dict[str, List]):
        self.param_space = param_space
    
    def get_next_params(self) -> Dict[str, Any]:
        if self.current_trial >= self.max_trials:
            return None  # All trials completed
        
        self.current_trial += 1
        return {
            param: np.random.choice(values) 
            for param, values in self.param_space.items()
        }
    
    def update(self, params: Dict[str, Any], score: float):
        self.results.append((params, score))
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
    
    def get_best_params(self) -> Dict[str, Any]:
        return self.best_params


class BayesianOptimization(SearchAlgorithm):
    """
    Bayesian optimization search algorithm.
    """
    def __init__(self, max_trials: int = 20):
        self.param_space = {}
        self.max_trials = max_trials
        self.current_trial = 0
        self.results = []
        self.best_score = float('-inf')
        self.best_params = {}
        self._param_mapping = {}  # Maps parameter names to indices
        self._param_inv_mapping = {}  # Maps indices to parameter names
        self.observed_X = []
        self.observed_y = []
    
    def initialize(self, param_space: Dict[str, List]):
        self.param_space = param_space
        
        # Create mappings for parameters to make them numeric
        for i, (param, values) in enumerate(self.param_space.items()):
            self._param_mapping[param] = i
            self._param_inv_mapping[i] = param
    
    def _params_to_point(self, params: Dict[str, Any]) -> List[int]:
        """Convert a parameter dictionary to a point in the search space."""
        point = [0] * len(self._param_mapping)
        for param, value in params.items():
            idx = self._param_mapping[param]
            options = self.param_space[param]
            value_idx = options.index(value)
            point[idx] = value_idx
        return point
    
    def _point_to_params(self, point: List[int]) -> Dict[str, Any]:
        """Convert a point in the search space to a parameter dictionary."""
        params = {}
        for i, value_idx in enumerate(point):
            param = self._param_inv_mapping[i]
            options = self.param_space[param]
            params[param] = options[value_idx % len(options)]
        return params
    
    def _acquisition_function(self, x: np.ndarray) -> float:
        """
        Upper Confidence Bound (UCB) acquisition function.
        This is a simple implementation; a real-world one would use Gaussian processes.
        """
        if not self.observed_X:
            return 0.0
        
        # Distance to closest observed point
        distances = [np.sum(np.abs(x - obs_x)) for obs_x in self.observed_X]
        closest_idx = np.argmin(distances)
        min_distance = distances[closest_idx]
        
        # UCB value: exploit + explore
        exploit = self.observed_y[closest_idx]
        explore = np.sqrt(min_distance)
        
        return exploit + 0.1 * explore
    
    def get_next_params(self) -> Dict[str, Any]:
        if self.current_trial >= self.max_trials:
            return None  # All trials completed
        
        self.current_trial += 1
        
        # If we have fewer than 3 observations, use random sampling
        if len(self.observed_X) < 3:
            return {
                param: np.random.choice(values) 
                for param, values in self.param_space.items()
            }
        
        # Otherwise, use Bayesian Optimization
        # For simplicity, we'll just try 100 random candidates and pick the best
        # A real implementation would use more sophisticated methods
        best_acq_value = float('-inf')
        best_point = None
        
        for _ in range(100):
            candidate = [
                np.random.randint(0, len(self.param_space[self._param_inv_mapping[i]]))
                for i in range(len(self._param_mapping))
            ]
            acq_value = self._acquisition_function(np.array(candidate))
            
            if acq_value > best_acq_value:
                best_acq_value = acq_value
                best_point = candidate
        
        return self._point_to_params(best_point)
    
    def update(self, params: Dict[str, Any], score: float):
        self.results.append((params, score))
        
        # Convert params to point
        point = self._params_to_point(params)
        self.observed_X.append(np.array(point))
        self.observed_y.append(score)
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = params
    
    def get_best_params(self) -> Dict[str, Any]:
        return self.best_params


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
    
    def replace_layer(self, layer_name: str, layer_type: Type[nn.Module], params: Dict[str, Any]) -> bool:
        """
        Replace a layer with its sketched version.
        
        Args:
            layer_name: Name of the layer to replace
            layer_type: Type of the layer
            params: Parameters for the sketched layer
            
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
                W_init=original_layer.weight,
                **params
            )
        elif isinstance(original_layer, nn.Conv2d):
            sketched_layer = sketched_class(
                in_channels=original_layer.in_channels,
                out_channels=original_layer.out_channels,
                kernel_size=original_layer.kernel_size,
                stride=original_layer.stride,
                padding=original_layer.padding,
                **params
            )
        elif isinstance(original_layer, nn.MultiheadAttention):
            sketched_layer = sketched_class(
                embed_dim=original_layer.embed_dim,
                num_heads=original_layer.num_heads,
                dropout=original_layer.dropout,
                bias=original_layer.in_proj_bias is not None,
                **params
            )
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