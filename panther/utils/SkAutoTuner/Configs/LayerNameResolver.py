""" USAGE
The LayerNameResolver provides intuitive ways to select layers for tuning.
It supports several types of selectors:

1. Single String Pattern
   - Regex pattern: "encoder.*attention"
   - Simple substring: "linear"
   
   Example:
   resolver.resolve("encoder")  # All layers with "encoder" in their name

2. List of String Patterns
   - Combines results from multiple patterns
   
   Example:
   resolver.resolve(["encoder", "decoder"])  # All encoder and decoder layers

3. Dictionary with Selection Criteria
   - Most flexible option

   a. Pattern-based Selection:
      resolver.resolve({"pattern": "encoder.*"})  # All encoder layers
      resolver.resolve({"pattern": ["encoder.*", "decoder.*"]})  # All encoder and decoder layers

   b. Type-based Selection:
      resolver.resolve({"type": "Linear"})  # All linear layers
      resolver.resolve({"type": ["Conv2d", "ConvTranspose2d"]})  # All Conv2d and ConvTranspose2d layers

   c. Contains-based Selection:
      resolver.resolve({"contains": "attention"})  # All layers with "attention" in their name

   d. Index-based Selection:
      resolver.resolve({"pattern": "encoder.*", "indices": [0, 2, 4]})  # First, third, and fifth encoder layers

   e. Range-based Selection:
      resolver.resolve({"pattern": "encoder.layer.*", "range": [0, 6]})  # First 6 encoder layers
      resolver.resolve({"pattern": "encoder.layer.*", "range": [0, 12, 2]})  # Even-indexed encoder layers

   f. Combined Criteria:
      resolver.resolve({
          "pattern": "encoder.*",
          "type": "Linear",
          "range": [0, 4]
      })  # First 4 linear layers in the encoder

In SKAutoTuner usage, these selectors are passed directly to the LayerConfig:

# Example configurations using different selector types
configs = TuningConfigs([
    # Select all linear layers
    LayerConfig(
        layer_names={"type": "Linear"},
        params={"low_rank": [16, 32], "num_terms": [2, 3]},
        separate=True
    ),
    
    # Select attention layers in encoder
    LayerConfig(
        layer_names={"pattern": "encoder.*attention"},
        params={"low_rank": [32, 64], "num_terms": [2]},
        separate=False
    ),
    
    # Select even-indexed transformer layers
    LayerConfig(
        layer_names={"pattern": "transformer.layers.*", "range": [0, 12, 2]},
        params={"low_rank": [64], "num_terms": [2]},
        separate=True
    )
])

# Create autotuner with these configurations
tuner = SKAutoTuner(model, configs, eval_func)
"""

import re
from typing import List, Union, Dict, Set, Optional
import torch.nn as nn

class LayerNameResolver:
    """
    Provides intuitive ways to select layers for tuning in large models.
    
    This resolver allows users to specify layers using patterns, types, or indices
    rather than requiring exact layer names.
    """
    
    def __init__(self, model: nn.Module, layer_map: Optional[Dict[str, nn.Module]] = None):
        """
        Initialize the resolver with a model.
        
        Args:
            model: The neural network model to analyze
        """
        self.model = model
        self.layer_map = layer_map if layer_map is not None else {}
    
    def resolve(self, selectors: Union[str, List[str], Dict[str, Union[str, List[str], int, List[int]]]]) -> List[str]:
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
                    - 'indices': Indices to select from matched layers 
                      (e.g., [0, 2, 4] for first, third, fifth)
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
            matched_layers = set(self.layer_map.keys())
            
            # Filter by pattern
            if 'pattern' in selectors:
                patterns = selectors['pattern']
                if isinstance(patterns, str):
                    patterns = [patterns]
                
                pattern_matches = set()
                for pattern in patterns:
                    pattern_matches.update(self._resolve_pattern(pattern))
                
                matched_layers = matched_layers.intersection(pattern_matches)
            
            # Filter by type
            if 'type' in selectors:
                types = selectors['type']
                if isinstance(types, str):
                    types = [types]
                
                type_matches = set()
                for layer_name, layer in self.layer_map.items():
                    layer_type = type(layer).__name__
                    if layer_type in types:
                        type_matches.add(layer_name)
                
                matched_layers = matched_layers.intersection(type_matches)
            
            # Filter by containing string
            if 'contains' in selectors:
                contains = selectors['contains']
                contains_matches = {name for name in matched_layers if contains in name}
                matched_layers = matched_layers.intersection(contains_matches)
            
            # Convert to list for indexing operations
            matched_list = sorted(list(matched_layers))
            
            # Apply indices filter
            if 'indices' in selectors:
                indices = selectors['indices']
                if isinstance(indices, int):
                    indices = [indices]
                selected_layers = [matched_list[i] for i in indices if i < len(matched_list)]
                return selected_layers
            
            # Apply range filter
            if 'range' in selectors:
                start, end, *step = selectors['range']
                step = step[0] if step else 1
                selected_layers = matched_list[start:end:step]
                return selected_layers
            
            return matched_list
        
        else:
            raise ValueError("Selectors must be a string, list of strings, or a dictionary with selection criteria")
    
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
    
    def get_layer_info(self, include_parameters: bool = False) -> Dict:
        """
        Get information about layers in the model.
        
        Args:
            include_parameters: Whether to include parameter counts
            
        Returns:
            Dictionary with layer types and counts
        """
        layer_types = {}
        for name, module in self.layer_map.items():
            layer_type = type(module).__name__
            if layer_type not in layer_types:
                layer_types[layer_type] = []
            
            if include_parameters:
                param_count = sum(p.numel() for p in module.parameters())
                layer_types[layer_type].append((name, param_count))
            else:
                layer_types[layer_type].append(name)
        
        return layer_types