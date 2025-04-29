from typing import Dict, List, Union, Any

class LayerConfig:
    """
    Configuration object for a single layer or group of layers.
    Contains the layer names, parameters to tune and whether these layers should be tuned separately.
    """
    def __init__(
        self, 
        layer_names: Union[str, List[str], Dict[str, Union[str, List[str], int, List[int]]]], 
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