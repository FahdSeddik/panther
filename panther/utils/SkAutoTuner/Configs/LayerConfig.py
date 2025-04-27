from typing import Dict, List

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