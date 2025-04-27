from typing import Dict, List
from .LayerConfig import LayerConfig

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