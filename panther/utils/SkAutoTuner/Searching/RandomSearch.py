from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from panther.utils.SkAutoTuner.Searching import SearchAlgorithm
import numpy as np

class RandomSearch(SearchAlgorithm):
    """
    Random search algorithm that randomly samples from the parameter space.
    """
    def __init__(self, max_trials: int = 20):
        self.param_space = {}
        self.max_trials = max_trials
        self.current_trial = 0
    
    def initialize(self, param_space: Dict[str, List]):
        self.param_space = param_space
        self.current_trial = 0
    
    def get_next_params(self) -> Dict[str, Any]:
        if self.current_trial >= self.max_trials:
            return None  # All trials completed
        
        self.current_trial += 1
        return {
            param: np.random.choice(values) 
            for param, values in self.param_space.items()
        }
    
    def update(self, params: Dict[str, Any], score: float):
        # no need to do anything
        pass