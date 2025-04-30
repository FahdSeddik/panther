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
        self.param_combinations = []
    
    def initialize(self, param_space: Dict[str, List]):
        self.param_space = param_space
        self.current_trial = 0
        self.param_combinations = []
        self._generate_combinations()

    def _generate_combinations(self):
        """Generate all combinations of parameters."""
        from itertools import product
        
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())
        
        for combination in product(*values):
            self.param_combinations.append(dict(zip(keys, combination)))
    
    def get_next_params(self) -> Dict[str, Any]:
        if self.current_trial >= self.max_trials or len(self.param_combinations) == 0:
            return None  # All trials completed
        
        self.current_trial += 1
        choice = np.random.randint(0, len(self.param_combinations))
        # Remove and return the chosen combination
        selected_params = self.param_combinations.pop(choice)
        return selected_params
    
    def update(self, params: Dict[str, Any], score: float):
        # no need to do anything
        pass