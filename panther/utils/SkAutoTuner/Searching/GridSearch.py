from typing import Any, Dict, List

from panther.utils.SkAutoTuner.Searching import SearchAlgorithm

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