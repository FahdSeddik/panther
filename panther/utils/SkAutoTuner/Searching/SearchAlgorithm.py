from abc import ABC, abstractmethod
from typing import Any, Dict, List

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