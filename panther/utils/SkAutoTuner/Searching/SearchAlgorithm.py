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
    def save_state(self, filepath: str):
        """
        Save the current state of the search algorithm to a file.

        Args:
            filepath: The path to the file where the state should be saved.
        """
        pass

    @abstractmethod
    def load_state(self, filepath: str):
        """
        Load the state of the search algorithm from a file.

        Args:
            filepath: The path to the file from which the state should be loaded.
        """
        pass

    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best set of parameters found so far.

        Returns:
            Dictionary of the best parameter names and values.
        """
        pass

    @abstractmethod
    def get_best_score(self) -> float:
        """
        Get the best score achieved so far.

        Returns:
            The best score.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the search algorithm to its initial state.
        """
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        """
        Check if the search algorithm has finished its search (e.g., budget exhausted).

        Returns:
            True if the search is finished, False otherwise.
        """
        pass