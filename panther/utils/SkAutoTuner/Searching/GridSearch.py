import pickle  # Used for serialization in save_state and load_state
from typing import Any, Dict, List

from .SearchAlgorithm import SearchAlgorithm


class GridSearch(SearchAlgorithm):
    """
    Grid search algorithm that systematically tries all combinations of parameters.

    This implementation performs an exhaustive search through all possible parameter
    combinations in the defined parameter space. It keeps track of the best parameters
    found and their corresponding score throughout the search process.

    Attributes:
        param_space: Dictionary mapping parameter names to lists of possible values
        current_idx: Current index in the parameter combinations list
        param_combinations: List of all possible parameter combinations
        history: List of dictionaries containing tried parameters and their scores
        best_score: The highest score achieved so far
        best_params: The parameters that achieved the best score
    """

    def __init__(self):
        """Initialize the GridSearch algorithm with default values."""
        self.param_space: Dict[str, List] = {}
        self.current_idx = 0
        self.param_combinations: List[Dict[str, Any]] = []
        self.history: List[
            Dict[str, Any]
        ] = []  # Stores {'params': params, 'score': score}
        self.best_score: float = -float("inf")  # Assuming higher score is better
        self.best_params: Dict[str, Any] | None = None

    def initialize(self, param_space: Dict[str, List]):
        """
        Initialize the grid search with a parameter space.

        Args:
            param_space: Dictionary mapping parameter names to lists of possible values
        """
        self.current_idx = 0
        self.param_space = param_space
        self.param_combinations = []
        self.history: List[Dict[str, Any]] = []
        self.best_score: float = -float("inf")  # Assuming higher score is better
        self.best_params: Dict[str, Any] | None = None
        self._generate_combinations()

    def _generate_combinations(self):
        """
        Generate all possible combinations of parameters from the parameter space.

        Uses itertools.product to create the Cartesian product of all parameter values.
        """
        from itertools import product

        keys = list(self.param_space.keys())
        values = list(self.param_space.values())

        # Create all possible combinations using Cartesian product
        for combination in product(*values):
            self.param_combinations.append(dict(zip(keys, combination)))

    def get_next_params(self) -> Dict[str, Any] | None:
        """
        Get the next set of parameters to try.

        Returns:
            Dictionary of parameter names and values to try next, or None if all combinations have been tried
        """
        if self.current_idx >= len(self.param_combinations):
            return None  # All combinations tried

        params = self.param_combinations[self.current_idx]
        self.current_idx += 1
        return params

    def update(self, params: Dict[str, Any], score: float):
        """
        Update the search algorithm with the results of the latest trial.

        Stores the parameters and score in history and updates the best parameters
        if the current score is better than the previous best.

        Args:
            params: Dictionary of parameter names and values that were tried
            score: The evaluation score for the parameters
        """
        self.history.append({"params": params.copy(), "score": score})
        # Assuming higher score is better. If lower is better, change to score < self.best_score
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()

    def save_state(self, filepath: str):
        """
        Save the current state of the search algorithm to a file.

        Serializes the entire state using pickle to enable resuming the search later.

        Args:
            filepath: The path to the file where the state should be saved.
        """
        state = {
            "param_space": self.param_space,
            "current_idx": self.current_idx,
            "param_combinations": self.param_combinations,
            "history": self.history,
            "best_score": self.best_score,
            "best_params": self.best_params,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """
        Load the state of the search algorithm from a file.

        Restores a previously saved state to continue the search process.

        Args:
            filepath: The path to the file from which the state should be loaded.
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        # Restore all state variables
        self.param_space = state["param_space"]
        self.current_idx = state["current_idx"]
        self.param_combinations = state["param_combinations"]
        self.history = state["history"]
        self.best_score = state["best_score"]
        self.best_params = state["best_params"]

    def get_best_params(self) -> Dict[str, Any] | None:
        """
        Get the best set of parameters found so far.

        Returns:
            Dictionary of the best parameter names and values, or None if no parameters have been evaluated.
        """
        return self.best_params

    def get_best_score(self) -> float:
        """
        Get the best score achieved so far.

        Returns:
            The best score. Returns -float('inf') if no trials have been updated or if reset.
        """
        return self.best_score

    def reset(self):
        """
        Reset the search algorithm to its initial state, preserving param_space.

        This allows reusing the same GridSearch instance for multiple searches
        without regenerating parameter combinations.
        """
        self.current_idx = 0
        # self.param_combinations are not regenerated here as they are fixed for GridSearch once initialized
        self.history = []
        self.best_score = -float("inf")  # Assuming higher score is better
        self.best_params = None
        # If param_space is available, param_combinations would have been set by initialize
        # and _generate_combinations. Resetting current_idx is sufficient.

    def is_finished(self) -> bool:
        """
        Check if the search algorithm has finished its search (e.g., all combinations tried).

        Returns:
            True if the search is finished, False otherwise.
        """
        return self.current_idx >= len(self.param_combinations)
