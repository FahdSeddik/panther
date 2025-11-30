import pickle
from typing import Any, Dict, List, Optional

import numpy as np

from .SearchAlgorithm import SearchAlgorithm


class RandomSearch(SearchAlgorithm):
    """
    Random search algorithm that randomly samples parameter combinations.

    This implementation first generates all possible parameter combinations from the
    given parameter space. It then randomly samples a specified number of these
    combinations (max_trials) without replacement. It keeps track of the best
    parameters found and their corresponding score throughout the search process.

    Attributes:
        max_trials: The maximum number of random parameter combinations to try.
        param_space: Dictionary mapping parameter names to lists of their possible values.
        indexed_param_space: A list of dictionaries, where each dictionary represents a unique
                             combination of parameter values. During the search, tried combinations
                             are removed from this list.
        curr_iteration: The current number of trials performed.
        best_score: The highest score achieved so far during the search.
        best_params: The dictionary of parameters that achieved the best_score.
    """

    def __init__(self, max_trials: int = 20):
        self._original_max_trials = max_trials  # Store original value for reset
        self.max_trials = max_trials
        self.param_space: Dict[str, List] = {}
        self.curr_iteration: int = 0
        self.indexed_param_space: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = -float("inf")
        self.reset()

    @property
    def current_trial(self):
        """Alias for curr_iteration for backward compatibility."""
        return self.curr_iteration

    @current_trial.setter
    def current_trial(self, value):
        self.curr_iteration = value

    @property
    def param_combinations(self):
        """Alias for indexed_param_space for backward compatibility."""
        return self.indexed_param_space

    @param_combinations.setter
    def param_combinations(self, value):
        self.indexed_param_space = value

    def initialize(self, param_space: Dict[str, List]):
        """
        Initialize the search algorithm with the parameter space.

        Args:
            param_space: Dictionary of parameter names and their possible values
        """
        self.reset()
        self.param_space = param_space
        self.indexed_param_space = self._generate_indexed_param_space()
        self.max_trials = min(self.max_trials, len(self.indexed_param_space))

    def _my_product(self, value_lists: List[List[Any]]) -> List[tuple[Any, ...]]:
        """
        Computes the Cartesian product of a list of lists.
        Example: _my_product([[1, 2], ['a', 'b']]) -> [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
        """
        if not value_lists:
            return [()]  # Product of no lists is a list with one empty tuple

        # Initialize with a list containing an empty tuple.
        # Each item from the subsequent lists will be appended to these tuples.
        product_tuples: List[tuple[Any, ...]] = [()]

        for current_list_values in value_lists:
            if not current_list_values:
                # If any input list is empty, the Cartesian product is empty.
                return []

            # Build the new product by combining existing tuples with items from the current list
            product_tuples = [
                existing_tuple + (item,)
                for existing_tuple in product_tuples
                for item in current_list_values
            ]

        return product_tuples

    def _generate_indexed_param_space(self):
        """
        Generates an indexed parameter space from the provided param_space.
        The indexed parameter space is a list of dictionaries, where each dictionary
        represents a unique combination of parameter values.

        Raises:
            ValueError: If param_space is None.

        Returns:
            List[Dict[str, Any]]: The indexed parameter space.
        """
        if self.param_space is None:
            raise ValueError("param_space is None")

        param_names = list(self.param_space.keys())
        value_lists = list(self.param_space.values())

        values_combined = self._my_product(value_lists)
        indexed_param_space = [{} for _ in range(len(values_combined))]

        i = 0
        for value_combination in values_combined:
            param = {}
            j = 0
            for value in value_combination:
                param[param_names[j]] = value
                j = j + 1
            indexed_param_space[i] = param
            i = i + 1

        return indexed_param_space

    def get_next_params(self) -> Optional[Dict[str, Any]]:
        """
        Get the next set of parameters to try.

        Returns:
            Dictionary of parameter names and values to try, or None if finished
        """
        if self.is_finished():
            return None

        choice = np.random.randint(
            low=0, high=len(self.indexed_param_space), size=1, dtype=np.int32
        )[0]

        params = self.indexed_param_space[choice]
        self.indexed_param_space.pop(choice)

        self.curr_iteration = self.curr_iteration + 1
        return params

    def update(self, params: Dict[str, Any], score: float):
        """
        Update the search algorithm with the results of the latest trial.

        Args:
            params: Dictionary of parameter names and values that were tried
            score: The evaluation score for the parameters
        """
        self.history.append({"params": params, "score": score})
        if self.best_score < score:
            self.best_score = score
            self.best_params = params

    def save_state(self, filepath: str):
        """
        Save the current state of the search algorithm to a file.

        Args:
            filepath: The path to the file where the state should be saved.
        """
        state = {
            "param_space": self.param_space,
            "curr_iteration": self.curr_iteration,
            "indexed_param_space": self.indexed_param_space,
            "history": self.history,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "_original_max_trials": self._original_max_trials,
            "max_trials": self.max_trials,
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """
        Load the state of the search algorithm from a file.

        Args:
            filepath: The path to the file from which the state should be loaded.
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.param_space = state["param_space"]
        self.curr_iteration = state["curr_iteration"]
        self.indexed_param_space = state["indexed_param_space"]
        self.history = state.get("history", [])
        self.best_params = state["best_params"]
        self.best_score = state["best_score"]
        self._original_max_trials = state.get(
            "_original_max_trials", state.get("max_trials", self.max_trials)
        )
        self.max_trials = state.get("max_trials", self.max_trials)

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """
        Get the best set of parameters found so far.

        Returns:
            Dictionary of the best parameter names and values, or None if no params yet.
        """
        return self.best_params

    def get_best_score(self) -> Optional[float]:
        """
        Get the best score achieved so far.

        Returns:
            The best score, or None if no score yet.
        """
        return self.best_score

    def reset(self):
        """
        Reset the search algorithm to its initial state while preserving param_space.
        If param_space is set, it regenerates the combinations.
        """
        # Preserve param_space if it exists
        preserved_param_space = getattr(self, "param_space", {})

        # Reset all state - always set param_space to {} (not None) after first init
        if hasattr(self, "_original_max_trials"):
            # After __init__, always use {}
            self.param_space = {}
        else:
            # During __init__, can be None temporarily
            self.param_space = None

        self.curr_iteration = 0
        self.indexed_param_space = []
        self.history = []
        self.best_params = None
        self.best_score = -float("inf")

        # Restore original max_trials
        if hasattr(self, "_original_max_trials"):
            self.max_trials = self._original_max_trials

        # Restore and regenerate if param_space was set
        if preserved_param_space:
            self.param_space = preserved_param_space
            self.indexed_param_space = self._generate_indexed_param_space()

    def is_finished(self) -> bool:
        """
        Check if the search algorithm has finished its search (e.g., budget exhausted).

        Returns:
            True if the search is finished, False otherwise.
        """
        return self.curr_iteration == self.max_trials
