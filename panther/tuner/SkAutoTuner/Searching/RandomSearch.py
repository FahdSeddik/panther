import pickle  # Used for state serialization
from typing import Any, Dict, List

import numpy as np

from .SearchAlgorithm import SearchAlgorithm


class RandomSearch(SearchAlgorithm):
    """
    Random search algorithm that randomly samples from the parameter space.

    This class implements a simple random search algorithm for hyperparameter optimization.
    It randomly selects parameter combinations from the defined parameter space and
    keeps track of the best performing configuration.

    Attributes:
        param_space (Dict[str, List]): Dictionary mapping parameter names to possible values
        max_trials (int): Maximum number of trials to perform
        current_trial (int): Current trial number
        param_combinations (List[Dict[str, Any]]): List of all possible parameter combinations
        history (List[Dict[str, Any]]): History of tried parameters and their scores
        best_score (float): Best score found so far
        best_params (Dict[str, Any]): Parameters that achieved the best score
    """

    def __init__(self, max_trials: int = 20):
        """
        Initialize the RandomSearch algorithm.

        Args:
            max_trials (int): Maximum number of trials to perform. Defaults to 20.
        """
        self.param_space: Dict[str, List] = {}
        self.max_trials = max_trials
        self.current_trial = 0
        self.param_combinations: List[Dict[str, Any]] = []
        self.history: List[
            Dict[str, Any]
        ] = []  # Stores {'params': params, 'score': score}
        self.best_score: float = -float("inf")  # Assuming higher score is better
        self.best_params: Dict[str, Any] | None = None

    def initialize(self, param_space: Dict[str, List]):
        """
        Initialize the search algorithm with a parameter space.

        Args:
            param_space (Dict[str, List]): Dictionary mapping parameter names to possible values
        """
        self.param_space = param_space
        self.current_trial = 0
        self.param_combinations = []
        self.history: List[Dict[str, Any]] = []
        self.best_score: float = -float("inf")  # Assuming higher score is better
        self.best_params: Dict[str, Any] | None = None
        self._generate_combinations()

    def _generate_combinations(self):
        """
        Generate all possible combinations of parameters from the parameter space.

        This method uses itertools.product to create the Cartesian product of all
        parameter values and stores them as dictionaries in param_combinations.
        """
        from itertools import product

        keys = list(self.param_space.keys())
        values = list(self.param_space.values())

        # Create Cartesian product of all parameter values
        for combination in product(*values):
            self.param_combinations.append(dict(zip(keys, combination)))

    def get_next_params(self) -> Dict[str, Any] | None:
        """
        Get the next set of parameters to try.

        Randomly selects a parameter combination from the remaining untried combinations.

        Returns:
            Dict[str, Any] | None: Dictionary of parameter names and values, or None if
                                 all trials have been completed or no combinations remain.
        """
        if self.current_trial >= self.max_trials or len(self.param_combinations) == 0:
            return None  # All trials completed or no combinations left

        self.current_trial += 1
        # Randomly select a parameter combination
        choice = np.random.randint(0, len(self.param_combinations))
        # Remove and return the chosen combination
        selected_params = self.param_combinations.pop(choice)
        return selected_params

    def update(self, params: Dict[str, Any], score: float):
        """
        Update the search algorithm with the results of the latest trial.

        Records the trial in the history and updates the best parameters if
        the current score is better than the previous best.

        Args:
            params (Dict[str, Any]): Dictionary of parameter names and values that were tried
            score (float): The evaluation score for the parameters
        """
        self.history.append({"params": params.copy(), "score": score})
        # Update best parameters if current score is better
        # Assuming higher score is better. If lower is better, change to score < self.best_score
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()

    def save_state(self, filepath: str):
        """
        Save the current state of the search algorithm to a file.

        Serializes all instance variables to preserve the current state for later resumption.

        Args:
            filepath (str): The path to the file where the state should be saved.
        """
        state = {
            "param_space": self.param_space,
            "max_trials": self.max_trials,
            "current_trial": self.current_trial,
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

        Restores all instance variables from a previously saved state.

        Args:
            filepath (str): The path to the file from which the state should be loaded.
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        # Restore all instance variables
        self.param_space = state["param_space"]
        self.max_trials = state["max_trials"]
        self.current_trial = state["current_trial"]
        self.param_combinations = state["param_combinations"]
        self.history = state["history"]
        self.best_score = state["best_score"]
        self.best_params = state["best_params"]

    def get_best_params(self) -> Dict[str, Any] | None:
        """
        Get the best set of parameters found so far.

        Returns:
            Dict[str, Any] | None: Dictionary of the best parameter names and values,
                                 or None if no parameters have been evaluated.
        """
        return self.best_params

    def get_best_score(self) -> float:
        """
        Get the best score achieved so far.

        Returns:
            float: The best score. Returns -float('inf') if no trials have been updated or if reset.
        """
        return self.best_score

    def reset(self):
        """
        Reset the search algorithm to its initial state.

        Preserves param_space and max_trials, but resets all other instance variables
        to their initial values and regenerates parameter combinations.
        """
        self.current_trial = 0
        self.param_combinations = []
        self.history = []
        self.best_score = -float("inf")  # Assuming higher score is better
        self.best_params = None
        if self.param_space:  # Regenerate combinations if param_space is defined
            self._generate_combinations()

    def is_finished(self) -> bool:
        """
        Check if the search algorithm has finished its search.

        Returns:
            bool: True if the search is finished (max_trials reached), False otherwise.
        """
        return self.current_trial >= self.max_trials
