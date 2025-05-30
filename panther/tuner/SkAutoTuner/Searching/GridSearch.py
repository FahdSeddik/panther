import pickle
from typing import Any, Dict, List

from .SearchAlgorithm import SearchAlgorithm

class GridSearch(SearchAlgorithm):
    """
    Grid search algorithm that systematically tries all combinations of parameters up to a maximum number of iterations.

    This implementation performs a search through a pre-generated grid of parameter
    combinations. It iterates through these combinations and keeps track of the best
    parameters found and their corresponding score.

    Attributes:
        max_iterations: The maximum number of parameter combinations to try.
        param_space: Dictionary mapping parameter names to lists of their possible values.
        indexed_param_space: A list of dictionaries, where each dictionary represents a unique
                             combination of parameter values.
        curr_iteration: The current iteration number, indicating which parameter combination
                        is being evaluated or will be evaluated next.
        best_score: The highest score achieved so far during the search.
        best_params: The dictionary of parameters that achieved the best_score.
    """

    def __init__(self, max_iterations: int = 10):
        """
        Initialize the GridSearch algorithm.

        Args:
            max_iterations: The maximum number of iterations to run.
        """
        self.max_iterations = max_iterations
        self.reset()
    
    def initialize(self, param_space: Dict[str, List]):
        """
        Initialize the search algorithm with the parameter space.
        
        Args:
            param_space: Dictionary of parameter names and their possible values
        """
        self.reset()
        self.param_space = param_space
        self.indexed_param_space = self._generate_indexed_param_space()
        self.max_iterations = min(self.max_iterations, len(self.indexed_param_space))

    def _my_product(self, value_lists: List[List[Any]]) -> List[tuple]:
        """
        Computes the Cartesian product of a list of lists.
        Example: _my_product([[1, 2], ['a', 'b']]) -> [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
        """
        if not value_lists:
            return [()]  

        product_tuples = [()]

        for current_list_values in value_lists:
            if not current_list_values:
                return []
            
            product_tuples = [existing_tuple + (item,)
                              for existing_tuple in product_tuples
                              for item in current_list_values]
        
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
    
    def get_next_params(self) -> Dict[str, Any]:
        """
        Get the next set of parameters to try.
        
        Returns:
            Dictionary of parameter names and values to try
        """
        if self.is_finished():
            return None
        
        params = self.indexed_param_space[self.curr_iteration]
        self.curr_iteration = self.curr_iteration + 1
        return params
    
    def update(self, params: Dict[str, Any], score: float):
        """
        Update the search algorithm with the results of the latest trial.
        
        Args:
            params: Dictionary of parameter names and values that were tried
            score: The evaluation score for the parameters
        """
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
            "param_space" : self.param_space,
            "curr_iteration" : self.curr_iteration,
            "indexed_param_space" : self.indexed_param_space,
            "best_params" : self.best_params,
            "best_score" : self.best_score,
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
        self.best_params = state["best_params"]
        self.best_score = state["best_score"]

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best set of parameters found so far.

        Returns:
            Dictionary of the best parameter names and values.
        """
        return self.best_params

    def get_best_score(self) -> float:
        """
        Get the best score achieved so far.

        Returns:
            The best score.
        """
        return self.best_score

    def reset(self):
        """
        Reset the search algorithm to its initial state.
        """
        self.param_space = None
        self.curr_iteration = 0
        self.indexed_param_space = []
        self.best_params = None
        self.best_score = -float("inf")

    def is_finished(self) -> bool:
        """
        Check if the search algorithm has finished its search (e.g., budget exhausted).

        Returns:
            True if the search is finished, False otherwise.
        """
        return self.curr_iteration == self.max_iterations