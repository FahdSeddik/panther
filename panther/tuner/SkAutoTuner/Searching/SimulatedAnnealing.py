import math
import pickle
import random
from typing import Any, Dict, List

from .SearchAlgorithm import SearchAlgorithm


class SimulatedAnnealing(SearchAlgorithm):
    def __init__(
        self,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
        max_iterations: int = 1000,
    ):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.temperature = self.initial_temperature
        self.current_solution: Dict[str, Any] = {}
        self.current_score = float("-inf")
        self.best_solution: Dict[str, Any] = {}
        self.best_score = float("-inf")
        self.iterations = 0
        self.param_space: Dict[str, List[Any]] = {}

    def initialize(self, param_space: Dict[str, List[Any]]):
        self.param_space = param_space
        self.current_solution = self._get_random_solution()
        # Initial score will be evaluated by the tuner, so no need to set it here
        self.best_solution = self.current_solution
        self.temperature = self.initial_temperature
        self.iterations = 0
        # Assume initial solution needs to be evaluated first
        self.current_score = float("-inf")  # Mark as unevaluated
        self.best_score = float("-inf")  # Mark as unevaluated

    def _get_random_solution(self) -> Dict[str, Any]:
        if not self.param_space:
            raise ValueError(
                "Parameter space not initialized or empty. Call initialize() first."
            )
        return {
            param: random.choice(values) for param, values in self.param_space.items()
        }

    def _get_neighbor(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        if not self.param_space:
            raise ValueError("Parameter space not initialized or empty.")

        neighbor = solution.copy()
        # Ensure there are parameters to change
        if not self.param_space.keys():
            return neighbor  # No parameters to change

        param_to_change = random.choice(list(self.param_space.keys()))

        current_value = neighbor[param_to_change]
        possible_values = self.param_space[param_to_change]

        if len(possible_values) == 1:  # Only one possible value
            return neighbor  # No change possible

        new_value = random.choice(possible_values)
        while new_value == current_value:  # Ensure the new value is different
            if (
                len(possible_values) <= 1
            ):  # Should not happen if we check above, but for safety
                break
            new_value = random.choice(possible_values)

        neighbor[param_to_change] = new_value
        return neighbor

    def get_next_params(self) -> Dict[str, Any]:
        if self.is_finished():
            return self.get_best_params()

        if self.iterations == 0:
            # If current_solution is empty (e.g., just after __init__ but before/without initialize),
            # or if it's the first iteration after initialize(), current_solution should be returned.
            # initialize() sets current_solution.
            if (
                not self.current_solution and self.param_space
            ):  # Not initialized but space is available
                self.current_solution = self._get_random_solution()  # Initialize it
            # If current_solution is still empty (e.g. param_space was also empty),
            # or it's the intended first solution, return it.
            # The tuner expects to evaluate this first.
            return self.current_solution

        # Fallback if current_solution is somehow empty after the first iteration check, get a random one.
        # This ideally shouldn't be hit if initialize() was called and param_space is valid.
        if not self.current_solution:
            if not self.param_space:
                raise ValueError(
                    "Cannot get next params: param_space is empty and current_solution is not set."
                )
            return self._get_random_solution()

        # For subsequent calls, generate a neighbor.
        # The previous logic with current_score == float("-inf") is implicitly handled
        # by the iteration count and the update logic.

        candidate_solution = self._get_neighbor(self.current_solution)
        return candidate_solution

    def update(self, params: Dict[str, Any], score: float):
        self.iterations += 1

        if self.current_score == float("-inf"):  # First evaluation (initial solution)
            self.current_score = score
            self.current_solution = params
            if score > self.best_score:
                self.best_score = score
                self.best_solution = params
            return  # No annealing logic for the very first evaluation

        # Now, params is the candidate_solution evaluated, and score is its score.
        # self.current_solution is the *previous* solution, and self.current_score is its score.

        delta_score = score - self.current_score

        if delta_score > 0:  # New solution is better
            self.current_solution = params
            self.current_score = score
            if score > self.best_score:
                self.best_score = score
                self.best_solution = params
        else:
            # New solution is worse, accept with a probability
            acceptance_probability = math.exp(delta_score / self.temperature)
            if random.random() < acceptance_probability:
                self.current_solution = params
                self.current_score = score

        self.temperature *= self.cooling_rate
        if self.temperature < self.min_temperature:
            self.temperature = self.min_temperature  # Clip temperature to minimum

    def save_state(self, filepath: str):
        state = {
            "initial_temperature": self.initial_temperature,
            "cooling_rate": self.cooling_rate,
            "min_temperature": self.min_temperature,
            "max_iterations": self.max_iterations,
            "temperature": self.temperature,
            "current_solution": self.current_solution,
            "current_score": self.current_score,
            "best_solution": self.best_solution,
            "best_score": self.best_score,
            "iterations": self.iterations,
            "param_space": self.param_space,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.initial_temperature = state["initial_temperature"]
        self.cooling_rate = state["cooling_rate"]
        self.min_temperature = state["min_temperature"]
        self.max_iterations = state["max_iterations"]
        self.temperature = state["temperature"]
        self.current_solution = state["current_solution"]
        self.current_score = state["current_score"]
        self.best_solution = state["best_solution"]
        self.best_score = state["best_score"]
        self.iterations = state["iterations"]
        self.param_space = state["param_space"]

    def get_best_params(self) -> Dict[str, Any]:
        if not self.best_solution and self.param_space:
            # If no evaluations led to a best_solution, but we have a param space,
            # return a random one as a fallback. initialize() should have set best_solution.
            return self._get_random_solution()
        return self.best_solution  # Already {} if not set, or the actual best solution

    def get_best_score(self) -> float:
        return self.best_score

    def reset(self):
        self.temperature = self.initial_temperature
        self.current_solution = {}
        self.current_score = float("-inf")
        self.best_solution = {}
        self.best_score = float("-inf")
        self.iterations = 0
        self.param_space = {}  # Reset param_space, expect initialize() to be called again.

    def is_finished(self) -> bool:
        return (
            self.iterations >= self.max_iterations
            or self.temperature <= self.min_temperature
        )
