import math
import pickle
import random
from typing import Any, Dict, List, Optional

from .SearchAlgorithm import SearchAlgorithm


class TreeParzenEstimator(SearchAlgorithm):
    """
    Tree-structured Parzen Estimator (TPE) algorithm.

    TPE is a sequential model-based optimization (SMBO) algorithm that models
    P(score|parameters) and P(parameters) by maintaining two distributions:
    l(x) for parameters 'x' associated with good scores, and g(x) for
    parameters associated with bad scores. It aims to find parameters that
    maximize l(x) / g(x).
    """

    def __init__(
        self,
        n_initial_points: int = 10,
        max_trials: int = 100,
        gamma_ratio: float = 0.25,
        n_ei_candidates: int = 24,
        verbose: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize the TPE algorithm.

        Args:
            n_initial_points: Number of initial random trials before TPE modeling starts.
            max_trials: Total budget of trials to run.
            gamma_ratio: Quantile to split trials into "good" and "bad".
                         For example, 0.25 means top 25% are "good".
            n_ei_candidates: Number of candidate configurations to sample and evaluate
                             in each TPE step.
            verbose: Whether to print debug information.
            seed: Optional random seed for reproducibility.
        """
        super().__init__()
        if not (0 < gamma_ratio < 1):
            raise ValueError("gamma_ratio must be between 0 and 1 exclusive.")

        self.n_initial_points = n_initial_points
        self.max_trials = max_trials
        self.gamma_ratio = gamma_ratio
        self.n_ei_candidates = n_ei_candidates
        self.verbose = verbose

        if seed is not None:
            random.seed(seed)

        self.param_space: Dict[str, List[Any]] = {}
        self.history: List[Dict[str, Any]] = []
        self.scores: List[float] = []

        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = -float("inf")  # Assuming higher score is better
        self.trial_count: int = 0

        # Ensure n_initial_points is at least 2 to allow for good/bad split later if gamma is high
        if self.n_initial_points < 2:
            # A single point cannot be split into good/bad for TPE logic.
            # Need at least 2 points if gamma_ratio implies one good one bad.
            if self.verbose:
                print(
                    f"Warning: n_initial_points ({self.n_initial_points}) is low. Setting to 2 for TPE."
                )
            self.n_initial_points = max(2, self.n_initial_points)

    def _generate_random_params(self) -> Dict[str, Any]:
        """Generates a random set of parameters from the param_space."""
        if not self.param_space:
            raise ValueError("Parameter space not initialized.")
        params = {}
        for p_name, p_values in self.param_space.items():
            if not p_values:  # Should not happen with valid param_space
                raise ValueError(f"Parameter {p_name} has no values in param_space.")
            params[p_name] = random.choice(p_values)
        return params

    def initialize(self, param_space: Dict[str, List[Any]]):
        """
        Initialize the search algorithm with the parameter space.

        Args:
            param_space: Dictionary of parameter names and their possible discrete values.
        """
        if not param_space:
            raise ValueError("Parameter space cannot be empty.")
        for p_name, p_values in param_space.items():
            if not p_values:
                raise ValueError(
                    f"Parameter '{p_name}' in param_space must have at least one value."
                )

        self.param_space = param_space
        self.reset()

    def reset(self):
        """
        Reset the search algorithm to its initial state.
        """
        self.history = []
        self.scores = []
        self.trial_count = 0
        self.best_score = -float("inf")  # Assuming higher score is better

        if self.param_space:
            # Initialize best_params to a random configuration.
            # This will be updated by the first real trial result.
            self.best_params = self._generate_random_params()
        else:
            # If param_space is not set yet (e.g. during __init__ before initialize call)
            self.best_params = None

    def _sample_param_value(
        self, observed_param_values: List[Any], all_possible_values: List[Any]
    ) -> Any:
        """
        Sample a parameter value.
        If observed_param_values is empty, sample uniformly from all_possible_values.
        Otherwise, sample uniformly from observed_param_values (empirical distribution).
        """
        if not observed_param_values:
            return random.choice(all_possible_values)
        return random.choice(observed_param_values)

    def _calculate_param_prob(
        self,
        value_to_eval: Any,
        observed_param_values: List[Any],
        all_possible_values: List[Any],
    ) -> float:
        """
        Calculate the probability of 'value_to_eval' given 'observed_param_values'
        using Laplace smoothing.
        Assumes discrete parameter values.
        """
        smoothing_alpha = 1.0  # Laplace smoothing (add-1)
        num_categories = len(all_possible_values)

        count_value = observed_param_values.count(value_to_eval)
        total_observed = len(observed_param_values)

        prob = (count_value + smoothing_alpha) / (
            total_observed + smoothing_alpha * num_categories
        )
        return prob

    def get_next_params(self) -> Dict[str, Any]:
        """
        Get the next set of parameters to try.
        Uses random search for initial points, then TPE.
        """
        if not self.param_space:
            raise RuntimeError(
                "Optimizer not initialized. Call initialize(param_space) first."
            )

        if self.trial_count < self.n_initial_points:
            if self.verbose:
                print(
                    f"Trial {self.trial_count + 1}/{self.max_trials}: Random search phase"
                )
            return self._generate_random_params()

        if self.verbose:
            print(f"Trial {self.trial_count + 1}/{self.max_trials}: TPE search phase")

        # TPE Phase
        num_total_trials = len(self.scores)
        if num_total_trials < 2:  # Need at least 2 trials to split into good/bad
            if self.verbose:
                print("Not enough trials for TPE, falling back to random.")
            return self._generate_random_params()

        sorted_indices = sorted(
            range(num_total_trials), key=lambda k: self.scores[k], reverse=True
        )

        # Determine split point for good/bad trials
        # n_good aims for gamma_ratio, but must be at least 1 and leave at least 1 for bad.
        n_good_ideal = math.ceil(self.gamma_ratio * num_total_trials)
        n_good = max(1, int(min(n_good_ideal, num_total_trials - 1)))

        good_indices = sorted_indices[:n_good]
        bad_indices = sorted_indices[n_good:]

        if not good_indices or not bad_indices:
            # This should ideally not be reached if num_total_trials >= 2
            # and n_initial_points ensures enough diversity.
            if self.verbose:
                print("Could not form valid good/bad sets, falling back to random.")
            return self._generate_random_params()

        good_trials_params = [self.history[i] for i in good_indices]
        bad_trials_params = [self.history[i] for i in bad_indices]

        best_candidate_params: Optional[Dict[str, Any]] = None
        max_ei_surrogate_score = -float("inf")

        for _ in range(self.n_ei_candidates):
            candidate_params: Dict[str, Any] = {}
            # Sample candidate parameters based on the 'good' distribution (l(x))
            # For each param, sample from its empirical distribution in good_trials_params
            for param_name, all_values in self.param_space.items():
                good_param_observations = [
                    trial[param_name] for trial in good_trials_params
                ]
                candidate_params[param_name] = self._sample_param_value(
                    good_param_observations, all_values
                )

            # Calculate log P(x|good) - log P(x|bad)
            log_prob_good = 0.0
            log_prob_bad = 0.0

            for param_name, value in candidate_params.items():
                all_possible_values_for_param = self.param_space[param_name]

                good_param_observations = [
                    trial[param_name] for trial in good_trials_params
                ]
                prob_l = self._calculate_param_prob(
                    value, good_param_observations, all_possible_values_for_param
                )
                log_prob_good += math.log(
                    prob_l + 1e-12
                )  # Add epsilon for numerical stability

                bad_param_observations = [
                    trial[param_name] for trial in bad_trials_params
                ]
                prob_g = self._calculate_param_prob(
                    value, bad_param_observations, all_possible_values_for_param
                )
                log_prob_bad += math.log(prob_g + 1e-12)  # Add epsilon

            current_ei_surrogate_score = log_prob_good - log_prob_bad

            if current_ei_surrogate_score > max_ei_surrogate_score:
                max_ei_surrogate_score = current_ei_surrogate_score
                best_candidate_params = candidate_params

        if best_candidate_params is not None:
            return best_candidate_params
        else:
            # Fallback if no candidate was better (e.g., all had -inf score due to log(0))
            # or if n_ei_candidates is 0 (though __init__ should prevent this)
            if self.verbose:
                print("No suitable TPE candidate found, falling back to random.")
            return self._generate_random_params()

    def update(self, params: Dict[str, Any], score: float):
        """
        Update the search algorithm with the results of the latest trial.

        Args:
            params: Dictionary of parameter names and values that were tried.
            score: The evaluation score for the parameters (higher is better).
        """
        self.history.append(params)
        self.scores.append(score)
        self.trial_count += 1

        if score > self.best_score:
            self.best_score = score
            self.best_params = params
        elif self.best_params is None:  # Handles the very first update
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
            "history": self.history,
            "scores": self.scores,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "trial_count": self.trial_count,
            "n_initial_points": self.n_initial_points,
            "max_trials": self.max_trials,
            "gamma_ratio": self.gamma_ratio,
            "n_ei_candidates": self.n_ei_candidates,
            "verbose": self.verbose,
            "random_state": random.getstate(),  # Save Python's random generator state
        }
        try:
            with open(filepath, "wb") as f:
                pickle.dump(state, f)
            if self.verbose:
                print(f"TPE state saved to {filepath}")
        except IOError as e:
            print(f"Error saving TPE state to {filepath}: {e}")

    def load_state(self, filepath: str):
        """
        Load the state of the search algorithm from a file.

        Args:
            filepath: The path to the file from which the state should be loaded.
        """
        try:
            with open(filepath, "rb") as f:
                state = pickle.load(f)

            self.param_space = state["param_space"]
            self.history = state["history"]
            self.scores = state["scores"]
            self.best_params = state["best_params"]
            self.best_score = state["best_score"]
            self.trial_count = state["trial_count"]

            # Load TPE-specific hyperparams if they exist in the state,
            # otherwise keep existing ones from __init__
            self.n_initial_points = state.get("n_initial_points", self.n_initial_points)
            self.max_trials = state.get("max_trials", self.max_trials)
            self.gamma_ratio = state.get("gamma_ratio", self.gamma_ratio)
            self.n_ei_candidates = state.get("n_ei_candidates", self.n_ei_candidates)
            self.verbose = state.get("verbose", self.verbose)

            if "random_state" in state:
                random.setstate(state["random_state"])

            if self.verbose:
                print(f"TPE state loaded from {filepath}")

        except FileNotFoundError:
            print(f"Error: State file {filepath} not found.")
        except (IOError, pickle.PickleError, KeyError) as e:
            print(f"Error loading TPE state from {filepath}: {e}")

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best set of parameters found so far.

        Returns:
            Dictionary of the best parameter names and values.
            Returns a random configuration if no trials have been run or no
            parameters have been found to be better than the initial random set.
        """
        if self.best_params is None:
            if not self.param_space:
                raise RuntimeError(
                    "Cannot get best parameters: optimizer not initialized or param_space is empty."
                )
            if self.verbose:
                print(
                    "Warning: get_best_params() called before any updates or with no improvement; returning random params."
                )
            return (
                self._generate_random_params()
            )  # Fallback if no updates made best_params valid
        return self.best_params

    def get_best_score(self) -> float:
        """
        Get the best score achieved so far.

        Returns:
            The best score. Returns -float('inf') if no trials run.
        """
        return self.best_score

    def is_finished(self) -> bool:
        """
        Check if the search algorithm has finished its search (budget exhausted).

        Returns:
            True if the search is finished, False otherwise.
        """
        return self.trial_count >= self.max_trials
