from typing import Any, Dict, List, Optional

import numpy as np
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    LogExpectedImprovement,
    PosteriorMean,
    PosteriorStandardDeviation,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qLowerBoundMaxValueEntropy,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from .SearchAlgorithm import SearchAlgorithm


class BayesianOptimization(SearchAlgorithm):
    """
    Bayesian optimization search algorithm using botorch and GPyTorch.
    This implementation leverages the efficient and robust implementations
    from the botorch library for Bayesian Optimization.
    """

    def __init__(
        self,
        max_trials: int = 20,
        random_trials: int = 3,
        exploration_weight: float = 0.1,
        acquisition_type: str = "logei",
        seed: Optional[int] = None,
    ):
        """
        Initialize Bayesian Optimization algorithm.

        Args:
            max_trials: Maximum number of trials to run
            random_trials: Number of initial random trials before using GP
            exploration_weight: Weight for exploration in acquisition function (higher = more exploration)
            acquisition_type: Type of acquisition function ('ei', 'ucb', or 'logei')
            noise_level: Assumed noise level in observations
            seed: Random seed for reproducibility
        """
        self.param_space = {}
        self.max_trials = max_trials
        self.random_trials = random_trials
        self.exploration_weight = exploration_weight
        self.acquisition_type = acquisition_type.lower()
        self.current_trial = 0

        # Parameter mapping
        self._param_mapping = {}  # Maps parameter names to indices
        self._param_inv_mapping = {}  # Maps indices to parameter names
        # Observation history
        self.train_x = None  # Normalized tensor of parameter values
        self.train_y = None  # Tensor of observed scores
        self.train_y_raw = []  # Raw observation values for reporting

        # Best observed value and parameters
        self.best_value = None
        self.best_params = None

        # GP model
        self.model = None
        self.bounds = None

    def initialize(self, param_space: Dict[str, List]):
        """
        Initialize the search algorithm with the parameter space.

        Args:
            param_space: Dictionary of parameter names and their possible values
        """
        # Reset state
        self.param_space = param_space
        self.current_trial = 0
        self._param_mapping = {}
        self._param_inv_mapping = {}

        # Determine parameter types and create mappings
        for i, (param, values) in enumerate(self.param_space.items()):
            self._param_mapping[param] = i
            self._param_inv_mapping[i] = param

        # Initialize tensors for observations
        self.train_x = torch.zeros((0, len(self._param_mapping)), dtype=torch.float64)
        self.train_y = torch.zeros((0, 1), dtype=torch.float64)
        self.train_y_raw = []

        # Setup optimization bounds (normalized to [0, 1])
        self.bounds = torch.stack(
            [
                torch.zeros(len(self._param_mapping), dtype=torch.float64),
                torch.ones(len(self._param_mapping), dtype=torch.float64),
            ]
        )

        # Reset best observed value
        self.best_value = None
        self.best_params = None

        # Reset model
        self.model = None

    def _params_to_point(self, params: Dict[str, Any]) -> torch.Tensor:
        """
        Convert a parameter dictionary to a normalized point in the search space.

        Args:
            params: Dictionary of parameter values

        Returns:
            Tensor of normalized parameter values
        """
        point = torch.zeros(len(self._param_mapping), dtype=torch.float64)
        for param, value in params.items():
            idx = self._param_mapping[param]
            options = self.param_space[param]

            try:
                value_idx = options.index(value)
            except ValueError:
                raise ValueError(
                    f"Value '{value}' for parameter '{param}' not found in options {options}"
                )

            point[idx] = value_idx / (len(options) - 1) if len(options) > 1 else 0.5
        return point.unsqueeze(0)  # Add batch dimension

    def _point_to_params(self, point: torch.Tensor) -> Dict[str, Any]:
        """
        Convert a normalized point in the search space to a parameter dictionary.

        Args:
            point: Tensor of normalized parameter values

        Returns:
            Dictionary of parameter values
        """
        # Remove batch dimension if present
        if point.ndim > 1:
            point = point.squeeze(0)

        point = point.detach().numpy()
        params = {}
        for i, norm_value in enumerate(point):
            param = self._param_inv_mapping[i]
            options = self.param_space[param]

            # For all parameters, select the closest discrete option
            if len(options) > 1:
                raw_idx = norm_value * (len(options) - 1)
                idx = min(int(round(raw_idx)), len(options) - 1)
            else:
                idx = 0

            params[param] = options[idx]

        return params

    def _create_gp_model(self):
        """
        Create a Gaussian Process model with appropriate priors and constraints.
        """
        try:
            # Create the GP model with custom priors and constraints
            model = SingleTaskGP(
                self.train_x,
                self.train_y,
            )

            return model
        except Exception as e:
            raise RuntimeError(f"Failed to create GP model: {e}")

    def _update_model(self):
        """
        Update the GP model with the current observations.
        """
        # Create GP model
        self.model = self._create_gp_model()

        # Fit the model
        try:
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_mll(mll)
        except Exception as e:
            raise RuntimeError(f"Failed to fit GP model: {e}")

    def _create_acquisition_function(self):
        """
        Create an acquisition function based on the specified type.

        Returns:
            BoTorch acquisition function
        """
        if self.acquisition_type == "ucb":
            return UpperConfidenceBound(
                model=self.model,
                beta=self.exploration_weight,
            )
        elif self.acquisition_type == "logei":
            return LogExpectedImprovement(
                model=self.model, best_f=self.best_value, maximize=True
            )
        elif self.acquisition_type == "ei":
            return ExpectedImprovement(
                model=self.model, best_f=self.best_value, maximize=True
            )
        elif self.acquisition_type == "pi":
            return ProbabilityOfImprovement(
                model=self.model, best_f=self.best_value, maximize=True
            )
        elif self.acquisition_type == "mes":
            return qLowerBoundMaxValueEntropy(
                model=self.model, candidate_set=self.train_x
            )
        elif self.acquisition_type == "lcb":
            # Lower Confidence Bound - pessimistic/risk-averse acquisition
            return UpperConfidenceBound(
                model=self.model, beta=self.exploration_weight, maximize=False
            )
        elif self.acquisition_type == "pm":
            # Posterior Mean - pure exploitation
            return PosteriorMean(model=self.model, maximize=True)
        elif self.acquisition_type == "psd":
            # Posterior Standard Deviation - pure exploration
            return PosteriorStandardDeviation(model=self.model, maximize=True)
        else:
            raise ValueError(
                f"Unknown acquisition function type: {self.acquisition_type}"
            )

    def get_next_params(self) -> Dict[str, Any]:
        """
        Get the next set of parameters to try using Bayesian Optimization.

        Returns:
            Dictionary of parameter names and values to try
        """
        if self.current_trial >= self.max_trials:
            return None  # All trials completed

        self.current_trial += 1

        # Use random search for the first few trials
        if len(self.train_y) < self.random_trials:
            random_params = {}
            for param, values in self.param_space.items():
                # Always select directly from available options
                random_params[param] = np.random.choice(values)

            return random_params

        # Update the GP model
        self._update_model()

        # Create acquisition function
        acq_func = self._create_acquisition_function()

        # Optimize the acquisition function
        candidates, acq_values = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,  # Batch size
            num_restarts=5
            + len(self._param_mapping),  # More restarts for higher dimensions
            raw_samples=100
            * len(self._param_mapping),  # More samples for higher dimensions
        )

        # Convert the candidate to parameters
        next_params = self._point_to_params(candidates)

        return next_params

    def update(self, params: Dict[str, Any], score: float):
        """
        Update the search algorithm with the results of the latest trial.

        Args:
            params: Dictionary of parameter names and values that were tried
            score: The evaluation score for the parameters
        """
        # Convert params to normalized point
        point = self._params_to_point(params)

        # Store raw score
        self.train_y_raw.append(score)

        # Update the best observed value and parameters
        if self.best_value is None or score > self.best_value:
            self.best_value = torch.tensor(score, dtype=torch.float64)
            self.best_params = params.copy()

        # Add to observations
        score_tensor = torch.tensor([[score]], dtype=torch.float64)

        # Update training data
        self.train_x = torch.cat([self.train_x, point], dim=0)
        self.train_y = torch.cat([self.train_y, score_tensor], dim=0)

    def save_state(self, filepath: str):
        """
        Save the current state of the search algorithm to a file.

        Args:
            filepath: The path to the file where the state should be saved.
        """
        state = {
            "param_space": self.param_space,
            "max_trials": self.max_trials,
            "random_trials": self.random_trials,
            "exploration_weight": self.exploration_weight,
            "acquisition_type": self.acquisition_type,
            "current_trial": self.current_trial,
            "_param_mapping": self._param_mapping,
            "_param_inv_mapping": self._param_inv_mapping,
            "train_x": self.train_x,
            "train_y": self.train_y,
            "train_y_raw": self.train_y_raw,
            "best_value": self.best_value,
            "best_params": self.best_params,
            "bounds": self.bounds,
            # Note: self.model is not saved as it's recreated during _update_model
        }
        torch.save(state, filepath)

    def load_state(self, filepath: str):
        """
        Load the state of the search algorithm from a file.

        Args:
            filepath: The path to the file from which the state should be loaded.
        """
        state = torch.load(filepath)
        self.param_space = state["param_space"]
        self.max_trials = state["max_trials"]
        self.random_trials = state["random_trials"]
        self.exploration_weight = state["exploration_weight"]
        self.acquisition_type = state["acquisition_type"]
        self.current_trial = state["current_trial"]
        self._param_mapping = state["_param_mapping"]
        self._param_inv_mapping = state["_param_inv_mapping"]
        self.train_x = state["train_x"]
        self.train_y = state["train_y"]
        self.train_y_raw = state["train_y_raw"]
        self.best_value = state["best_value"]
        self.best_params = state["best_params"]
        self.bounds = state["bounds"]

        # Model needs to be rebuilt if there's training data
        if self.train_x is not None and len(self.train_x) > 0:
            self._update_model()

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
        if self.best_value is None:
            return None
        if isinstance(self.best_value, torch.Tensor):
            return self.best_value.item()
        return self.best_value

    def reset(self):
        """
        Reset the search algorithm to its initial state.
        This re-initializes the algorithm with the existing parameter space.
        If no parameter space was set, it resets to a clean state.
        """
        param_space_before_reset = self.param_space

        # Re-initialize attributes as in __init__
        self.max_trials = getattr(self, "max_trials", 20)  # Keep original if set
        self.random_trials = getattr(self, "random_trials", 3)  # Keep original if set
        self.exploration_weight = getattr(
            self, "exploration_weight", 0.1
        )  # Keep original if set
        self.acquisition_type = getattr(
            self, "acquisition_type", "logei"
        ).lower()  # Keep original if set

        self.current_trial = 0
        self._param_mapping = {}
        self._param_inv_mapping = {}
        self.train_x = None
        self.train_y = None
        self.train_y_raw = []
        self.best_value = None
        self.best_params = None
        self.model = None
        self.bounds = None

        # If param_space was defined, re-initialize with it
        if param_space_before_reset:
            self.initialize(param_space_before_reset)
        else:
            self.param_space = {}

    def is_finished(self) -> bool:
        """
        Check if the search algorithm has finished its search (e.g., budget exhausted).

        Returns:
            True if the search is finished, False otherwise.
        """
        return self.current_trial >= self.max_trials
