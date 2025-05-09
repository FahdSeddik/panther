from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from panther.utils.SkAutoTuner.Searching.SearchAlgorithm import SearchAlgorithm

try:
    import botorch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_model
    from botorch.acquisition import ExpectedImprovement
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.utils.transforms import normalize, standardize
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    print("Warning: botorch is not available. Install with: pip install botorch")

class BayesianOptimization(SearchAlgorithm):
    """
    Bayesian optimization search algorithm using botorch and GPyTorch.
    This implementation leverages the efficient and robust implementations
    from the botorch library for Bayesian Optimization.
    """
    def __init__(self, max_trials: int = 20, random_trials: int = 3, exploration_weight: float = 0.01):
        """
        Initialize Bayesian Optimization algorithm.
        
        Args:
            max_trials: Maximum number of trials to run
            random_trials: Number of initial random trials before using GP
            exploration_weight: Weight for exploration in acquisition function (higher = more exploration)
        """
        if not BOTORCH_AVAILABLE:
            raise ImportError("botorch is required for this implementation. Install with: pip install botorch")
        
        self.param_space = {}
        self.max_trials = max_trials
        self.random_trials = random_trials
        self.exploration_weight = exploration_weight
        self.current_trial = 0
        
        # Parameter mapping
        self._param_mapping = {}  # Maps parameter names to indices
        self._param_inv_mapping = {}  # Maps indices to parameter names
        
        # Observation history
        self.train_x = None  # Normalized tensor of parameter values
        self.train_y = None  # Tensor of observed scores
        
        # Best observed value so far
        self.best_value = None
        
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
        
        # Create mappings for parameters to make them numeric
        for i, (param, values) in enumerate(self.param_space.items()):
            self._param_mapping[param] = i
            self._param_inv_mapping[i] = param
        
        # Initialize tensors for observations
        self.train_x = torch.zeros((0, len(self._param_mapping)), dtype=torch.float64)
        self.train_y = torch.zeros((0, 1), dtype=torch.float64)
        
        # Setup optimization bounds (normalized to [0, 1])
        self.bounds = torch.stack([
            torch.zeros(len(self._param_mapping), dtype=torch.float64),
            torch.ones(len(self._param_mapping), dtype=torch.float64)
        ])
        
        # Reset best observed value
        self.best_value = None
        
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
            value_idx = options.index(value)
            # Normalize to [0, 1]
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
            
            # Denormalize and find closest option
            if len(options) > 1:
                raw_idx = norm_value * (len(options) - 1)
                idx = min(int(round(raw_idx)), len(options) - 1)
            else:
                idx = 0
                
            params[param] = options[idx]
        return params
    
    def _update_model(self):
        """
        Update the GP model with the current observations.
        """
        if len(self.train_y) < 2:
            return
        
        # Initialize a GP model
        self.model = SingleTaskGP(self.train_x, self.train_y)
        
        # Fit the model
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)
    
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
                random_params[param] = np.random.choice(values)
            return random_params
        
        # Update the GP model
        self._update_model()
        
        # Define the acquisition function
        acq_func = ExpectedImprovement(
            model=self.model, 
            best_f=self.best_value,
            maximize=True
        )
        
        # Optimize the acquisition function
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,  # Batch size
            num_restarts=10,  # Number of restarts for optimization
            raw_samples=100,  # Number of raw samples for initialization
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
        
        # Update the best observed value
        if self.best_value is None or score > self.best_value:
            self.best_value = torch.tensor(score, dtype=torch.float64)
        
        # Add to observations
        score_tensor = torch.tensor([[score]], dtype=torch.float64)
        
        # Update training data
        self.train_x = torch.cat([self.train_x, point], dim=0)
        self.train_y = torch.cat([self.train_y, score_tensor], dim=0)