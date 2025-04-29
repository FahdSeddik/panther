from typing import Any, Dict, List, Tuple

import numpy as np
from panther.utils.SkAutoTuner.Searching import SearchAlgorithm
from scipy.optimize import minimize
from scipy.stats import norm

class BayesianOptimization(SearchAlgorithm):
    """
    Bayesian optimization search algorithm using Gaussian Process regression.
    This is a production-ready implementation for hyperparameter optimization.
    """
    def __init__(self, max_trials: int = 20, random_trials: int = 3, exploration_weight: float = 0.01):
        """
        Initialize Bayesian Optimization algorithm.
        
        Args:
            max_trials: Maximum number of trials to run
            random_trials: Number of initial random trials before using GP
            exploration_weight: Weight for exploration in acquisition function (higher = more exploration)
        """
        self.param_space = {}
        self.max_trials = max_trials
        self.random_trials = random_trials
        self.exploration_weight = exploration_weight
        self.current_trial = 0
        self._param_mapping = {}  # Maps parameter names to indices
        self._param_inv_mapping = {}  # Maps indices to parameter names
        self.observed_X = []  # Normalized parameter values
        self.observed_y = []  # Observed scores
        self.normalized_y = []  # Normalized scores
        
        # GP kernel hyperparameters
        self.length_scale = 1.0
        self.signal_variance = 1.0
        self.noise_variance = 1e-6
    
    def initialize(self, param_space: Dict[str, List]):
        """
        Initialize the search algorithm with the parameter space.
        
        Args:
            param_space: Dictionary of parameter names and their possible values
        """
        # Reset state
        self.param_space = param_space
        self.current_trial = 0
        self._param_mapping = {}  # Maps parameter names to indices
        self._param_inv_mapping = {}  # Maps indices to parameter names
        self.observed_X = []  # Normalized parameter values
        self.observed_y = []  # Observed scores
        self.normalized_y = []  # Normalized scores
        
        # Reset GP kernel hyperparameters
        self.length_scale = 1.0
        self.signal_variance = 1.0
        self.noise_variance = 1e-6
        
        # Create mappings for parameters to make them numeric
        for i, (param, values) in enumerate(self.param_space.items()):
            self._param_mapping[param] = i
            self._param_inv_mapping[i] = param
    
    def _params_to_point(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Convert a parameter dictionary to a normalized point in the search space.
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            Numpy array of normalized parameter values
        """
        point = np.zeros(len(self._param_mapping))
        for param, value in params.items():
            idx = self._param_mapping[param]
            options = self.param_space[param]
            value_idx = options.index(value)
            # Normalize to [0, 1]
            point[idx] = value_idx / (len(options) - 1) if len(options) > 1 else 0.5
        return point
    
    def _point_to_params(self, point: np.ndarray) -> Dict[str, Any]:
        """
        Convert a normalized point in the search space to a parameter dictionary.
        
        Args:
            point: Numpy array of normalized parameter values
            
        Returns:
            Dictionary of parameter values
        """
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
    
    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Radial Basis Function (RBF) kernel.
        
        Args:
            x1: First point
            x2: Second point
            
        Returns:
            Kernel value
        """
        dist = np.sum(((x1 - x2) / self.length_scale) ** 2)
        return self.signal_variance * np.exp(-0.5 * dist)
    
    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """
        Compute the kernel matrix between two sets of points.
        
        Args:
            X1: First set of points (n_points1, n_dims)
            X2: Second set of points (n_points2, n_dims), or None to use X1
            
        Returns:
            Kernel matrix (n_points1, n_points2)
        """
        if X2 is None:
            X2 = X1
            
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._rbf_kernel(X1[i], X2[j])
        
        return K
    
    def _gp_predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using Gaussian Process regression.
        
        Args:
            X_test: Test points (n_points, n_dims)
            
        Returns:
            Tuple of (mean, variance) for predictions
        """
        if not self.observed_X:
            return np.zeros(X_test.shape[0]), np.ones(X_test.shape[0])
        
        X_train = np.array(self.observed_X)
        y_train = np.array(self.normalized_y)
        
        # Compute kernel matrices
        K = self._kernel_matrix(X_train, X_train)
        K += np.eye(len(X_train)) * self.noise_variance  # Add noise
        K_s = self._kernel_matrix(X_train, X_test)
        K_ss = self._kernel_matrix(X_test, X_test)
        
        # Compute mean and variance
        K_inv = np.linalg.inv(K)
        mean = K_s.T @ K_inv @ y_train
        var = np.diag(K_ss - K_s.T @ K_inv @ K_s)
        
        return mean, var
    
    def _negative_log_likelihood(self, params: np.ndarray) -> float:
        """
        Negative log likelihood function for GP hyperparameter optimization.
        
        Args:
            params: Hyperparameters [log_length_scale, log_signal_variance, log_noise_variance]
            
        Returns:
            Negative log likelihood
        """
        if not self.observed_X:
            return 0.0
            
        # Extract hyperparameters
        self.length_scale = np.exp(params[0])
        self.signal_variance = np.exp(params[1])
        self.noise_variance = np.exp(params[2])
        
        X = np.array(self.observed_X)
        y = np.array(self.normalized_y)
        n = len(y)
        
        # Compute kernel matrix
        K = self._kernel_matrix(X)
        K += np.eye(n) * self.noise_variance
        
        try:
            # Compute negative log likelihood
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            nll = 0.5 * (y @ alpha + np.sum(np.log(np.diag(L))) + n * np.log(2 * np.pi))
            return nll
        except np.linalg.LinAlgError:
            # If cholesky decomposition fails, return a large value
            return 1e10
    
    def _optimize_hyperparameters(self):
        """
        Optimize the GP hyperparameters using maximum likelihood estimation.
        """
        if len(self.observed_X) < 2:
            return
            
        # Initial hyperparameters: [log_length_scale, log_signal_variance, log_noise_variance]
        initial_params = np.log(np.array([self.length_scale, self.signal_variance, self.noise_variance]))
        
        # Run optimization
        bounds = [(np.log(1e-3), np.log(10.0)), (np.log(1e-3), np.log(10.0)), (np.log(1e-6), np.log(1.0))]
        result = minimize(self._negative_log_likelihood, initial_params, bounds=bounds, method='L-BFGS-B')
        
        # Update hyperparameters
        optimized_params = result.x
        self.length_scale = np.exp(optimized_params[0])
        self.signal_variance = np.exp(optimized_params[1])
        self.noise_variance = np.exp(optimized_params[2])
    
    def _expected_improvement(self, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        """
        Expected Improvement acquisition function.
        
        Args:
            mean: Predicted mean values
            var: Predicted variance values
            
        Returns:
            Expected improvement values
        """
        # Get the best observed value
        if not self.normalized_y:
            return np.ones_like(mean)
            
        f_best = np.max(self.normalized_y)
        std = np.sqrt(var)
        
        # Compute z for normal CDF
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (mean - f_best - self.exploration_weight) / std
            
        # Compute expected improvement
        ei = (mean - f_best - self.exploration_weight) * norm.cdf(z) + std * norm.pdf(z)
        
        # Handle numerical issues
        ei[std < 1e-10] = 0.0
        return ei
    
    def _normalize_y(self, y: List[float]) -> List[float]:
        """
        Normalize the observed scores for better GP performance.
        
        Args:
            y: List of scores
            
        Returns:
            List of normalized scores
        """
        if not y:
            return []
            
        y = np.array(y)
        y_min = np.min(y)
        y_max = np.max(y)
        
        if y_max > y_min:
            return list((y - y_min) / (y_max - y_min))
        else:
            return list(y - y_min)
    
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
        if len(self.observed_X) < self.random_trials:
            random_params = {}
            for param, values in self.param_space.items():
                random_params[param] = np.random.choice(values)
            return random_params
        
        # Optimize GP hyperparameters
        self._optimize_hyperparameters()
        
        # Generate candidate points
        num_candidates = 1000
        candidates = np.random.random((num_candidates, len(self._param_mapping)))
        
        # Predict mean and variance for candidates
        mean, var = self._gp_predict(candidates)
        
        # Compute acquisition function values
        acq_values = self._expected_improvement(mean, var)
        
        # Find the best candidate
        best_idx = np.argmax(acq_values)
        best_candidate = candidates[best_idx]
        
        # Convert to parameters
        next_params = self._point_to_params(best_candidate)
        
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
        self.observed_X.append(point)
        self.observed_y.append(score)
        
        # Update normalized y values
        self.normalized_y = self._normalize_y(self.observed_y)