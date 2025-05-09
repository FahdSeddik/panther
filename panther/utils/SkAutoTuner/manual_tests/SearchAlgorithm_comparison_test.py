"""
This file provides concrete manual tests for comparing different search algorithms.
It compares GridSearch, RandomSearch, and BayesianOptimization in terms of:
- Time taken
- Best score achieved
- Best parameters found
- Score progression over time

The test uses a real-world optimization scenario (optimizing neural networks)
and includes visualization to help manually analyze the results.
"""

import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from panther.utils.SkAutoTuner.Searching.SearchAlgorithm import SearchAlgorithm
from panther.utils.SkAutoTuner.Searching.GridSearch import GridSearch
from panther.utils.SkAutoTuner.Searching.RandomSearch import RandomSearch
from panther.utils.SkAutoTuner.Searching.BayesianOptimization import BayesianOptimization


class TestFunctionMLP:
    """A real regression problem using a simple MLP model"""
    
    def __init__(self, random_seed=42):
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Load the diabetes dataset
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_seed
        )
        
        # Convert to PyTorch tensors
        self.X_train = torch.FloatTensor(X_train)
        self.y_train = torch.FloatTensor(y_train).reshape(-1, 1)
        self.X_test = torch.FloatTensor(X_test)
        self.y_test = torch.FloatTensor(y_test).reshape(-1, 1)
        
        # Define feature dimensions
        self.input_dim = X_train.shape[1]  # Number of features
        
        print(f"Dataset loaded: {len(self.X_train)} training samples, {len(self.X_test)} test samples")
        print(f"Input features: {self.input_dim}")
    
    def create_model(self, hidden_dim, num_layers, dropout_rate, activation='relu'):
        """Create an MLP model with the specified hyperparameters"""
        layers = []
        
        # Input layer
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Add input layer
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(act_fn)
        layers.append(nn.Dropout(dropout_rate))
        
        # Add hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout_rate))
        
        # Add output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        # Create model
        model = nn.Sequential(*layers)
        return model
    
    def evaluate_model(self, model_params):
        """
        Create and evaluate a model with the given parameters.
        Returns the negative mean squared error (higher is better).
        """
        # Extract parameters
        hidden_dim = model_params['hidden_dim']
        num_layers = model_params['num_layers']
        dropout_rate = model_params['dropout_rate']
        learning_rate = model_params['learning_rate']
        activation = model_params['activation']
        epochs = model_params['epochs']
        
        # Create model
        model = self.create_model(hidden_dim, num_layers, dropout_rate, activation)
        
        # Define loss function and optimizer
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train model
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = model(self.X_train)
            loss = loss_fn(y_pred, self.y_train)
            loss.backward()
            optimizer.step()
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            y_pred = model(self.X_test)
            mse = loss_fn(y_pred, self.y_test).item()
        
        # Return negative MSE as score (higher is better)
        return -mse


class OptimizationExperiment:
    """Run an experiment to compare different search algorithms"""
    
    def __init__(self, param_space, test_function, max_trials=20):
        """
        Initialize the experiment.
        
        Args:
            param_space: Dictionary of parameter names and possible values
            test_function: Function that evaluates parameters and returns a score
            max_trials: Maximum number of trials for each algorithm
        """
        self.param_space = param_space
        self.test_function = test_function
        self.max_trials = max_trials
        
        # Initialize results storage
        self.results = {}
    
    def _run_algorithm(self, algorithm_name, search_algorithm):
        """
        Run a search algorithm and collect results.
        
        Args:
            algorithm_name: Name of the algorithm (for display)
            search_algorithm: Instance of a SearchAlgorithm
            
        Returns:
            Dictionary with results for this algorithm
        """
        # Initialize the algorithm
        search_algorithm.initialize(self.param_space)
        
        # Track results
        trials = []
        best_score = float('-inf')
        best_params = None
        start_time = time.time()
        total_time = 0
        
        # Run trials
        trial_num = 0
        while True:
            # Get next parameters to try
            params = search_algorithm.get_next_params()
            if params is None:
                break  # No more trials
            
            # Evaluate parameters
            trial_start_time = time.time()
            score = self.test_function(params)
            trial_time = time.time() - trial_start_time
            total_time = time.time() - start_time
            
            # Update best score and parameters
            if score > best_score:
                best_score = score
                best_params = params.copy()
            
            # Update the search algorithm
            search_algorithm.update(params, score)
            
            # Record trial results
            trial_num += 1
            trials.append({
                'trial': trial_num,
                'params': params,
                'score': score,
                'time': trial_time,
                'total_time': total_time,
                'best_score': best_score
            })
            
            print(f"{algorithm_name} - Trial {trial_num}/{self.max_trials}: Score = {score:.6f}, Time = {trial_time:.3f}s")
        
        return {
            'name': algorithm_name,
            'trials': trials,
            'best_score': best_score,
            'best_params': best_params,
            'total_time': total_time
        }
    
    def run_experiment(self):
        """Run the experiment with all search algorithms and collect results"""
        algorithms = [
            ('GridSearch', GridSearch()),
            ('RandomSearch', RandomSearch(max_trials=self.max_trials)),
            ('BayesianOptimization', BayesianOptimization(max_trials=self.max_trials, random_trials=3))
        ]
        
        for name, algorithm in algorithms:
            print(f"\n=== Running {name} ===")
            result = self._run_algorithm(name, algorithm)
            self.results[name] = result
        
        return self.results
    
    def print_comparison(self):
        """Print a comparison of the results"""
        print("\n=== Experiment Results ===")
        print(f"{'Algorithm':<20} {'Best Score':<12} {'Total Time (s)':<15} {'# Trials':<10}")
        print("-" * 60)
        
        for name, result in self.results.items():
            print(f"{name:<20} {result['best_score']:<12.6f} {result['total_time']:<15.2f} {len(result['trials']):<10}")
    
    def print_best_params(self):
        """Print the best parameters found by each algorithm"""
        print("\n=== Best Parameters ===")
        
        for name, result in self.results.items():
            print(f"\n{name}:")
            for param, value in result['best_params'].items():
                print(f"  {param}: {value}")
    
    def plot_score_progression(self):
        """Plot the progression of best scores over trials for each algorithm"""
        plt.figure(figsize=(12, 6))
        
        for name, result in self.results.items():
            trials = result['trials']
            trial_nums = [t['trial'] for t in trials]
            best_scores = [t['best_score'] for t in trials]
            
            plt.plot(trial_nums, best_scores, marker='o', label=name)
        
        plt.xlabel('Trial Number')
        plt.ylabel('Best Score (Negative MSE)')
        plt.title('Best Score Progression')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('score_progression.png')
        print("Score progression saved to 'score_progression.png'")
        plt.close()
    
    def plot_score_vs_time(self):
        """Plot the progression of best scores over time for each algorithm"""
        plt.figure(figsize=(12, 6))
        
        for name, result in self.results.items():
            trials = result['trials']
            times = [t['total_time'] for t in trials]
            best_scores = [t['best_score'] for t in trials]
            
            plt.plot(times, best_scores, marker='o', label=name)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Best Score (Negative MSE)')
        plt.title('Best Score vs Time')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('score_vs_time.png')
        print("Score vs time plot saved to 'score_vs_time.png'")
        plt.close()
    
    def plot_param_importance(self):
        """Plot parameter importance for each algorithm based on best params"""
        param_names = list(self.param_space.keys())
        algorithm_names = list(self.results.keys())
        
        # Create normalized values for parameters
        param_values = {}
        for param in param_names:
            param_values[param] = []
            for alg_name in algorithm_names:
                best_params = self.results[alg_name]['best_params']
                possible_values = self.param_space[param]
                
                # Get index of value in possible values
                value = best_params[param]
                if isinstance(possible_values[0], (int, float)):
                    # Normalize numeric values
                    min_val = min(possible_values)
                    max_val = max(possible_values)
                    normalized_value = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                else:
                    # For categorical values, use the index
                    normalized_value = possible_values.index(value) / max(1, len(possible_values) - 1)
                
                param_values[param].append(normalized_value)
        
        # Plot
        plt.figure(figsize=(12, 8))
        bar_width = 0.2
        index = np.arange(len(param_names))
        
        for i, alg_name in enumerate(algorithm_names):
            values = [param_values[param][i] for param in param_names]
            plt.bar(index + i * bar_width, values, bar_width, label=alg_name)
        
        plt.xlabel('Parameters')
        plt.ylabel('Normalized Value')
        plt.title('Best Parameter Values Comparison')
        plt.xticks(index + bar_width, param_names)
        plt.legend()
        
        plt.savefig('param_importance.png')
        print("Parameter importance plot saved to 'param_importance.png'")
        plt.close()


def run_search_algorithm_comparison():
    """Main function to run the search algorithm comparison"""
    print("===== Search Algorithm Comparison Test =====")
    
    # Define parameter space for the MLP model
    param_space = {
        'hidden_dim': [16, 32, 64, 128],
        'num_layers': [1, 2, 3, 4],
        'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'activation': ['relu', 'tanh', 'sigmoid'],
        'epochs': [50, 100, 150, 200]
    }
    
    # Create test function (MLP on real data)
    test_func = TestFunctionMLP()
    
    # Define the evaluation function
    def evaluate_params(params):
        return test_func.evaluate_model(params)
    
    # Run experiment
    experiment = OptimizationExperiment(
        param_space=param_space,
        test_function=evaluate_params,
        max_trials=1000  # Adjust this based on your time constraints
    )
    
    # Run all algorithms
    results = experiment.run_experiment()
    
    # Print results
    experiment.print_comparison()
    experiment.print_best_params()
    
    # Plot results
    experiment.plot_score_progression()
    experiment.plot_score_vs_time()
    experiment.plot_param_importance()
    
    print("\nExperiment completed. Results saved to:")
    print("  - score_progression.png")
    print("  - score_vs_time.png")
    print("  - param_importance.png")


if __name__ == "__main__":
    run_search_algorithm_comparison()