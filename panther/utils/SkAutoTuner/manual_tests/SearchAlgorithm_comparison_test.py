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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from panther.utils.SkAutoTuner.Searching.SearchAlgorithm import SearchAlgorithm
from panther.utils.SkAutoTuner.Searching.GridSearch import GridSearch
from panther.utils.SkAutoTuner.Searching.RandomSearch import RandomSearch
from panther.utils.SkAutoTuner.Searching.BayesianOptimization import BayesianOptimization
# from panther.utils.SkAutoTuner.manual_tests.plot_helpers import plot_aggregated_results, create_results_dir

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
    
    def __init__(self, param_space, test_function, max_trials=20, output_dir=None):
        """
        Initialize the experiment.
        
        Args:
            param_space: Dictionary of parameter names and possible values
            test_function: Function that evaluates parameters and returns a score
            max_trials: Maximum number of trials for each algorithm
            output_dir: Directory to save results to
        """
        self.param_space = param_space
        self.test_function = test_function
        self.max_trials = max_trials
        self.output_dir = output_dir
        
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
            ('BayesianOptimization', BayesianOptimization(
                max_trials=self.max_trials, 
                random_trials=3,
                exploration_weight=0.8  # Increase exploitation to show BO advantage
            ))
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
        
        file_path = os.path.join(self.output_dir, 'score_progression.png')
        plt.savefig(file_path)
        print(f"Score progression saved to '{file_path}'")
        plt.close()
        
        # Also save individual algorithm plots
        for name, result in self.results.items():
            plt.figure(figsize=(10, 5))
            trials = result['trials']
            trial_nums = [t['trial'] for t in trials]
            best_scores = [t['best_score'] for t in trials]
            
            plt.plot(trial_nums, best_scores, marker='o', label=name, color='blue')
            plt.xlabel('Trial Number')
            plt.ylabel('Best Score (Negative MSE)')
            plt.title(f'{name} - Best Score Progression')
            plt.grid(True)
            
            file_path = os.path.join(self.output_dir, f'{name}_score_progression.png')
            plt.savefig(file_path)
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
        
        file_path = os.path.join(self.output_dir, 'score_vs_time.png')
        plt.savefig(file_path)
        print(f"Score vs time plot saved to '{file_path}'")
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
        
        file_path = os.path.join(self.output_dir, 'param_importance.png')
        plt.savefig(file_path)
        print(f"Parameter importance plot saved to '{file_path}'")
        plt.close()

    def analyze_early_performance(self):
        """Analyze how quickly each algorithm finds good solutions"""
        # Find the global best score across all algorithms
        global_best = max([r['best_score'] for r in self.results.values()])
        threshold = 0.9 * global_best  # 90% of best score
        
        print("\n=== Early Performance Analysis ===")
        print(f"{'Algorithm':<20} {'Trials to 90% of Best':<25} {'Time to 90% of Best (s)':<25}")
        print("-" * 70)
        
        plt.figure(figsize=(12, 6))
        
        for name, result in self.results.items():
            trials = result['trials']
            # Find first trial reaching threshold
            for i, trial in enumerate(trials):
                if trial['best_score'] >= threshold:
                    print(f"{name:<20} {i+1:<25} {trial['total_time']:<25.2f}")
                    plt.axvline(x=i+1, linestyle='--', label=f"{name} threshold", alpha=0.5)
                    break
            else:
                print(f"{name:<20} Never reached{'':15} Never reached{'':15}")
            
            # Plot trial scores
            trial_nums = [t['trial'] for t in trials]
            scores = [t['score'] for t in trials]  # Use actual scores, not best_scores
            plt.scatter(trial_nums, scores, label=f"{name} trials", alpha=0.7)
        
        plt.axhline(y=threshold, color='r', linestyle='-', label='90% Threshold')
        plt.xlabel('Trial Number')
        plt.ylabel('Score')
        plt.title('Individual Trial Scores and Threshold')
        plt.legend()
        plt.grid(True)
        
        file_path = os.path.join(self.output_dir, 'early_performance.png')
        plt.savefig(file_path)
        print(f"Early performance analysis saved to '{file_path}'")
        plt.close()


def run_search_algorithm_comparison():
    """Main function to run the search algorithm comparison"""
    print("===== Search Algorithm Comparison Test =====")
    
    # Create results directory with timestamp
    base_output_dir = create_results_dir()
    print(f"Results will be saved to: {base_output_dir}")
    
    # Define parameter space for the MLP model
    param_space = {
        'hidden_dim': [64, 16, 128],
        'num_layers': [1, 4],
        'dropout_rate': [0.0, 0.5],
        # 'learning_rate': [0.05, 0.001, 0.1],
        'activation': ['relu', 'tanh', 'sigmoid'],
        'epochs': [50, 200]
        # 'hidden_dim': [16, 32, 64, 128],
        # 'num_layers': [1, 2, 3, 4],
        # 'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        # 'learning_rate': [0.001, 0.01, 0.05, 0.1],
        # 'activation': ['relu', 'tanh', 'sigmoid'],
        # 'epochs': [50, 100, 150, 200]
    }

    params_space_size = np.prod([len(values) for values in param_space.values()])
    print(f"Total parameter space size: {params_space_size}")
    
    # Save parameter space information
    with open(os.path.join(base_output_dir, 'experiment_info.txt'), 'w') as f:
        f.write(f"Total parameter space size: {params_space_size}\n\n")
        f.write("Parameter space:\n")
        for param, values in param_space.items():
            f.write(f"  {param}: {values}\n")
    
    # Number of repetitions for statistical significance
    num_repetitions = 1
    all_results = {
        'GridSearch': [],
        'RandomSearch': [],
        'BayesianOptimization': []
    }
    
    # Run multiple experiments with different seeds
    for seed in range(num_repetitions):
        print(f"\n\n==== Running Experiment with Seed {seed} ====")
        
        # Create seed-specific output directory
        seed_output_dir = os.path.join(base_output_dir, f"seed_{seed}")
        os.makedirs(seed_output_dir, exist_ok=True)
        
        # Create test function (MLP on real data)
        test_func = TestFunctionMLP(random_seed=seed)
        
        # Define the evaluation function
        def evaluate_params(params):
            return test_func.evaluate_model(params)
        
        # Run experiment with limited trials to show algorithm efficiency
        max_trials = min(int(params_space_size * 0.5), 50)  # 5% of space or 50 trials max
        experiment = OptimizationExperiment(
            param_space=param_space,
            test_function=evaluate_params,
            max_trials=max_trials,
            output_dir=seed_output_dir
        )
        
        # Run all algorithms
        results = experiment.run_experiment()
        
        # Print results for this seed
        experiment.print_comparison()
        experiment.print_best_params()
          # Plot results for this seed
        experiment.plot_score_progression()
        experiment.plot_score_vs_time()
        experiment.plot_param_importance()
        experiment.analyze_early_performance()
        
        # Save detailed results to CSV
        for alg_name, result in results.items():
            # Convert trials to DataFrame
            trials_df = pd.DataFrame([
                {
                    'trial': t['trial'],
                    'score': t['score'],
                    'best_score': t['best_score'],
                    'time': t['time'],
                    'total_time': t['total_time'],
                    **{f"param_{k}": v for k, v in t['params'].items()}
                }
                for t in result['trials']
            ])
            
            # Save DataFrame to CSV
            trials_df.to_csv(
                os.path.join(seed_output_dir, f"{alg_name}_trials.csv"),
                index=False
            )
        
        # Store results for this seed
        for alg_name, result in results.items():
            all_results[alg_name].append(result)
    
    # Aggregate and print summary statistics
    print("\n===== Summary Statistics =====")
    print(f"{'Algorithm':<20} {'Avg Best Score':<15} {'Std Dev':<10} {'Avg Time (s)':<15}")
    print("-" * 60)
    
    summary_data = []
    for alg_name, results_list in all_results.items():
        best_scores = [r['best_score'] for r in results_list]
        times = [r['total_time'] for r in results_list]
        avg_score = np.mean(best_scores)
        std_score = np.std(best_scores)
        avg_time = np.mean(times)
        
        summary_data.append({
            'Algorithm': alg_name,
            'Avg_Best_Score': avg_score,
            'Std_Dev': std_score,
            'Avg_Time': avg_time
        })
        
        print(f"{alg_name:<20} {avg_score:<15.6f} {std_score:<10.6f} {avg_time:<15.2f}")
    
    # Save summary to CSV
    pd.DataFrame(summary_data).to_csv(
        os.path.join(base_output_dir, 'summary_statistics.csv'),
        index=False
    )
    
    # Plot aggregated results
    plot_aggregated_results(all_results, base_output_dir)
    
    print("\nExperiment completed. Results saved to:")
    print(f"  - {base_output_dir}")
    
    # Create a summary HTML file
    with open(os.path.join(base_output_dir, 'summary.html'), 'w') as f:
        f.write("<html><head><title>Search Algorithm Comparison Results</title></head><body>\n")
        f.write("<h1>Search Algorithm Comparison Results</h1>\n")
        f.write(f"<p>Run completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
        f.write(f"<p>Parameter space size: {params_space_size}</p>\n")
        f.write(f"<p>Number of repetitions: {num_repetitions}</p>\n")
        
        f.write("<h2>Aggregate Results</h2>\n")
        f.write("<img src='algorithm_efficiency.png' width='800'><br>\n")
        
        f.write("<h2>Summary Statistics</h2>\n")
        f.write("<table border='1'>\n")
        f.write("<tr><th>Algorithm</th><th>Avg Best Score</th><th>Std Dev</th><th>Avg Time (s)</th></tr>\n")
        for data in summary_data:
            f.write(f"<tr><td>{data['Algorithm']}</td><td>{data['Avg_Best_Score']:.6f}</td>")
            f.write(f"<td>{data['Std_Dev']:.6f}</td><td>{data['Avg_Time']:.2f}</td></tr>\n")
        f.write("</table>\n")
        
        f.write("<h2>Individual Runs</h2>\n")
        for seed in range(num_repetitions):
            f.write(f"<h3>Seed {seed}</h3>\n")
            f.write(f"<a href='seed_{seed}/score_progression.png'><img src='seed_{seed}/score_progression.png' width='400'></a>\n")
            f.write(f"<a href='seed_{seed}/score_vs_time.png'><img src='seed_{seed}/score_vs_time.png' width='400'></a><br>\n")
            f.write(f"<a href='seed_{seed}/param_importance.png'><img src='seed_{seed}/param_importance.png' width='400'></a>\n")
            f.write(f"<a href='seed_{seed}/early_performance.png'><img src='seed_{seed}/early_performance.png' width='400'></a><br>\n")
        
        f.write("</body></html>\n")


if __name__ == "__main__":
    run_search_algorithm_comparison()