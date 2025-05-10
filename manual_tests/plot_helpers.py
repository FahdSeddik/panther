"""
Helper functions for plotting results from search algorithm comparisons
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def create_results_dir():
    """Create a timestamped results directory and return its path"""
    # Create base results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create a timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"run_{timestamp}")
    os.makedirs(run_dir)
    
    return run_dir

def plot_aggregated_results(all_results, output_dir):
    """Plot aggregated results across multiple seeds"""
    # Efficiency plot (best score vs trials)
    plt.figure(figsize=(12, 6))
    
    for alg_name, results_list in all_results.items():
        # Get average best score at each trial across all seeds
        max_trials = max([len(r['trials']) for r in results_list])
        avg_scores = np.zeros(max_trials)
        counts = np.zeros(max_trials)
        
        for result in results_list:
            trials = result['trials']
            for i, trial in enumerate(trials):
                avg_scores[i] += trial['best_score']
                counts[i] += 1
        
        # Calculate average (handling zeros in counts)
        for i in range(max_trials):
            if counts[i] > 0:
                avg_scores[i] /= counts[i]
            else:
                # For missing trials, use last known score
                avg_scores[i] = avg_scores[i-1] if i > 0 else 0
        
        plt.plot(range(1, max_trials+1), avg_scores, label=alg_name, linewidth=2)
    
    plt.xlabel('Number of Trials')
    plt.ylabel('Average Best Score')
    plt.title('Algorithm Efficiency Comparison (Higher is Better)')
    plt.legend()
    plt.grid(True)
    
    file_path = os.path.join(output_dir, 'algorithm_efficiency.png')
    plt.savefig(file_path)
    print(f"Algorithm efficiency plot saved to '{file_path}'")
    plt.close()