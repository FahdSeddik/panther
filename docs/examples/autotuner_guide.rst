AutoTuner Guide
===============

This guide shows how to use Panther's AutoTuner to automatically optimize sketching parameters for your specific use case.

Introduction to AutoTuning
---------------------------

Choosing optimal sketching parameters (``num_terms`` and ``low_rank``) can be challenging. The AutoTuner provides multiple search algorithms including Bayesian optimization, Grid Search, Random Search, and more to automatically find the best parameters for your model and dataset.

**Why AutoTune?**

- **Better Performance**: Find parameters that work best for your specific data
- **Save Time**: Avoid manual hyperparameter search
- **Multiple Algorithms**: Choose from Bayesian optimization, Grid Search, Random Search, and advanced methods
- **Model Visualization**: Visualize your model architecture and identify layers to optimize

Basic AutoTuner Usage
----------------------

**Importing the AutoTuner Components**

.. code-block:: python

   import torch
   import torch.nn as nn
   from panther.nn import SKLinear
   from panther.tuner.SkAutoTuner import (
       SKAutoTuner,
       LayerConfig,
       TuningConfigs,
       GridSearch,
       RandomSearch,
       BayesianOptimization,
       ModelVisualizer
   )
   
   # Setting up device
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")

**Creating a Simple Model**

.. code-block:: python

   # Define a simple model with regular PyTorch layers
   class SimpleModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(784, 512)
           self.relu1 = nn.ReLU()
           self.fc2 = nn.Linear(512, 256)
           self.relu2 = nn.ReLU()
           self.fc3 = nn.Linear(256, 10)
       
       def forward(self, x):
           x = x.view(-1, 784)
           x = self.relu1(self.fc1(x))
           x = self.relu2(self.fc2(x))
           x = self.fc3(x)
           return x
   
   model = SimpleModel().to(device)

**Defining Evaluation Function and Tuning Configuration**

.. code-block:: python

   # Define evaluation function - this function assesses the model's performance
   def accuracy_eval_func(model):
       """Evaluate model accuracy on validation data."""
       model.eval()
       correct = 0
       total = 0
       
       with torch.no_grad():
           # Create dummy validation data (replace with real data)
           for _ in range(20):
               x = torch.randn(32, 784, device=device)
               y = torch.randint(0, 10, (32,), device=device)
               output = model(x)
               _, predicted = torch.max(output.data, 1)
               total += y.size(0)
               correct += (predicted == y).sum().item()
       
       accuracy = correct / total
       return accuracy
   
   # Configure which layers to tune and their parameter ranges
   config1 = LayerConfig(
       layer_names=["fc1"],  # Name of layer to tune
       params={
           "num_terms": [2, 4, 8],      # Try these num_terms values
           "low_rank": [32, 64, 128]     # Try these low_rank values
       },
       separate=True,         # Tune this layer independently
       copy_weights=True      # Copy weights from original layer
   )
   
   config2 = LayerConfig(
       layer_names=["fc2"],
       params={
           "num_terms": [2, 4],
           "low_rank": [32, 64]
       },
       separate=True,
       copy_weights=True
   )
   
   tuning_configs = TuningConfigs(configs=[config1, config2])
   
   # Create the AutoTuner
   tuner = SKAutoTuner(
       model=model,
       configs=tuning_configs,
       accuracy_eval_func=accuracy_eval_func,
       search_algorithm=GridSearch(),  # Can also use RandomSearch() or BayesianOptimization()
       verbose=True
   )
   
   # Run tuning
   print("Starting AutoTuning...")
   tuner.tune()
   
   # Get best parameters
   best_params = tuner.get_best_params()
   print("\\nBest parameters found:")
   for layer_name, params_info in best_params.items():
       print(f"Layer: {layer_name}, Params: {params_info['params']}")
   
   # Apply best parameters to model
   optimized_model = tuner.apply_best_params()
   
   # Visualize results
   tuner.visualize_tuning_results(save_path="tuning_results.png", show_plot=False)

Using Different Search Algorithms
----------------------------------

The AutoTuner supports multiple search algorithms:

**Grid Search (Default)**

.. code-block:: python

   from panther.tuner.SkAutoTuner import GridSearch
   
   tuner = SKAutoTuner(
       model=model,
       configs=tuning_configs,
       accuracy_eval_func=accuracy_eval_func,
       search_algorithm=GridSearch(),  # Tries all combinations
       verbose=True
   )
   
   tuner.tune()

**Random Search**

.. code-block:: python

   from panther.tuner.SkAutoTuner import RandomSearch
   
   tuner = SKAutoTuner(
       model=model,
       configs=tuning_configs,
       accuracy_eval_func=accuracy_eval_func,
       search_algorithm=RandomSearch(n_trials=50),  # Try 50 random combinations
       verbose=True
   )
   
   tuner.tune()

**Bayesian Optimization**

.. code-block:: python

   from panther.tuner.SkAutoTuner import BayesianOptimization
   
   tuner = SKAutoTuner(
       model=model,
       configs=tuning_configs,
       accuracy_eval_func=accuracy_eval_func,
       search_algorithm=BayesianOptimization(n_trials=30),
       verbose=True
   )
   
   tuner.tune()

Advanced Search Algorithms
---------------------------

The AutoTuner also supports additional advanced search algorithms:

* **SimulatedAnnealing** - Optimization using simulated annealing
* **Hyperband** - Resource-efficient hyperparameter optimization
* **EvolutionaryAlgorithm** - Genetic algorithm-based search
* **ParticleSwarmOptimization** - PSO-based optimization
* **TreeParzenEstimator** - TPE-based Bayesian optimization

Example with Simulated Annealing:

.. code-block:: python

   from panther.tuner.SkAutoTuner import SimulatedAnnealing
   
   tuner = SKAutoTuner(
       model=model,
       configs=tuning_configs,
       accuracy_eval_func=accuracy_eval_func,
       search_algorithm=SimulatedAnnealing(),
       verbose=True
   )
   
   tuner.tune()

Model Visualization
-------------------

The AutoTuner includes a ModelVisualizer to help you understand your model structure:

.. code-block:: python

   from panther.tuner.SkAutoTuner import ModelVisualizer
   
   # Print model structure
   ModelVisualizer.print_module_tree(model)
   
   # This will output something like:
   # SimpleModel
   #   ├── fc1: Linear(784, 512)
   #   ├── relu1: ReLU()
   #   ├── fc2: Linear(512, 256)
   #   ├── relu2: ReLU()
   #   └── fc3: Linear(256, 10)

Best Practices for AutoTuning
------------------------------

**1. Start with Quick Evaluations**

Use a subset of your data during tuning to speed up the search process:

.. code-block:: python

   def accuracy_eval_func(model):
       """Fast evaluation for tuning."""
       model.eval()
       correct = 0
       total = 0
       
       with torch.no_grad():
           # Use only a subset of validation data
           for batch_idx, (data, target) in enumerate(val_loader):
               if batch_idx >= 20:  # Limit to 20 batches
                   break
               
               data, target = data.to(device), target.to(device)
               output = model(data)
               _, predicted = torch.max(output.data, 1)
               total += target.size(0)
               correct += (predicted == target).sum().item()
       
       return correct / total if total > 0 else 0.0

**2. Start with Coarse Grid, Then Refine**

Begin with a coarse parameter grid and refine around promising regions:

.. code-block:: python

   # Phase 1: Coarse search
   coarse_config = LayerConfig(
       layer_names=["fc1"],
       params={
           "num_terms": [2, 8, 16],       # Wide range
           "low_rank": [32, 128, 256]     # Wide range
       },
       separate=True,
       copy_weights=True
   )
   
   # Phase 2: Fine-tune around best from Phase 1
   # Suppose best was num_terms=8, low_rank=128
   fine_config = LayerConfig(
       layer_names=["fc1"],
       params={
           "num_terms": [6, 7, 8, 9, 10],          # Narrow range
           "low_rank": [96, 112, 128, 144, 160]    # Narrow range
       },
       separate=True,
       copy_weights=True
   )

**3. Monitor Resource Usage**

Track memory and computation time during tuning:

.. code-block:: python

   import time
   
   def accuracy_eval_with_monitoring(model):
       """Evaluation with resource monitoring."""
       start_time = time.time()
       
       if torch.cuda.is_available():
           torch.cuda.reset_peak_memory_stats()
       
       # Perform evaluation
       accuracy = evaluate_model(model)
       
       elapsed_time = time.time() - start_time
       
       if torch.cuda.is_available():
           peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
           print(f"Evaluation time: {elapsed_time:.2f}s, "
                 f"Peak memory: {peak_memory:.2f} GB")
       
       return accuracy

**4. Save and Resume Tuning**

For long tuning sessions, save progress:

.. code-block:: python

   # After tuning
   tuner.tune()
   best_params = tuner.get_best_params()
   
   # Save results
   import pickle
   with open('tuning_results.pkl', 'wb') as f:
       pickle.dump(best_params, f)
   
   # Load later
   with open('tuning_results.pkl', 'rb') as f:
       loaded_params = pickle.load(f)

Tips for Better Results
------------------------

1. **Use consistent evaluation**: Always evaluate on the same validation set
2. **Set random seeds**: Ensure reproducible results across runs
3. **Consider parameter constraints**: Ensure total parameters < original model
4. **Balance speed vs. accuracy**: Choose search algorithm based on your time budget
5. **Visualize results**: Use the built-in visualization to understand parameter effects

This guide covers the essential aspects of using Panther's AutoTuner effectively.
       
       # Evaluate on smaller validation set
       accuracy = quick_evaluate(model, validation_subset_size=500)
       
       return accuracy

**2. Use Early Stopping**

.. code-block:: python

   def evaluation_with_early_stopping(num_terms, low_rank):
       """Evaluation with early stopping to save time."""
       
       model = create_model(int(num_terms), int(low_rank))
       
       best_val_acc = 0
       patience = 3
       patience_counter = 0
       
       for epoch in range(20):  # Max epochs
           train_one_epoch(model)
           val_acc = validate_model(model)
           
           if val_acc > best_val_acc:
               best_val_acc = val_acc
               patience_counter = 0
           else:
               patience_counter += 1
           
           if patience_counter >= patience:
               break  # Early stopping
       
       return best_val_acc

**3. Cache Expensive Computations**

.. code-block:: python

   from functools import lru_cache
   
   @lru_cache(maxsize=128)
   def cached_evaluation(num_terms, low_rank):
       """Cache results to avoid re-evaluating same parameters."""
       return expensive_evaluation_function(num_terms, low_rank)

**4. Parallel Evaluation (Advanced)**

.. code-block:: python

   from joblib import Parallel, delayed
   
   def parallel_autotuning():
       """Use parallel evaluation for faster tuning."""
       
       def parallel_objective_wrapper(params_dict):
           return evaluate_model_function(**params_dict)
       
       # This requires a custom implementation
       # Standard SkAutoTuner doesn't support parallel evaluation yet
       # But you can implement batch evaluation
       
       # Generate parameter combinations
       param_combinations = generate_parameter_grid()
       
       # Evaluate in parallel
       results = Parallel(n_jobs=4)(
           delayed(parallel_objective_wrapper)(params) 
           for params in param_combinations
       )
       
       # Find best
       best_idx = np.argmax(results)
       return param_combinations[best_idx], results[best_idx]

Interpreting AutoTuning Results
-------------------------------

**Analyzing Parameter Sensitivity**

.. code-block:: python

   def analyze_tuning_results(tuner):
       """Analyze which parameters are most important."""
       
       # Get all evaluated points (if tuner stores them)
       evaluations = tuner.get_evaluation_history()  # Hypothetical method
       
       # Analyze parameter correlations
       import pandas as pd
       import matplotlib.pyplot as plt
       
       df = pd.DataFrame(evaluations)
       
       # Correlation matrix
       correlation_matrix = df.corr()
       print("Parameter correlations:")
       print(correlation_matrix)
       
       # Plot parameter vs. performance
       fig, axes = plt.subplots(2, 2, figsize=(12, 10))
       
       axes[0, 0].scatter(df['num_terms'], df['score'])
       axes[0, 0].set_xlabel('num_terms')
       axes[0, 0].set_ylabel('Score')
       
       axes[0, 1].scatter(df['low_rank'], df['score'])
       axes[0, 1].set_xlabel('low_rank')
       axes[0, 1].set_ylabel('Score')
       
       # Parameter interaction
       scatter = axes[1, 0].scatter(df['num_terms'], df['low_rank'], c=df['score'], cmap='viridis')
       axes[1, 0].set_xlabel('num_terms')
       axes[1, 0].set_ylabel('low_rank')
       plt.colorbar(scatter, ax=axes[1, 0])
       
       plt.tight_layout()
       plt.show()

This comprehensive guide covers all aspects of using Panther's AutoTuner effectively. Remember to start with quick evaluations and progressively refine your search for the best results.
