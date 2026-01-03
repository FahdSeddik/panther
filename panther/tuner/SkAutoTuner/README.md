# SKAutoTuner

A powerful automatic tuning framework for sketched neural networks with industry-standard HPO.

## Overview

SKAutoTuner is a specialized toolkit for optimizing and tuning sketch-based neural network layers. It allows for automated exploration of parameter spaces to find optimal configurations that balance accuracy and efficiency. The toolkit is designed to work with PyTorch models and integrates **Optuna** as the default search backend, providing state-of-the-art hyperparameter optimization.

## Key Features

- **Industry-Standard HPO**: Uses Optuna by default for state-of-the-art optimization (TPE sampler)
- **Mixed Parameter Spaces**: Supports categorical, integer, and continuous parameters via `ParamSpec`
- **Constraint-Based Optimization**: Maximize speed while maintaining accuracy ≥ threshold
- **Multiple Search Algorithms**: Including Optuna (recommended), Grid Search, Random Search, Bayesian Optimization, and more
- **Layer Configuration**: Flexible configuration system for defining tuning parameters
- **Visualization Tools**: Built-in visualization for tuning results and model configurations
- **Full Metrics Tracking**: Records accuracy, speed, and score for every trial

## Package Structure

```
SKAutoTuner/
├── __init__.py                 # Package exports
├── SKAutoTuner.py              # Main auto-tuner implementation
├── layer_type_mapping.py       # Mappings for supported layer types
├── Configs/                    # Configuration components
│   ├── __init__.py
│   ├── LayerConfig.py          # Layer configuration classes
│   ├── TuningConfigs.py        # Tuning configuration classes
│   ├── ParamSpec.py            # Parameter distribution specs (Categorical, Int, Float)
│   ├── ParamsResolver.py       # Auto parameter generation
│   └── LayerNameResolver.py    # Layer name resolution utilities
├── Searching/                  # Search algorithm implementations
│   ├── __init__.py
│   ├── SearchAlgorithm.py      # Base search algorithm class
│   ├── OptunaSearch.py         # Optuna-backed search (RECOMMENDED)
│   ├── GridSearch.py           # Grid search implementation
│   ├── RandomSearch.py         # Random search implementation
│   ├── BayesianOptimization.py # Bayesian optimization (legacy)
│   └── ...                     # Other legacy algorithms
└── Visualizer/                 # Visualization tools
    └── ...
```

## Installation

SKAutoTuner is included as part of the Panther framework. Optuna is automatically installed as a dependency.

## Usage

### Basic Usage (Recommended: Optuna)

```python
from panther.tuner.SkAutoTuner import SKAutoTuner, TuningConfigs, LayerConfig
from panther.tuner.SkAutoTuner.Configs import Int, Categorical
import torch.nn as nn

# Define a model to tune
model = MyNeuralNetwork()

# Define evaluation functions
def evaluate_accuracy(model):
    # Custom evaluation logic
    return accuracy_score

def evaluate_speed(model):
    # Custom speed measurement (throughput - higher is better)
    return throughput

# Create tuning configurations with modern ParamSpec types
configs = TuningConfigs([
    LayerConfig(
        layer_names=["conv1", "conv2"],
        params={
            "num_terms": Categorical([1, 2, 3]),  # Categorical choices
            "low_rank": Int(8, 128, step=8),      # Integer range with step
        },
        separate=True
    )
])

# Initialize the auto-tuner (uses OptunaSearch by default)
tuner = SKAutoTuner(
    model=model,
    configs=configs,
    accuracy_eval_func=evaluate_accuracy,
    accuracy_threshold=0.90,              # Minimum acceptable accuracy
    optmization_eval_func=evaluate_speed, # Maximize this after meeting threshold
    verbose=True
)

# Run the tuning process
tuner.tune()

# Apply the best parameters found
optimized_model = tuner.apply_best_params()

# Get detailed results with accuracy and speed metrics
results_df = tuner.get_results_dataframe()
print(results_df[["layer_name", "num_terms", "low_rank", "accuracy", "speed", "score"]])
```

### Using Legacy List-Based Parameters

For backward compatibility, you can still use lists:

```python
configs = TuningConfigs([
    LayerConfig(
        layer_names=["fc1"],
        params={
            "num_terms": [1, 2, 3],      # Legacy list format
            "low_rank": [8, 16, 32, 64], # Legacy list format
        },
    )
])
```

### Customizing Optuna Search

```python
from panther.tuner.SkAutoTuner.Searching import OptunaSearch
from optuna.samplers import CmaEsSampler

# Use CMA-ES sampler for continuous optimization
tuner = SKAutoTuner(
    model=model,
    configs=configs,
    accuracy_eval_func=evaluate_accuracy,
    search_algorithm=OptunaSearch(
        n_trials=200,
        sampler=CmaEsSampler(seed=42),
        seed=42,
    ),
    verbose=True
)

# Access the underlying Optuna study for advanced analysis
study = tuner.search_algorithm.study
print(study.best_params)
print(study.trials_dataframe())
```

### Using Legacy Search Algorithms

Legacy algorithms are still available but not recommended for new projects:

```python
from panther.tuner.SkAutoTuner.Searching import GridSearch, RandomSearch, BayesianOptimization

# Grid Search (exhaustive, use for small spaces only)
tuner = SKAutoTuner(
    model=model,
    configs=configs,
    accuracy_eval_func=evaluate_accuracy,
    search_algorithm=GridSearch(),
)

# Random Search
tuner = SKAutoTuner(
    model=model,
    configs=configs,
    accuracy_eval_func=evaluate_accuracy,
    search_algorithm=RandomSearch(n_trials=50),
)
```

## Configuration Components

### LayerConfig

The `LayerConfig` class defines which layers to tune and what parameters to explore:

- `layer_names`: List of layer names to tune
- `params`: Dictionary mapping parameter names to possible values
- `separate`: Whether to tune each layer separately or jointly
- `copy_weights`: Whether to copy weights from the original layer

### TuningConfigs

The `TuningConfigs` class holds multiple `LayerConfig` objects for a complete tuning configuration.

### ParamSpec Types

Modern parameter specification types for mixed search spaces:

- **Categorical(choices)**: Fixed set of choices (any type)
- **Int(low, high, step=1, log=False)**: Integer range with optional step and log scale
- **Float(low, high, step=None, log=False)**: Float range with optional step and log scale

```python
from panther.tuner.SkAutoTuner.Configs import Categorical, Int, Float

params = {
    "num_terms": Categorical([1, 2, 3, 4]),
    "low_rank": Int(8, 256, step=8),
    "dropout": Float(0.0, 0.5, step=0.1),
    "learning_rate": Float(1e-5, 1e-2, log=True),
}
```

## Search Algorithms

### Recommended: OptunaSearch

OptunaSearch provides industry-standard HPO with state-of-the-art samplers:

- **TPESampler** (default): Tree-structured Parzen Estimator
- **CmaEsSampler**: Covariance Matrix Adaptation Evolution Strategy
- **RandomSampler**: Simple random sampling
- **GridSampler**: Exhaustive grid search

### Legacy Algorithms (Maintenance Mode)

These algorithms are still available but not recommended for new projects:

- **GridSearch**: Exhaustive search over all parameter combinations
- **RandomSearch**: Random sampling from parameter space
- **BayesianOptimization**: Model-based optimization using Gaussian processes
- **EvolutionaryAlgorithm**: Genetic algorithm-based optimization
- **ParticleSwarmOptimization**: Swarm intelligence-based optimization
- **SimulatedAnnealing**: Probabilistic optimization with temperature cooling
- **TreeParzenEstimator**: Sequential model-based optimization
- **Hyperband**: Bandit-based approach for resource allocation

## Performance Tracking

The auto-tuner tracks comprehensive metrics during the tuning process:

- **accuracy**: Accuracy score for each parameter combination
- **speed**: Speed/throughput metric when `optmization_eval_func` is provided
- **score**: Final objective score (speed if accuracy ≥ threshold, else -inf)
- Best parameter combinations for each layer
- Full trial history accessible via `get_results_dataframe()`

## License

This tool is part of the Panther framework and is subject to the same licensing terms. 