# SKAutoTuner

A powerful automatic tuning framework for sketched neural networks.

## Overview

SKAutoTuner is a specialized toolkit for optimizing and tuning sketch-based neural network layers. It allows for automated exploration of parameter spaces to find optimal configurations that balance accuracy and efficiency. The toolkit is designed to work with PyTorch models and provides various search algorithms for hyperparameter optimization.

## Key Features

- **Automated Parameter Tuning**: Optimize sketch parameters for neural network layers
- **Multiple Search Algorithms**: Including Grid Search, Random Search, Bayesian Optimization, and more
- **Layer Configuration**: Flexible configuration system for defining tuning parameters
- **Visualization Tools**: Built-in visualization for tuning results and model configurations
- **Performance Evaluation**: Evaluate both accuracy and efficiency metrics

## Package Structure

```
SKAutoTuner/
├── __init__.py                 # Package exports
├── SKAutoTuner.py              # Main auto-tuner implementation
├── ModelVisualizer.py          # Model visualization utilities
├── ConfigVisualizer.py         # Configuration visualization utilities
├── layer_type_mapping.py       # Mappings for supported layer types
├── Configs/                    # Configuration components
│   ├── __init__.py
│   ├── LayerConfig.py          # Layer configuration classes
│   ├── TuningConfigs.py        # Tuning configuration classes
│   └── LayerNameResolver.py    # Layer name resolution utilities
└── Searching/                  # Search algorithm implementations
    ├── __init__.py
    ├── SearchAlgorithm.py      # Base search algorithm class
    ├── GridSearch.py           # Grid search implementation
    ├── RandomSearch.py         # Random search implementation
    ├── BayesianOptimization.py # Bayesian optimization implementation
    ├── EvolutionaryAlgorithm.py # Evolutionary algorithm implementation
    ├── ParticleSwarmOptimization.py # PSO implementation
    ├── SimulatedAnnealing.py   # Simulated annealing implementation
    ├── TreeParzenEstimator.py  # TPE implementation
    └── Hyperband.py            # Hyperband implementation
```

## Installation

SKAutoTuner is included as part of the Panther framework. No separate installation is required if you have the Panther package installed.

## Usage

### Basic Usage

```python
from panther.tuner.SkAutoTuner import SKAutoTuner, TuningConfigs, LayerConfig
import torch.nn as nn

# Define a model to tune
model = MyNeuralNetwork()

# Define a function to evaluate model accuracy
def evaluate_accuracy(model):
    # Custom evaluation logic
    return accuracy_score

# Create tuning configurations
configs = TuningConfigs([
    LayerConfig(
        layer_names=["conv1", "conv2"],  # Layers to tune
        params={
            "sketch_type": ["channel", "spatial"],
            "sketch_factor": [0.25, 0.5, 0.75]
        },
        separate=True  # Tune each layer separately
    )
])

# Initialize the auto-tuner
tuner = SKAutoTuner(
    model=model,
    configs=configs,
    accuracy_eval_func=evaluate_accuracy,
    verbose=True
)

# Run the tuning process
tuner.tune()

# Apply the best parameters found
optimized_model = tuner.apply_best_params()

# Visualize the results
tuner.visualize_tuning_results(save_path="tuning_results.png")
```

### Advanced Usage

#### Using Different Search Algorithms

```python
from panther.tuner.SkAutoTuner import SKAutoTuner, BayesianOptimization

# Initialize with Bayesian Optimization
tuner = SKAutoTuner(
    model=model,
    configs=configs,
    accuracy_eval_func=evaluate_accuracy,
    search_algorithm=BayesianOptimization(),
    verbose=True
)
```

#### Balancing Accuracy and Efficiency

```python
# Function to evaluate efficiency (e.g., inference time)
def evaluate_efficiency(model):
    # Custom efficiency measurement
    return speed_score

tuner = SKAutoTuner(
    model=model,
    configs=configs,
    accuracy_eval_func=evaluate_accuracy,
    accuracy_threshold=0.90,  # Minimum acceptable accuracy
    optmization_eval_func=evaluate_efficiency,  # Optimize this after meeting accuracy threshold
    verbose=True
)
```

#### Visualizing Model Configurations

```python
from panther.tuner.SkAutoTuner import ModelVisualizer, ConfigVisualizer

# Visualize the model architecture
visualizer = ModelVisualizer(model)
visualizer.visualize(save_path="model_architecture.png")

# Visualize configuration options
config_vis = ConfigVisualizer(tuner.configs)
config_vis.visualize_configs(save_path="config_options.png")
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

## Search Algorithms

SKAutoTuner supports multiple search algorithms for hyperparameter optimization:

- **GridSearch**: Exhaustive search over all parameter combinations
- **RandomSearch**: Random sampling from parameter space
- **BayesianOptimization**: Model-based optimization using Gaussian processes
- **EvolutionaryAlgorithm**: Genetic algorithm-based optimization
- **ParticleSwarmOptimization**: Swarm intelligence-based optimization
- **SimulatedAnnealing**: Probabilistic optimization with temperature cooling
- **TreeParzenEstimator**: Sequential model-based optimization
- **Hyperband**: Bandit-based approach for resource allocation

## Visualization Tools

### ModelVisualizer

Visualizes the model architecture with layer details and connections.

### ConfigVisualizer

Visualizes tuning configurations and parameter spaces for better understanding of the search space.

## Performance Tracking

The auto-tuner tracks performance metrics during the tuning process:

- Accuracy scores for each parameter combination
- Efficiency metrics when specified
- Best parameter combinations for each layer
- Tuning history for analysis

## License

This tool is part of the Panther framework and is subject to the same licensing terms. 