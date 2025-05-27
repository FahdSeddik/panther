# SKAutoTuner Search Algorithms

This directory contains the search algorithm implementations for the SKAutoTuner framework.

## Overview

The search algorithms are responsible for efficiently exploring the parameter space to find optimal configurations for sketched neural network layers. Each algorithm provides different strategies for navigating the parameter space with different trade-offs between exploration, exploitation, and computational efficiency.

## Base Class

### SearchAlgorithm

All search algorithms inherit from the `SearchAlgorithm` base class:

```python
class SearchAlgorithm:
    def __init__(self):
        # ...
    
    def search(self, param_space, eval_func):
        """
        Search for optimal parameters.
        
        Args:
            param_space: Dictionary mapping parameter names to lists of possible values
            eval_func: Function that takes a parameter combination and returns a score
            
        Returns:
            Tuple of (best_params, results)
        """
        raise NotImplementedError
```

## Available Algorithms

### GridSearch

Exhaustive search over all possible parameter combinations.

```python
from panther.utils.SkAutoTuner import GridSearch

search_algo = GridSearch()
```

**Advantages**:
- Guaranteed to find the global optimum
- Simple and deterministic

**Disadvantages**:
- Computationally expensive for large parameter spaces
- Scales exponentially with the number of parameters

### RandomSearch

Random sampling from the parameter space.

```python
from panther.utils.SkAutoTuner import RandomSearch

search_algo = RandomSearch(num_samples=30)
```

**Advantages**:
- More efficient than grid search for high-dimensional spaces
- Good exploration of the parameter space

**Disadvantages**:
- Not guaranteed to find the global optimum
- Performance depends on the number of samples

### BayesianOptimization

Model-based optimization using Gaussian processes.

```python
from panther.utils.SkAutoTuner import BayesianOptimization

search_algo = BayesianOptimization(
    initial_points=5,
    n_iterations=20,
    exploitation_ratio=0.8
)
```

**Advantages**:
- Efficient exploration of the parameter space
- Good balance between exploration and exploitation
- Works well with expensive evaluation functions

**Disadvantages**:
- Computationally expensive for large numbers of iterations
- May converge to local optima

### EvolutionaryAlgorithm

Genetic algorithm-based optimization.

```python
from panther.utils.SkAutoTuner import EvolutionaryAlgorithm

search_algo = EvolutionaryAlgorithm(
    population_size=20,
    n_generations=10,
    mutation_rate=0.1,
    crossover_rate=0.7
)
```

**Advantages**:
- Good for complex, non-convex parameter spaces
- Can escape local optima
- Parallelizable

**Disadvantages**:
- Requires tuning of metaparameters
- May require many evaluations

### ParticleSwarmOptimization

Swarm intelligence-based optimization.

```python
from panther.utils.SkAutoTuner import ParticleSwarmOptimization

search_algo = ParticleSwarmOptimization(
    n_particles=20,
    n_iterations=15,
    inertia_weight=0.8,
    cognitive_weight=1.5,
    social_weight=1.5
)
```

**Advantages**:
- Good for continuous parameter spaces
- Efficient global search
- Parallelizable

**Disadvantages**:
- May converge prematurely
- Performance depends on parameter settings

### SimulatedAnnealing

Probabilistic optimization with temperature cooling.

```python
from panther.utils.SkAutoTuner import SimulatedAnnealing

search_algo = SimulatedAnnealing(
    initial_temp=100,
    cooling_rate=0.95,
    n_iterations=50
)
```

**Advantages**:
- Can escape local optima
- Works well for discrete and continuous spaces
- Simple to implement

**Disadvantages**:
- Slow convergence
- Performance depends on cooling schedule

### TreeParzenEstimator

Sequential model-based optimization using density estimation.

```python
from panther.utils.SkAutoTuner import TreeParzenEstimator

search_algo = TreeParzenEstimator(
    n_startup_trials=10,
    n_ei_candidates=24,
    n_trials=50
)
```

**Advantages**:
- Works well for conditional parameter spaces
- Efficient for mixed discrete-continuous spaces
- Good for high-dimensional spaces

**Disadvantages**:
- Complex implementation
- Requires sufficient startup trials

### Hyperband

Bandit-based approach for resource allocation.

```python
from panther.utils.SkAutoTuner import Hyperband

search_algo = Hyperband(
    max_iter=81,
    eta=3,
    resource_param='epochs'
)
```

**Advantages**:
- Efficient for expensive evaluations
- Automatically allocates resources to promising configurations
- Good for large parameter spaces

**Disadvantages**:
- Requires a resource parameter
- Complex implementation

## Algorithm Selection Guide

| Algorithm | Best For | Parameter Space | Evaluation Cost |
|-----------|----------|-----------------|-----------------|
| GridSearch | Small parameter spaces | Discrete | Low to Medium |
| RandomSearch | Medium parameter spaces | Discrete/Continuous | Medium |
| BayesianOptimization | Expensive evaluations | Continuous | High |
| EvolutionaryAlgorithm | Complex, non-convex spaces | Discrete/Continuous | Medium to High |
| ParticleSwarmOptimization | Continuous spaces | Continuous | Medium |
| SimulatedAnnealing | Rough landscapes | Discrete/Continuous | Medium |
| TreeParzenEstimator | Conditional parameters | Discrete/Continuous | High |
| Hyperband | Large spaces with early stopping | Discrete/Continuous | High |

## Integration with SKAutoTuner

The search algorithms are used when initializing the `SKAutoTuner`:

```python
from panther.utils.SkAutoTuner import SKAutoTuner, BayesianOptimization

# Initialize the auto-tuner with a specific search algorithm
tuner = SKAutoTuner(
    model=model,
    configs=configs,
    accuracy_eval_func=evaluate_accuracy,
    search_algorithm=BayesianOptimization(),
    verbose=True
)
```

## Custom Search Algorithms

To create a custom search algorithm, inherit from the `SearchAlgorithm` base class:

```python
from panther.utils.SkAutoTuner import SearchAlgorithm

class MyCustomAlgorithm(SearchAlgorithm):
    def __init__(self, custom_param1, custom_param2):
        super().__init__()
        self.custom_param1 = custom_param1
        self.custom_param2 = custom_param2
    
    def search(self, param_space, eval_func):
        # Implement your search strategy
        best_params = {}
        results = []
        
        # ... custom search logic ...
        
        return best_params, results
``` 