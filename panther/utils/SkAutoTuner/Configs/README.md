# SKAutoTuner Configuration Components

This directory contains the configuration components for the SKAutoTuner framework.

## Overview

The configuration components define how the auto-tuning process should be structured, which layers to tune, what parameters to explore, and how to resolve layer names in complex models.

## Key Components

### LayerConfig

`LayerConfig` defines the configuration for tuning one or more layers:

```python
class LayerConfig:
    def __init__(
        self, 
        layer_names: List[str], 
        params: Dict[str, List[Any]], 
        separate: bool = True,
        copy_weights: bool = True
    ):
        # ...
```

#### Parameters

- `layer_names`: A list of layer names to tune
- `params`: A dictionary mapping parameter names to lists of possible values
- `separate`: Whether to tune each layer separately (True) or jointly (False)
- `copy_weights`: Whether to copy weights from the original layer

#### Example

```python
config = LayerConfig(
    layer_names=["features.0", "features.3"], 
    params={
        "sketch_type": ["channel", "spatial"],
        "sketch_factor": [0.25, 0.5, 0.75]
    },
    separate=True,
    copy_weights=True
)
```

### TuningConfigs

`TuningConfigs` is a container for multiple `LayerConfig` objects:

```python
class TuningConfigs:
    def __init__(self, configs: List[LayerConfig]):
        self.configs = configs
        # ...
```

#### Parameters

- `configs`: A list of `LayerConfig` objects

#### Example

```python
tuning_configs = TuningConfigs([
    LayerConfig(
        layer_names=["features.0", "features.3"],
        params={
            "sketch_type": ["channel", "spatial"],
            "sketch_factor": [0.25, 0.5, 0.75]
        }
    ),
    LayerConfig(
        layer_names=["classifier.0"],
        params={
            "sketch_type": ["row", "column"],
            "sketch_factor": [0.3, 0.6, 0.9]
        }
    )
])
```

### LayerNameResolver

`LayerNameResolver` provides utilities for resolving layer names in complex model architectures:

```python
class LayerNameResolver:
    def __init__(self, model: nn.Module, layer_map: Dict[str, nn.Module]):
        self.model = model
        self.layer_map = layer_map
        # ...
```

#### Key Methods

- `resolve(layer_names)`: Resolves a list of layer names, which can include wildcards and patterns
- `resolve_wildcard(pattern)`: Resolves a wildcard pattern to a list of matching layer names
- `resolve_regex(pattern)`: Resolves a regex pattern to a list of matching layer names
- `resolve_type(type_name)`: Resolves a layer type name to a list of matching layer names

#### Example

```python
# Resolve all convolutional layers
resolver = LayerNameResolver(model, layer_map)
conv_layers = resolver.resolve(["*.Conv2d"])

# Resolve layers by regex pattern
feature_layers = resolver.resolve(["features\\.\\d+"])

# Resolve specific layer types
linear_layers = resolver.resolve(["Linear"])
```

## Advanced Usage

### Pattern Matching

The `LayerNameResolver` supports several types of patterns for matching layers:

1. **Exact Names**: `"features.0"`, `"classifier.1"`
2. **Wildcards**: `"features.*"`, `"*.Conv2d"`
3. **Regular Expressions**: `"features\\.\\d+"`, `"classifier\\.[13]"`
4. **Layer Types**: `"Conv2d"`, `"Linear"`

### Parameter Space Definition

The `params` dictionary in `LayerConfig` defines the parameter space to explore:

```python
params = {
    "sketch_type": ["channel", "spatial", "row", "column"],
    "sketch_factor": [0.25, 0.5, 0.75],
    "custom_param": [1, 2, 3]
}
```

### Joint vs. Separate Tuning

- **Separate Tuning** (`separate=True`): Each layer is tuned independently
- **Joint Tuning** (`separate=False`): All layers in the group are tuned together with the same parameters

## Integration with SKAutoTuner

The configuration components are used to initialize the `SKAutoTuner`:

```python
from panther.utils.SkAutoTuner import SKAutoTuner, TuningConfigs, LayerConfig

# Define configurations
configs = TuningConfigs([
    LayerConfig(
        layer_names=["features.*"],
        params={
            "sketch_type": ["channel", "spatial"],
            "sketch_factor": [0.25, 0.5, 0.75]
        }
    )
])

# Initialize the auto-tuner
tuner = SKAutoTuner(
    model=model,
    configs=configs,
    accuracy_eval_func=evaluate_accuracy,
    verbose=True
)
``` 