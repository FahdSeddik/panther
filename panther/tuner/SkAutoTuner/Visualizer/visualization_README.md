# SKAutoTuner Visualization Components

This document describes the visualization components available in the SKAutoTuner framework.

## Overview

The SKAutoTuner provides powerful visualization tools to help understand model architectures, tuning configurations, and optimization results. These tools aid in the interpretation of complex neural network structures and tuning processes.

## Key Components

### ModelVisualizer

The `ModelVisualizer` class provides tools for visualizing neural network architectures:

```python
from panther.tuner.SkAutoTuner import ModelVisualizer

visualizer = ModelVisualizer(model)
```

#### Key Features

- **Model Architecture Visualization**: Display the structure of neural networks
- **Layer Type Highlighting**: Colorize different layer types for better understanding
- **Parameter Statistics**: Show parameter counts and distributions across layers
- **Interactive Diagrams**: Generate interactive visualizations for complex models

#### Example Usage

```python
from panther.tuner.SkAutoTuner import ModelVisualizer
import torch.nn as nn

# Create a model
model = nn.Sequential(
    nn.Conv2d(3, 16, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(32*13*13, 10)
)

# Create a visualizer
visualizer = ModelVisualizer(model)

# Generate a visualization
visualizer.visualize(save_path="model_architecture.png")

# Show parameter statistics
visualizer.parameter_summary()

# Visualize specific layers
visualizer.visualize_layer("0", save_path="conv_layer.png")
```

### ConfigVisualizer

The `ConfigVisualizer` class provides tools for visualizing tuning configurations:

```python
from panther.tuner.SkAutoTuner import ConfigVisualizer

config_vis = ConfigVisualizer(tuner.configs)
```

#### Key Features

- **Parameter Space Visualization**: Display the parameter spaces for tuning
- **Layer Group Visualization**: Show which layers are tuned together
- **Parameter Distribution Analysis**: Analyze the distribution of parameters
- **Search Space Complexity**: Calculate and visualize the complexity of the search space

#### Example Usage

```python
from panther.tuner.SkAutoTuner import ConfigVisualizer, TuningConfigs, LayerConfig

# Create a configuration
configs = TuningConfigs([
    LayerConfig(
        layer_names=["features.0", "features.3"],
        params={
            "sketch_type": ["channel", "spatial"],
            "sketch_factor": [0.25, 0.5, 0.75]
        }
    )
])

# Create a visualizer
config_vis = ConfigVisualizer(configs)

# Visualize configurations
config_vis.visualize_configs(save_path="config_visualization.png")

# Analyze search space
config_vis.search_space_analysis()
```

## Tuning Results Visualization

The `SKAutoTuner` class itself provides methods for visualizing tuning results:

```python
# After tuning
tuner.visualize_tuning_results(save_path="tuning_results.png")
```

### Visualization Types

- **Parameter Impact Plots**: Visualize how different parameters affect performance
- **Accuracy vs. Efficiency Plots**: Show the trade-off between accuracy and efficiency
- **Learning Curves**: Display how performance evolves during tuning
- **Layer-specific Results**: Show tuning results for individual layers

### Advanced Customization

Both visualization components support customization options:

```python
# Customized model visualization
visualizer.visualize(
    save_path="custom_viz.png",
    figsize=(12, 8),
    dpi=300,
    show_layer_names=True,
    show_layer_types=True,
    show_parameter_counts=True,
    color_scheme="pastel"
)

# Customized tuning results visualization
tuner.visualize_tuning_results(
    save_path="detailed_results.png",
    show_plot=True,
    figsize=(15, 10),
    max_cols=3,
    plot_type="bar",
    highlight_best=True,
    sort_by="accuracy"
)
```

## Integration with Jupyter Notebooks

The visualization components are designed to work well within Jupyter notebooks:

```python
# In a Jupyter notebook
from panther.tuner.SkAutoTuner import ModelVisualizer
import matplotlib.pyplot as plt

# Create a visualizer
visualizer = ModelVisualizer(model)

# Display visualization inline
plt.figure(figsize=(12, 8))
visualizer.visualize(show_plot=True)
plt.show()
```

## Exporting Visualizations

Visualizations can be exported in various formats:

```python
# Export as PNG
visualizer.visualize(save_path="model.png")

# Export as SVG (for better quality in publications)
visualizer.visualize(save_path="model.svg")

# Export as interactive HTML
visualizer.visualize_interactive(save_path="model.html")
``` 