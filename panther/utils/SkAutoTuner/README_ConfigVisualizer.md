# SKAutoTuner ConfigVisualizer

The ConfigVisualizer is a tool that allows users to visually select layers from a PyTorch model and generate configurations for the SKAutoTuner. This helps users identify layer names and available sketching parameters without having to write configurations manually.

## Features

- Interactive visualization of the model architecture
- Visual selection of layers for tuning
- Automatic detection of supported layer types
- Configuration of sketching parameters for selected layers
- Automatic generation of `LayerConfig` code that can be used with the SKAutoTuner
- Support for both individual and grouped layer configurations

## Usage

### Basic Usage

```python
from panther.utils.SkAutoTuner.ConfigVisualizer import ConfigVisualizer
import torch.nn as nn

# Create or load your model
model = YourModel()

# Use the ConfigVisualizer
ConfigVisualizer.create_config_visualization(
    model,
    output_path="model_config.html",  # Optional: specify output path
    open_browser=True  # Optional: automatically open in browser
)
```

### Using the Generated Configuration

After selecting layers and parameters in the visual interface, you can copy the generated code and use it in your SKAutoTuner setup:

```python
# This is the code you would copy from the ConfigVisualizer
from panther.utils.SkAutoTuner.Configs.LayerConfig import LayerConfig
from panther.utils.SkAutoTuner.Configs.TuningConfigs import TuningConfigs

# Define layer configurations
configs = [
    LayerConfig(
        layer_names="features.3",
        params={"sketch": ["qr", "svd"], "sketch_ratio": [0.3, 0.5, 0.7]},
        separate=True,
        copy_weights=True
    ),
    LayerConfig(
        layer_names=["classifier.0", "classifier.2"],
        params={"sketch": ["svd"], "sketch_ratio": [0.5]},
        separate=False,
        copy_weights=True
    )
]

# Create tuning configuration
tuning_config = TuningConfigs(configs)

# Use with SKAutoTuner
from panther.utils.SkAutoTuner.SKAutoTuner import SKAutoTuner

tuner = SKAutoTuner(
    model=model,
    configs=tuning_config,
    accuracy_eval_func=your_accuracy_function,
    # ... other parameters
)

tuner.tune()
```

## How to Use the Visual Interface

1. **Select Layers**: Ctrl+Click (or Cmd+Click on Mac) on layers in the visualization to select them.
2. **Configure Parameters**: Check the boxes for sketching parameters you want to try for the selected layers.
3. **Set Configuration Options**:
   - Toggle "Configure layers separately" to tune each layer individually or as a group.
   - Toggle "Copy weights when replacing layers" to copy weights during tuning or not.
4. **Generate Configuration**: Click "Generate Configuration" to create the Python code.
5. **Copy Code**: Click "Copy Code" to copy the generated code to the clipboard.

## Example Files

- `config_visualizer_example.py` - Simple example with a basic model
- `complex_config_visualizer_example.py` - Advanced example with CNN and Transformer models

## Note

Only layer types that are supported by the SKAutoTuner for sketching will be available for selection. The current supported types include:
- Linear
- Conv2d
- MultiheadAttention

## Requirements

- PyTorch
- Graphviz (for visualization)
