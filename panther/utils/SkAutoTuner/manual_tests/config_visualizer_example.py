import torch
import torch.nn as nn
import torch.nn.functional as F
from panther.utils.SkAutoTuner.ConfigVisualizer import ConfigVisualizer

class SimpleModel(nn.Module):
    """
    A simple model with various layer types for demonstrating the ConfigVisualizer
    """
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleModel, self).__init__()
        
        # Feature extractor part (simulating convolutional layers)
        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )
        
        # Classifier part
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def demonstrate_config_visualizer():
    """
    Demonstrates how to use the ConfigVisualizer to generate layer configurations
    for the SKAutoTuner.
    """
    # Create a simple model
    model = SimpleModel()
    
    print("Model created. Opening configuration visualizer...")
    print("Instructions:")
    print("1. Ctrl+Click (or Cmd+Click on Mac) on layers in the visualization to select them")
    print("2. Configure sketch parameters for the selected layers")
    print("3. Choose whether to tune layers separately or together")
    print("4. Generate and copy the configuration code")
    print("5. Use the generated code with SKAutoTuner")
    
    # Use the ConfigVisualizer to create an interactive visualization with configuration options
    output_path = ConfigVisualizer.create_config_visualization(
        model,
        output_path="model_config_visualization.html",  # Will be saved in the current directory
        open_browser=True  # Set to True to automatically open in browser
    )
    
    print(f"\nConfiguration visualization saved to: {output_path}")
    print("If the browser doesn't open automatically, open the HTML file manually.")

if __name__ == "__main__":
    demonstrate_config_visualizer()
