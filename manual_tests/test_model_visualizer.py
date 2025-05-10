import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelVisualizer import ModelVisualizer

class CustomBlock(nn.Module):
    """A custom block with multiple layers"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CustomBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class CustomModel(nn.Module):
    """A simple custom model for testing the visualizer"""
    def __init__(self, num_classes=10):
        super(CustomModel, self).__init__()
        
        # Encoder part
        self.encoder = nn.Sequential(
            CustomBlock(3, 64),
            nn.MaxPool2d(2, 2),
            CustomBlock(64, 128),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            CustomBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            CustomBlock(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        classification = self.classifier(features)
        return classification, reconstruction

def test_model_visualizer():
    # Create a simple model
    model = CustomModel()
    
    print("Model created successfully. Structure:")
    # Print the model structure (text-based)
    print(model)
    
    # Use the ModelVisualizer to create an interactive visualization
    print("\nCreating interactive visualization...")
    output_path = ModelVisualizer.create_interactive_visualization(
        model,
        output_path="model_visualization.html",  # Will be saved in the current directory
        open_browser=True  # Set to True to automatically open in browser
    )
    
    print(f"\nVisualization saved to: {output_path}")
    print("If the browser doesn't open automatically, open the HTML file manually.")

if __name__ == "__main__":
    test_model_visualizer()