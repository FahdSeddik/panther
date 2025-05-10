import torch
import torch.nn as nn
import torch.nn.functional as F
from panther.utils.SkAutoTuner.ModelVisualizer import ModelVisualizer

# Define a custom block with multiple layers
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

# Define a custom model architecture that combines different types of layers
class CustomModel(nn.Module):
    """A custom model architecture that combines different types of layers"""
    def __init__(self, num_classes=10):
        super(CustomModel, self).__init__()
        
        # Encoder part with convolutional blocks
        self.encoder = nn.Sequential(
            CustomBlock(3, 64),
            nn.MaxPool2d(2, 2),
            CustomBlock(64, 128),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder for reconstruction with upsampling
        self.decoder = nn.Sequential(
            CustomBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            CustomBlock(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )
        
        # Classifier with attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # Encode input
        features = self.encoder(x)
        
        # Reconstruct input using decoder
        reconstruction = self.decoder(features)
        
        # Apply self-attention to features
        batch_size, C, H, W = features.shape
        features_flat = features.view(batch_size, C, -1).permute(0, 2, 1)  # [B, HW, C]
        attn_output, _ = self.attention(features_flat, features_flat, features_flat)
        attn_output = attn_output.permute(0, 2, 1).view(batch_size, C, H, W)
        
        # Combine features and attention output
        features_combined = features + attn_output
        
        # Classify
        classification = self.classifier(features_combined)
        
        return classification, reconstruction

def main():
    # Create a model instance
    model = CustomModel(num_classes=10)
    
    print("Model created successfully.")
    print("\nModel structure (text representation):")
    # Print the model structure in text format
    ModelVisualizer.print_module_tree(model)
    
    print("\nCreating interactive visualization...")
    # Generate an interactive HTML visualization of the model
    output_path = ModelVisualizer.create_interactive_visualization(
        model,
        output_path="model_visualization.html",  # Will be saved in the current directory
        open_browser=True  # Set to True to automatically open in browser
    )
    
    print(f"\nVisualization saved to: {output_path}")
    print("Open the HTML file in a web browser to explore the interactive visualization.")
    print("Features of the visualization:")
    print("- Click on nodes to see detailed information about each layer")
    print("- Use the search bar to find specific layers")
    print("- Double-click on nodes to collapse/expand sections")
    print("- Mouse wheel or zoom buttons for zooming in/out")
    print("- Right-click on nodes for additional options")

if __name__ == "__main__":
    main()