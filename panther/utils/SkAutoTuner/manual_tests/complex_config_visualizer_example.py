import torch
import torch.nn as nn
import torch.nn.functional as F
from .ConfigVisualizer import ConfigVisualizer

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SimpleConvNet(nn.Module):
    """
    A simple convolutional network for demonstrating the ConfigVisualizer
    with Conv2d layers.
    """
    def __init__(self, in_channels=3, num_classes=10):
        super(SimpleConvNet, self).__init__()
        
        # Encoder blocks
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32),
            nn.MaxPool2d(2, 2),
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2)
        )
        
        # Linear layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x

class TransformerModel(nn.Module):
    """
    A simple transformer-based model for demonstrating the ConfigVisualizer
    with attention layers.
    """
    def __init__(self, vocab_size=5000, d_model=256, nhead=8, num_layers=2, dim_feedforward=1024, max_seq_length=100, num_classes=10):
        super(TransformerModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(max_seq_length, d_model))
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(d_model, nhead)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # Add position encoding
        seq_length = x.size(1)
        x = self.embedding(x)
        x = x + self.pos_encoder[:seq_length, :]
        
        # Reshape for transformer: [seq_len, batch, dim]
        x = x.permute(1, 0, 2)
        
        # Apply transformer
        x = self.transformer_encoder(x)
        
        # Self-attention on the output sequence
        attn_output, _ = self.attention(x, x, x)
        
        # Average pooling across sequence dimension
        x = attn_output.mean(dim=0)
        
        # Classification
        x = self.classifier(x)
        return x

def demonstrate_complex_models():
    """
    Demonstrates the ConfigVisualizer with both convolutional and transformer models
    """
    # Ask which model to visualize
    print("Which model would you like to visualize?")
    print("1. Convolutional Neural Network")
    print("2. Transformer Model")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        # Create a CNN model
        model = SimpleConvNet()
        model_name = "cnn"
    else:
        # Create a Transformer model
        model = TransformerModel()
        model_name = "transformer"
    
    print(f"\nModel created. Opening configuration visualizer for {model_name} model...")
    print("Instructions:")
    print("1. Ctrl+Click (or Cmd+Click on Mac) on layers in the visualization to select them")
    print("2. Configure sketch parameters for the selected layers")
    print("3. Choose whether to tune layers separately or together")
    print("4. Generate and copy the configuration code")
    print("5. Use the generated code with SKAutoTuner")
    
    # Use the ConfigVisualizer
    output_path = ConfigVisualizer.create_config_visualization(
        model,
        output_path=f"{model_name}_config_visualization.html",
        open_browser=True
    )
    
    print(f"\nConfiguration visualization saved to: {output_path}")

if __name__ == "__main__":
    demonstrate_complex_models()
