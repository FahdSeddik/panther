import torch
from transformers import BertForMaskedLM
from panther.utils.SkAutoTuner.ModelVisualizer import ModelVisualizer
import torch.nn as nn

def print_linear_layer_params(model):
    """
    Print parameters of all Linear layers in the BERT model.
    This provides detailed information about each Linear layer's configuration.
    """
    print("\n===== Linear Layer Parameters =====")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"Layer: {name}")
            print(f"  - in_features: {module.in_features}")
            print(f"  - out_features: {module.out_features}")
            print(f"  - bias: {module.bias is not None}")
            # Print shape information
            weight_shape = module.weight.shape
            print(f"  - weight shape: {weight_shape}")
            if module.bias is not None:
                bias_shape = module.bias.shape
                print(f"  - bias shape: {bias_shape}")
            print("----------------------------------------")

def visualize_bert_model():
    """
    Load a BERT model and create an interactive visualization of its structure.
    This function demonstrates how to use ModelVisualizer with a BERT model.
    """
    print("Loading BERT model for visualization...")
    
    # Load BERT model
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    
    # Set model to evaluation mode
    model.eval()
    
    # Print basic model information
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: BERT base uncased")
    print(f"Total trainable parameters: {total_params:,}")
    
    # Print details of all linear layers in the model
    print_linear_layer_params(model)
    
    # Print model tree structure in console
    print("\nModel tree structure:")
    ModelVisualizer.print_module_tree(model)
    
    # Create an interactive visualization
    print("\nCreating interactive visualization...")
    output_path = ModelVisualizer.create_interactive_visualization(
        model=model,
        output_path="bert_model_visualization.html",
        open_browser=True
    )
    
    print(f"\nVisualization saved to: {output_path}")
    print("The visualization should have opened in your default web browser.")
    print("If not, you can open it manually by navigating to the file.")
    
    # Additional information about using the visualization
    print("\nUsing the visualization:")
    print("- Hover over modules to see details like parameter counts")
    print("- Click on modules to expand/collapse their children")
    print("- Use the search box to find specific modules")
    print("- The interactive interface allows exploring the entire model structure")

if __name__ == "__main__":
    # Call the visualization function
    visualize_bert_model()
