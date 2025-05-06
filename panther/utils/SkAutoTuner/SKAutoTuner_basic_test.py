import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Import components
from panther.utils.SkAutoTuner.SKAutoTuner import SKAutoTuner
from panther.utils.SkAutoTuner.Configs.LayerConfig import LayerConfig
from panther.utils.SkAutoTuner.Configs.TuningConfigs import TuningConfigs
from panther.utils.SkAutoTuner.Searching.GridSearch import GridSearch
from panther.utils.SkAutoTuner.Searching.RandomSearch import RandomSearch
from panther.utils.SkAutoTuner.Searching.BayesianOptimization import BayesianOptimization
from panther.nn.linear_tr import SKLinear_triton
from panther.nn.conv2d import SKConv2d
from panther.nn.attention import RandMultiHeadAttention

# Setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

##################################### HELPERS #######################################

def dump_tensor_info(tensor, name="Tensor"):
    """Print details about a tensor"""
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    print(f"  - Values: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")
    print(f"  - First few values: {tensor.flatten()[:5]}")

def measure_time(func, *args, n_runs=100, warmup=10):
  """Measure execution time of a function"""
  # Warmup
  for _ in range(warmup):
      func(*args)

  # Timed runs
  start = time.time()
  for _ in range(n_runs):
      func(*args)
  end = time.time()

  return (end - start) / n_runs

# Define accuracy evaluation function for model model
def model_accuracy_func(model):
    # In a real scenario, this would compute accuracy on validation data
    # For this example, we'll use a simple proxy for accuracy:
    # How close the model's output is to the expected output

    # Get reference output from the original trained model
    with torch.no_grad():
        reference_output = model_trained(
            model_feat_input,
            model_img_input,
            model_seq_input
        )
        
        test_output = model(
            model_feat_input,
            model_img_input,
            model_seq_input
        )
    
    # Calculate mean squared error between outputs
    mse = nn.functional.mse_loss(test_output, reference_output)
    # Convert to a similarity score where higher is better (max is 1.0)
    similarity = 1.0 / (1.0 + mse.item())
    
    return similarity

# Define speed evaluation function for model model (higher is better)
def model_speed_func(model):
    # Create forward pass function for timing
    def forward_pass(model, feat, img, seq):
        with torch.no_grad():
            return model(feat, img, seq)
    
    inference_time = measure_time(
        forward_pass, 
        model, 
        model_feat_input, 
        model_img_input, 
        model_seq_input,
        n_runs=20
    )
    return 1.0 / inference_time  # Higher is better

# Define evaluation function for model model
def model_eval_func(model):
    # Create forward pass function for timing
    def forward_pass(model, feat, img, seq):
        with torch.no_grad():
            return model(feat, img, seq)
    
    inference_time = measure_time(
        forward_pass, 
        model, 
        model_feat_input, 
        model_img_input, 
        model_seq_input,
        n_runs=20
    )
    return 1.0 / inference_time  # Higher is better

def test_tuner(configs_to_use, search_algorithm):
    # Create a fresh copy of the model for tuning
    model_tunned_grid = modelModel().to(device)
    model_tunned_grid.load_state_dict(model_trained.state_dict())


    # Create tuner with search - make sure we start with the original model
    grid_tuner = SKAutoTuner(
        model=model_tunned_grid,
        configs=configs_to_use,
        eval_func=model_eval_func,
        search_algorithm=search_algorithm,
        accuracy_threshold=0,  # Set a threshold for accuracy
        speed_eval_func=model_speed_func,
        verbose=True
    )

    # Run tuning
    print("\nRunning search tuning...")
    start_time = time.time()
    best_params = grid_tuner.tune()
    tuning_time = time.time() - start_time
    print(f"Tuning completed in {tuning_time:.2f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Best parameters from getParams: {grid_tuner.get_best_params()}")

    # check the tunned model should be the same as the original model exactly
    print("\nTunned model layer types before applying best params:")
    for name, module in model_tunned_grid.named_modules():
        print(f"{name}: {type(module).__name__}")

    # assertions on the tunner model to be the exact same model before tunning
    # Since tuning doesn't change the layers until apply_best_params is called,
    # let's verify that the layers are still the original types and have the same parameters
    for name, module in model_tunned_grid.named_modules():
        if "." not in name:  # Skip the main module
            continue
        
        # Get the corresponding module from the original trained model
        original_module = model_trained.get_submodule(name)
        
        # First check that the types are the same (no sketched layers yet)
        assert type(module) == type(original_module), f"Layer {name} has been changed from {type(original_module)} to {type(module)}!"
        
        # For layers with parameters, check they are identical
        if isinstance(module, nn.Linear):
            assert torch.allclose(module.weight, original_module.weight), f"Layer {name} weight mismatch!"
            if module.bias is not None and original_module.bias is not None:
                assert torch.allclose(module.bias, original_module.bias), f"Layer {name} bias mismatch!"
            print(f"Layer {name} is still a normal Linear layer with identical parameters.")
        
        elif isinstance(module, nn.Conv2d):
            assert torch.allclose(module.weight, original_module.weight), f"Layer {name} weight mismatch!"
            if module.bias is not None and original_module.bias is not None:
                assert torch.allclose(module.bias, original_module.bias), f"Layer {name} bias mismatch!"
            print(f"Layer {name} is still a normal Conv2d layer with identical parameters.")
        
        elif isinstance(module, nn.MultiheadAttention):
            assert torch.allclose(module.in_proj_weight, original_module.in_proj_weight), f"Layer {name} in_proj_weight mismatch!"
            assert torch.allclose(module.out_proj.weight, original_module.out_proj.weight), f"Layer {name} out_proj.weight mismatch!"
            
            if module.in_proj_bias is not None and original_module.in_proj_bias is not None:
                assert torch.allclose(module.in_proj_bias, original_module.in_proj_bias), f"Layer {name} in_proj_bias mismatch!"
            if module.out_proj.bias is not None and original_module.out_proj.bias is not None:
                assert torch.allclose(module.out_proj.bias, original_module.out_proj.bias), f"Layer {name} out_proj.bias mismatch!"
            
            print(f"Layer {name} is still a normal MultiheadAttention layer with identical parameters.")

    # Apply best parameters
    model_tunned_grid = grid_tuner.apply_best_params()

    # check the tunned model should reflect the applied best parameters
    print("\nTunned model layer types after applying best params:")
    for name, module in model_tunned_grid.named_modules():
        print(f"{name}: {type(module).__name__}")

    # Analyze results
    results_df = grid_tuner.get_results_dataframe()
    print("\nTuning results summary:")
    print(results_df)

    # visualize the tunning results using the new methods
    print("\nVisualizing tuning results...")
    # Use the built-in visualization methods
    grid_tuner.visualize_tuning_results(save_path="layer_param_tuning_results.png")

##############################################################################################################

class modelModel(nn.Module):
    def __init__(self, feat_dim=20, seq_len=10, img_channels=3, img_size=16):
        super().__init__()
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.img_size = img_size
        
        # Linear part
        self.fc1 = nn.Linear(feat_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        
        # CNN part
        self.conv = nn.Conv2d(img_channels, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        # Attention part
        self.attention = nn.MultiheadAttention(
            embed_dim=16,
            num_heads=2,
            batch_first=True
        )
        
        # Output
        # We'll combine outputs from all parts
        flat_cnn_size = 8 * (img_size // 2) * (img_size // 2)
        combined_size = 16 + flat_cnn_size + 16
        self.output = nn.Linear(combined_size, 1)
    
    def forward(self, feat, img, seq):
        # Process features with linear layers
        feat_out = nn.functional.relu(self.fc1(feat))
        feat_out = self.fc2(feat_out)
        
        # Process image with CNN
        cnn_out = nn.functional.relu(self.conv(img))
        cnn_out = self.pool(cnn_out)
        cnn_out = cnn_out.reshape(cnn_out.size(0), -1)  # Use reshape instead of view
        
        # Process sequence with attention
        seq_out, _ = self.attention(seq, seq, seq)
        seq_out = seq_out[:, 0, :]  # Use first token
        
        # Combine all outputs
        combined = torch.cat([feat_out, cnn_out, seq_out], dim=1)
        output = self.output(combined)
        
        return output

# model data
model_feat = torch.randn(50, 20).to(device)
model_img = torch.randn(50, 3, 16, 16).to(device)
model_seq = torch.randn(50, 10, 16).to(device)
model_labels = torch.randn(50, 1).to(device)
# Single input examples for inference
model_feat_input = torch.randn(1, 20).to(device)
model_img_input = torch.randn(1, 3, 16, 16).to(device)
model_seq_input = torch.randn(1, 10, 16).to(device)

# create the model to train
model_trained = modelModel().to(device)
print(f"model: {model_trained}")

# Train the model
print("\nTraining the model model...")
criterion = nn.MSELoss()
optimizer = optim.Adam(model_trained.parameters(), lr=0.001)

# Create dataset and dataloader
model_dataset = TensorDataset(model_feat, model_img, model_seq, model_labels)
model_loader = DataLoader(model_dataset, batch_size=10, shuffle=True)

# Training loop
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for feat_batch, img_batch, seq_batch, labels_batch in model_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model_trained(feat_batch, img_batch, seq_batch)
        
        # Calculate loss
        loss = criterion(outputs, labels_batch)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(model_loader):.4f}")

print("Training completed!")

########################################################### now we have a model trained #####################

# Test direct layer replacement (without tuning)

model_replaced = modelModel().to(device)
model_replaced.load_state_dict(model_trained.state_dict())

# Create config for model model (multiple layer types)
configs_to_use = TuningConfigs([
    LayerConfig(
        layer_names=["fc1", "fc2"],
        params={
            "num_terms": [1, 2],
            "low_rank": [8, 16],
        },
        separate=False  # Tune linear layers together
    ),
    LayerConfig(
        layer_names=["conv"],
        params={
            "num_terms": [1, 2],
            "low_rank": [4, 8],
        },
        separate=True
    ),
    LayerConfig(
        layer_names=["attention"],
        params={
            "num_random_features": [16, 32],
            "kernel_fn": ["softmax", "relu"],
        },
        separate=True
    )
])

# Create tuner for model model
model_tuner = SKAutoTuner(
    model=model_replaced,
    configs=configs_to_use,
    eval_func=model_eval_func,
    search_algorithm=GridSearch(),
    verbose=True
)

print("Replacing model linear layers with sketched versions (no tuning)...")
model_replaced = model_tuner.replace_without_tuning()

# print the state dicts
print("\nOriginal model layer types:")
for name, module in model_trained.named_modules():
    print(f"{name}: {type(module).__name__}")

print("\nReplaced model layer types:")
for name, module in model_replaced.named_modules():
    print(f"{name}: {type(module).__name__}")

# Test if model still works
print("\nTesting if replaced model still works...")
with torch.no_grad():
    replaced_output = model_replaced(
        model_feat_input, 
        model_img_input, 
        model_seq_input
    )

# Time comparison
def model_forward(model, feat, img, seq):
    with torch.no_grad():
        return model(feat, img, seq)
    
print("\nComparing inference times:")
original_time = measure_time(model_forward, model_trained, model_feat_input, model_img_input, model_seq_input, n_runs=100)
replaced_time = measure_time(model_forward, model_replaced, model_feat_input, model_img_input, model_seq_input, n_runs=100)

print(f"Original model inference time: {original_time * 1000:.4f} ms")
print(f"Replaced model inference time: {replaced_time * 1000:.4f} ms")
print(f"Speedup: {original_time / replaced_time:.2f}x")

# Create config for model model (multiple layer types)
configs_to_use = TuningConfigs([
    LayerConfig(
        layer_names=["fc1", "fc2"],
        params={
            "num_terms": [1, 2, 3],
            "low_rank": [10, 15, 20, 50, 100, 150],
        },
        separate=False  # Tune linear layers together
    ),
    LayerConfig(
        layer_names=["conv"],
        params={
            "num_terms": [3, 2, 1],
            "low_rank": [150, 100, 50, 20, 15, 10],
        },
        separate=True
    ),
    LayerConfig(
        layer_names=["attention"],
        params={
            "num_random_features": [16, 32],
            "kernel_fn": ["softmax", "relu"],
        },
        separate=True
    )
])

print("==============testing grid search=================")
test_tuner(configs_to_use, GridSearch())
print("==============testing Random Search=================")
# test_tuner(configs_to_use, RandomSearch())
# print("==============testing Bayesian Search=================")
# test_tuner(configs_to_use, BayesianOptimization())