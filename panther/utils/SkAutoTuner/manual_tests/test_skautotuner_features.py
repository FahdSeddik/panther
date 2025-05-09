import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split

# Import components from the SKAutoTuner module
from panther.utils.SkAutoTuner import (
    SKAutoTuner, 
    LayerConfig, 
    TuningConfigs, 
    GridSearch,
    RandomSearch, 
    ModelVisualizer
)
from panther.nn import SKLinear, SKLinear_triton, RandMultiHeadAttention

# Setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

##################################### CUSTOM MODEL #######################################

class ConvBlock(nn.Module):
    """A simple convolutional block with batch normalization and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class FCBlock(nn.Module):
    """A fully connected block with dropout and ReLU"""
    def __init__(self, in_features, out_features, dropout=0.2):
        super(FCBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.dropout(self.linear(x)))

class AttentionBlock(nn.Module):
    """An attention block using MultiheadAttention"""
    def __init__(self, embed_dim, num_heads):
        super(AttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, x):
        # Layer normalization and attention
        normalized = self.norm1(x)
        attention_output, _ = self.mha(normalized, normalized, normalized)
        x = x + attention_output
        
        # Layer normalization and feed-forward network
        normalized = self.norm2(x)
        ffn_output = self.ffn(normalized)
        
        return x + ffn_output

class MixedModel(nn.Module):
    """A model with mixed layer types (Conv2D, Linear, MultiheadAttention) in a nested structure"""
    def __init__(self, num_classes=10):
        super(MixedModel, self).__init__()
        
        # Convolutional feature extractor for images
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 32),                # 32x32x32
            nn.MaxPool2d(2, 2),              # 32x16x16
            ConvBlock(32, 64),               # 64x16x16
            nn.MaxPool2d(2, 2),              # 64x8x8
            ConvBlock(64, 128),              # 128x8x8
            nn.MaxPool2d(2, 2),              # 128x4x4
        )
        
        # Fully connected classifier
        self.classifier = nn.Sequential(
            FCBlock(128 * 4 * 4, 512),
            FCBlock(512, 256),
            nn.Linear(256, num_classes)
        )
        
        # Attention branch (processes flattened features from a different perspective)
        self.attention_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.LayerNorm(256)
        )
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(256, 8),
            AttentionBlock(256, 8)
        ])
        self.attention_output = nn.Linear(256, num_classes)
        
        # Final combination
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Process through classifier branch
        flattened = features.view(features.size(0), -1)
        classifier_output = self.classifier(flattened)
        
        # Process through attention branch
        attention_input = self.attention_branch(features)
        
        # Apply attention blocks sequentially
        for block in self.attention_blocks:
            attention_input = block(attention_input)
        
        attention_output = self.attention_output(attention_input)
        
        # Combined output
        combined_output = self.alpha * classifier_output + (1 - self.alpha) * attention_output
        
        return combined_output

##################################### TEST HELPERS #######################################

def generate_dummy_data(batch_size=32, input_shape=(3, 32, 32), num_classes=10):
    """Generate dummy data for testing"""
    inputs = torch.randn(batch_size, *input_shape)
    labels = torch.randint(0, num_classes, (batch_size,))
    return inputs, labels

def create_data_loaders(batch_size=32, num_samples=1000, input_shape=(3, 32, 32), num_classes=10):
    """Create data loaders for training and validation"""
    # Generate random data
    inputs, labels = generate_dummy_data(num_samples, input_shape, num_classes)
    dataset = TensorDataset(inputs, labels)
    
    # Split into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def measure_accuracy(model, data_loader, device):
    """Measure the accuracy of a model on a dataset"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def measure_time(func, *args, n_runs=10, warmup=2):
    """Measure execution time of a function"""
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    # Timed runs
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(n_runs):
        func(*args)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    return (end - start) / n_runs

def measure_memory(model, input_tensor):
    """Measure peak memory usage of a model during inference"""
    if not torch.cuda.is_available():
        return 0  # Cannot measure CUDA memory on CPU
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Run inference
    with torch.no_grad():
        model(input_tensor)
    
    # Get peak memory
    return torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB

##################################### TEST FUNCTIONS #######################################

def test_skautotuner_visualization():
    """Test SKAutoTuner's visualization functionality"""
    print("\n===== Testing SKAutoTuner Visualization =====")
    
    # Create data and model
    train_loader, val_loader = create_data_loaders()
    sample_batch, _ = next(iter(val_loader))
    sample_batch = sample_batch.to(device)
    
    model = MixedModel().to(device)

    # Create evaluation functions
    def accuracy_eval_func(model):
        return measure_accuracy(model, val_loader, device)
    
    def speed_eval_func(model):
        def inference(model, x):
            with torch.no_grad():
                return model(x)
        
        return 1.0 / measure_time(inference, model, sample_batch)
    
    # Create tuning configuration focusing on linear layers in the classifier
    config = TuningConfigs([
        LayerConfig(
            layer_names={"pattern": "classifier.*linear"},
            params={
                "num_terms": [1, 2],
                "low_rank": [8, 16],
            },
            separate=True
        )
    ])
    
    # Create tuner with limited search space for quick testing
    tuner = SKAutoTuner(
        model=copy.deepcopy(model),
        configs=config,
        accuracy_eval_func=accuracy_eval_func,
        search_algorithm=GridSearch(),
        verbose=True
    )
    
    # Run a simple tuning process
    print("Running tuning...")
    tuner.tune()
    
    # Test visualization function (save to file and show)
    save_path = "tuning_visualization.png"
    print(f"Generating visualization at {save_path}...")
    
    # Test with and without saving
    tuner.visualize_tuning_results(save_path=save_path, show_plot=False)
    tuner.visualize_tuning_results(show_plot=False)
    
    # Test with custom figure size
    tuner.visualize_tuning_results(figsize=(10, 8), show_plot=False)
    
    if os.path.exists(save_path):
        print(f"Visualization saved successfully to {save_path}")
    else:
        print("WARNING: Visualization file not created")
    
    return tuner

def test_skautotuner_results():
    """Test SKAutoTuner's result retrieval functions"""
    print("\n===== Testing SKAutoTuner Results Retrieval =====")
    
    # Create data and model
    train_loader, val_loader = create_data_loaders()
    
    model = MixedModel().to(device)
    
    # Create evaluation function
    def accuracy_eval_func(model):
        return measure_accuracy(model, val_loader, device)
    
    # Create tuning configuration with different layer types
    config = TuningConfigs([
        # Test Conv2d layers
        LayerConfig(
            layer_names={"pattern": "feature_extractor.*conv"},
            params={
                "num_terms": [1, 2],
                "low_rank": [8, 16],
            },
            separate=False
        ),
        # Test Linear layers
        LayerConfig(
            layer_names={"pattern": "classifier.*linear"},
            params={
                "num_terms": [1, 2],
                "low_rank": [8, 16],
            },
            separate=True
        ),
        # Test MultiheadAttention layers
        LayerConfig(
            layer_names={"pattern": "attention_blocks.*mha", "type" : "MultiheadAttention"},
            params={
                "num_random_features": [16, 32],
                "kernel_fn": ["softmax", "relu"],
            },
            separate=True
        )
    ])
    
    # Create tuner
    tuner = SKAutoTuner(
        model=copy.deepcopy(model),
        configs=config,
        accuracy_eval_func=accuracy_eval_func,
        search_algorithm=RandomSearch(max_trials=2),  # Using a small number of trials for quick testing
        verbose=True
    )
    
    # Run tuning
    print("Running tuning...")
    best_params = tuner.tune()
    
    # Test get_best_params
    print("\nTesting get_best_params:")
    retrieved_best_params = tuner.get_best_params()
    
    # Check if the returned parameters match the ones from tune()
    for layer_name in best_params:
        if layer_name in retrieved_best_params:
            print(f"Layer: {layer_name}")
            print(f"  Parameters consistent: {best_params[layer_name]['params'] == retrieved_best_params[layer_name]['params']}")
        else:
            print(f"WARNING: Layer {layer_name} not found in retrieved_best_params")
    
    # Test get_results_dataframe
    print("\nTesting get_results_dataframe:")
    results_df = tuner.get_results_dataframe()
    
    # Print summary of results
    print(f"DataFrame shape: {results_df.shape}")
    print("Columns in DataFrame:")
    for col in results_df.columns:
        print(f"  - {col}")
    
    # Check if layer names are in the DataFrame
    layers_in_df = results_df['layer_name'].unique()
    print(f"Unique layer names in results: {len(layers_in_df)}")
    
    # Try to find results for each layer group in the configs
    for config_group in config.configs:
        for layer_name in config_group.layer_names:
            found = any(layer_name in df_layer_name for df_layer_name in layers_in_df)
            if found:
                print(f"Results found for layer: {layer_name}")
            else:
                print(f"WARNING: No results found for layer: {layer_name}")
    
    return tuner, best_params, results_df

def test_skautotuner_apply_best_params():
    """Test SKAutoTuner's apply_best_params function"""
    print("\n===== Testing SKAutoTuner apply_best_params =====")
    
    # Create data and model
    train_loader, val_loader = create_data_loaders()
    sample_batch, _ = next(iter(val_loader))
    sample_batch = sample_batch.to(device)
    
    orig_model = MixedModel().to(device)
    print("visualize the model before doing anything")
    ModelVisualizer.print_module_tree(orig_model)
    
    # Measure original model characteristics
    orig_accuracy = measure_accuracy(orig_model, val_loader, device)
    
    def inference(model, x):
        with torch.no_grad():
            return model(x)
    
    orig_speed = 1.0 / measure_time(inference, orig_model, sample_batch)
    orig_memory = measure_memory(orig_model, sample_batch)
    
    print(f"Original model accuracy: {orig_accuracy:.4f}")
    print(f"Original model speed: {orig_speed:.2f} samples/sec")
    print(f"Original model memory: {orig_memory:.2f} MB")
    
    # Create evaluation function with threshold
    def accuracy_eval_func(model):
        return measure_accuracy(model, val_loader, device)
    
    def speed_eval_func(model):
        return 1.0 / measure_time(inference, model, sample_batch)
    
    # Create tuning configuration focusing on computationally intensive parts
    config = TuningConfigs([
        # Tune the final layers in each branch of the network
        LayerConfig(
            layer_names=[
                "classifier.2",  # Final linear in classifier
                "attention_output"  # Final linear in attention branch
            ],
            params={
                "num_terms": [1, 2, 3],
                "low_rank": [8, 16, 32],
            },
            separate=True
        ),
        # Tune the attention blocks
        LayerConfig(
            layer_names={"pattern": "attention_blocks.*mha", "type" : "MultiheadAttention"},
            params={
                "num_random_features": [16, 32, 64],
                "kernel_fn": ["softmax"],
            },
            separate=False  # Tune as a group
        )
    ])
    
    # Create tuner with accuracy threshold
    accuracy_threshold = orig_accuracy - 0.05  # Allow 5% accuracy drop
    tuner = SKAutoTuner(
        model=copy.deepcopy(orig_model),
        configs=config,
        accuracy_eval_func=accuracy_eval_func,
        accuracy_threshold=accuracy_threshold,
        optmization_eval_func=speed_eval_func,
        search_algorithm=RandomSearch(max_trials=3),  # Using a small number of trials for quick testing
        verbose=True
    )
    
    # Run tuning
    print("Running tuning...")
    best_params = tuner.tune()
    print("Best parameters:")
    for layer, params in best_params.items():
        print(f"  {layer}: {params}")
    
    # Apply best parameters
    print("Applying best parameters...")
    tuned_model = tuner.apply_best_params()
    
    # Print model structure after tuning
    print("\nTuned model structure:")
    ModelVisualizer.print_module_tree(tuned_model)
    
    # Evaluate the tuned model
    tuned_accuracy = measure_accuracy(tuned_model, val_loader, device)
    tuned_speed = 1.0 / measure_time(inference, tuned_model, sample_batch)
    tuned_memory = measure_memory(tuned_model, sample_batch)
    
    print(f"Tuned model accuracy: {tuned_accuracy:.4f} (diff: {tuned_accuracy - orig_accuracy:.4f})")
    print(f"Tuned model speed: {tuned_speed:.2f} samples/sec (diff: {tuned_speed - orig_speed:.2f})")
    print(f"Tuned model memory: {tuned_memory:.2f} MB (diff: {tuned_memory - orig_memory:.2f} MB)")
    
    # Check if the accuracy threshold was respected
    if tuned_accuracy >= accuracy_threshold:
        print("✅ Tuned model meets accuracy threshold")
    else:
        print("❌ Tuned model fails to meet accuracy threshold")
    
    return tuner, tuned_model

def test_skautotuner_combined_scenario():
    """Test a combined real-world scenario with the SKAutoTuner"""
    print("\n===== Testing SKAutoTuner in a Combined Scenario =====")
    
    # Create data and model
    train_loader, val_loader = create_data_loaders(batch_size=64, num_samples=2000)
    sample_batch, _ = next(iter(val_loader))
    sample_batch = sample_batch.to(device)
    
    orig_model = MixedModel().to(device)
    print("\nOriginal model structure:")
    ModelVisualizer.print_module_tree(orig_model)
    
    # Measure original model characteristics
    orig_accuracy = measure_accuracy(orig_model, val_loader, device)
    
    def inference(model, x):
        with torch.no_grad():
            return model(x)
    
    orig_speed = 1.0 / measure_time(inference, orig_model, sample_batch)
    orig_memory = measure_memory(orig_model, sample_batch)
    
    print(f"Original model accuracy: {orig_accuracy:.4f}")
    print(f"Original model speed: {orig_speed:.2f} samples/sec")
    print(f"Original model memory: {orig_memory:.2f} MB")
    
    # Define evaluation functions
    def accuracy_eval_func(model):
        return measure_accuracy(model, val_loader, device)
    
    def speed_eval_func(model):
        return 1.0 / measure_time(inference, model, sample_batch)
    
    def memory_eval_func(model):
        mem_usage = measure_memory(model, sample_batch)
        orig_mem = measure_memory(orig_model, sample_batch)
        
        # Return a score that's higher when memory reduction is greater
        # Normalize by original memory to get a relative improvement
        score = (orig_mem - mem_usage) / max(orig_mem, 1e-8)
        
        print(f"Memory evaluation: {mem_usage:.2f} MB (original: {orig_mem:.2f} MB, reduction: {score:.2f})")
        return score
    
    # Strategy 1: Focus on speed optimization
    print("\n----- Strategy 1: Speed Optimization -----")
    
    config_speed = TuningConfigs([
        # Tune convolutional layers
        LayerConfig(
            layer_names={"pattern": "feature_extractor.*conv"},
            params={
                "num_terms": [1, 2],
                "low_rank": [16, 32],
            },
            separate=False  # Tune as a group
        ),
        # Tune linear layers in classifier
        LayerConfig(
            layer_names={"pattern": "classifier.*linear"},
            params={
                "num_terms": [1, 2],
                "low_rank": [16, 32],
            },
            separate=True  # Tune individually
        )
    ])
    
    # Create tuner for speed
    accuracy_threshold = orig_accuracy - 0.05  # Allow 5% accuracy drop
    tuner_speed = SKAutoTuner(
        model=copy.deepcopy(orig_model),
        configs=config_speed,
        accuracy_eval_func=accuracy_eval_func,
        accuracy_threshold=accuracy_threshold,
        optmization_eval_func=speed_eval_func,
        search_algorithm=RandomSearch(max_trials=2),  # Using a small number of trials for quick testing
        verbose=True
    )
    
    # Run tuning
    print("Running speed-focused tuning...")
    best_params_speed = tuner_speed.tune()
    tuned_model_speed = tuner_speed.apply_best_params()
    
    # Evaluate speed-tuned model
    speed_accuracy = measure_accuracy(tuned_model_speed, val_loader, device)
    speed_speed = 1.0 / measure_time(inference, tuned_model_speed, sample_batch)
    speed_memory = measure_memory(tuned_model_speed, sample_batch)
    
    print(f"Speed-tuned model accuracy: {speed_accuracy:.4f} (diff: {speed_accuracy - orig_accuracy:.4f})")
    print(f"Speed-tuned model speed: {speed_speed:.2f} samples/sec (diff: {speed_speed - orig_speed:.2f})")
    print(f"Speed-tuned model memory: {speed_memory:.2f} MB (diff: {speed_memory - orig_memory:.2f} MB)")
    
    # Strategy 2: Focus on memory optimization
    print("\n----- Strategy 2: Memory Optimization -----")
    
    config_memory = TuningConfigs([
        # Tune attention blocks (memory-intensive)
        LayerConfig(
            layer_names={"pattern": "attention_blocks.*mha", "type" : "MultiheadAttention"},
            params={
                "num_random_features": [16, 32],
                "kernel_fn": ["relu", "softmax"],  # Different kernel functions affect memory usage
            },
            separate=False  # Tune as a group
        ),
        # Tune the attention output layer
        LayerConfig(
            layer_names={"pattern": "attention_output"},
            params={
                "num_terms": [1],
                "low_rank": [8, 16, 32],  # Lower rank = lower memory
            },
            separate=True
        )
    ])
    
    # Create tuner for memory
    tuner_memory = SKAutoTuner(
        model=copy.deepcopy(orig_model),
        configs=config_memory,
        accuracy_eval_func=accuracy_eval_func,
        accuracy_threshold=accuracy_threshold,
        optmization_eval_func=memory_eval_func,
        search_algorithm=GridSearch(),
        verbose=True
    )
    
    # Run tuning
    print("Running memory-focused tuning...")
    best_params_memory = tuner_memory.tune()
    tuned_model_memory = tuner_memory.apply_best_params()
    
    # Evaluate memory-tuned model
    memory_accuracy = measure_accuracy(tuned_model_memory, val_loader, device)
    memory_speed = 1.0 / measure_time(inference, tuned_model_memory, sample_batch)
    memory_memory = measure_memory(tuned_model_memory, sample_batch)
    
    print(f"Memory-tuned model accuracy: {memory_accuracy:.4f} (diff: {memory_accuracy - orig_accuracy:.4f})")
    print(f"Memory-tuned model speed: {memory_speed:.2f} samples/sec (diff: {memory_speed - orig_speed:.2f})")
    print(f"Memory-tuned model memory: {memory_memory:.2f} MB (diff: {memory_memory - orig_memory:.2f} MB)")
    
    # Get and visualize results
    speed_results_df = tuner_speed.get_results_dataframe()
    memory_results_df = tuner_memory.get_results_dataframe()
    
    print("\nSpeed tuning results summary:")
    print(f"Total trials: {len(speed_results_df)}")
    print(f"Average speed score: {speed_results_df['score'].mean():.4f}")
    
    print("\nMemory tuning results summary:")
    print(f"Total trials: {len(memory_results_df)}")
    print(f"Average memory score: {memory_results_df['score'].mean():.4f}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    tuner_speed.visualize_tuning_results(save_path="speed_tuning_results.png", show_plot=False)
    tuner_memory.visualize_tuning_results(save_path="memory_tuning_results.png", show_plot=False)
    
    # Final comparison
    print("\n----- Final Comparison -----")
    print(f"Original model - Accuracy: {orig_accuracy:.4f}, Speed: {orig_speed:.2f} samples/sec, Memory: {orig_memory:.2f} MB")
    print(f"Speed-tuned  - Accuracy: {speed_accuracy:.4f}, Speed: {speed_speed:.2f} samples/sec, Memory: {speed_memory:.2f} MB")
    print(f"Memory-tuned - Accuracy: {memory_accuracy:.4f}, Speed: {memory_speed:.2f} samples/sec, Memory: {memory_memory:.2f} MB")
    
    return tuner_speed, tuner_memory, tuned_model_speed, tuned_model_memory

if __name__ == "__main__":
    import sys
    
    # Default behavior: run all tests
    run_visualization = True
    run_results = True
    run_apply = True
    run_combined = True
    
    print("SKAutoTuner Test Suite")
    print("=====================")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Tests to run: Visualization={run_visualization}, Results={run_results}, Apply={run_apply}, Combined={run_combined}")
    print(f"model visualization:")
    ModelVisualizer.print_module_tree(MixedModel())

    # Run tests
    vis_tuner = None
    results_tuner = None
    apply_tuner = None
    combined_tuners = None
    
    try:
        if run_visualization:
            vis_tuner = test_skautotuner_visualization()
        
        if run_results:
            results_tuner, best_params, results_df = test_skautotuner_results()
        
        if run_apply:
            apply_tuner, tuned_model = test_skautotuner_apply_best_params()
        
        if run_combined:
            combined_tuners = test_skautotuner_combined_scenario()
        
        print("\nAll tests completed successfully!")
    
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()