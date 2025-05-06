import os
import time
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights
import tarfile
from pathlib import Path
import requests
from torch import nn, optim
from torchvision import datasets, transforms, models

# Import components
from panther.utils.SkAutoTuner import *

# Setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

##################################### HELPERS #######################################

def dump_tensor_info(tensor, name="Tensor"):
    """Print details about a tensor"""
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    print(f"  - Values: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")
    print(f"  - First few values: {tensor.flatten()[:5]}")

def measure_time(func, *args, n_runs=20, warmup=5):
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

def calculate_accuracy(outputs, labels):
    """Calculate top-1 and top-5 accuracy"""
    _, preds = outputs.topk(5, 1, True, True)
    preds = preds.t()
    correct = preds.eq(labels.view(1, -1).expand_as(preds))
    top1 = correct[:1].reshape(-1).float().sum(0, keepdim=True).item() / labels.size(0)
    top5 = correct[:5].reshape(-1).float().sum(0, keepdim=True).item() / labels.size(0)
    return top1, top5

def evaluate_model(model, dataloader):
    """Evaluate model accuracy on a dataset"""
    model.eval()
    total_top1 = 0
    total_top5 = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Calculate accuracy
            batch_size = inputs.size(0)
            top1, top5 = calculate_accuracy(outputs, labels)
            
            # Accumulate statistics
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size
            total_samples += batch_size
    
    return total_top1 / total_samples, total_top5 / total_samples

def accuracy_eval_func(model, val_loader, orig_model=None):
    """
    Real-world model evaluation function for accuracy
    
    Args:
        model: The model to evaluate
        val_loader: Validation data loader
        orig_model: Original model for reference (if None, only accuracy is considered)
    Returns:
        A score between 0 and 1 where higher is better
    """
    # Get accuracy
    model.eval()
    top1_acc, top5_acc = evaluate_model(model, val_loader)
    
    # If no original model, just return accuracy
    if orig_model is None:
        return top1_acc
    
    # Get original model accuracy for reference
    orig_model.eval()
    orig_top1_acc, _ = evaluate_model(orig_model, val_loader)
    
    score = (top1_acc - orig_top1_acc) # since they are < 1 both so the score would be greater than or eqal -1 and less than or equal 1
    
    print("running the accuracy validation function")
    print(f"Top-1 Accuracy: {top1_acc:.4f} (original: {orig_top1_acc:.4f}, diff: {score:.4f})")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    print(f"Final score: {score:.4f}")
    
    return score

def get_data():
    """Download and prepare dataset"""
    print("Preparing dataset...")

    dataset_path = "/kaggle/working/processed_imagenet"
    transform = ResNet50_Weights.IMAGENET1K_V1.transforms()

    single_class_dataset  = datasets.ImageFolder(root=dataset_path, transform=transform)
    
    # Download and load Dataset test set
    val_loader = DataLoader(single_class_dataset , batch_size=32, shuffle=False, num_workers=2)
    
    # Small batch for memory testing
    memory_batch_size = len(single_class_dataset)
    memory_batch = torch.stack([single_class_dataset [i][0] for i in range(memory_batch_size)]).to(device)
    
    print(f"Dataset validation set: {len(single_class_dataset )} samples")
    
    return val_loader, memory_batch

def test_specific_layers(model_name="resnet18"):
    """Test SKAutoTuner on specific layers of a model, with custom tuning strategy"""
    
    # Load pre-trained CNN
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    
    # Create a copy of the model for reference
    orig_model = copy.deepcopy(model)
    
    # Get real validation data
    val_loader, memory_batch = get_data()
    
    print("\n===== Original Model Structure =====")
    ModelVisualizer.print_module_tree(model)

    # Create an evaluation function for the model
    def acc_eval_func(model):
        return accuracy_eval_func(model, val_loader, orig_model)
    
    # Create a separate speed evaluation function
    def speed_eval_func(model):
        # Measure inference throughput
        batch, _ = next(iter(val_loader))
        batch = batch.to(device)
        
        def infer(model, x):
            with torch.no_grad():
                return model(x)
        
        # Higher is better (inverse of time)
        throughput = 1.0 / measure_time(infer, model, batch, n_runs=20)
        return throughput
    

    # Get baseline accuracy on Dataset
    print("\nBaseline CNN accuracy on Dataset:")
    baseline_top1, baseline_top5 = evaluate_model(model, val_loader)
    print(f"Top-1 accuracy: {baseline_top1:.4f}")
    print(f"Top-5 accuracy: {baseline_top5:.4f}")
    print(f"Original model memory usage: {measure_memory(orig_model, memory_batch):.2f}MB")
    print(f"Original model speed: {speed_eval_func(orig_model):.2f} samples/sec")
    
    # Strategy 1: Tuning specific network blocks
    print("\n===== Strategy 1: Tuning specific network blocks =====")
    
    # Create configs to tune only layer3 and layer4 blocks (higher layers)
    # This is typical in practice to preserve feature quality in early layers
    configs_strategy1 = TuningConfigs([
        LayerConfig(
            # Use pattern matching to select conv layers in layer3 and layer4
            layer_names={"pattern": "layer[3-4].*conv"},
            params={
                "num_terms": [1, 2, 3],
                "low_rank": [32, 64, 128],
            },
            separate=False  # Tune these layers as a group
        ),
    ])
    
    # Calculate accuracy threshold
    accuracy_threshold = -0.2 # which means 80% of the original accuracy
    print(f"Setting accuracy threshold to {accuracy_threshold:.4f}")
    
    # Create tuner with both accuracy and optimization functions
    tuner_strategy1 = SKAutoTuner(
        model=copy.deepcopy(model),
        configs=configs_strategy1,
        accuracy_eval_func=acc_eval_func,
        search_algorithm=GridSearch(),
        verbose=True,
        accuracy_threshold=accuracy_threshold,  # Set minimum acceptable accuracy
        optmization_eval_func=speed_eval_func   # Optimize for speed after meeting accuracy threshold
    )
    
    # Run tuning
    print("\nRunning block-specific tuning...")
    best_params = tuner_strategy1.tune()
    print(f"Best parameters: {best_params}")
    
    # Apply best parameters
    tuned_model_strategy1 = tuner_strategy1.apply_best_params()
    
    print("\n===== Tuned Model Structure (Strategy 1) =====")
    ModelVisualizer.print_module_tree(tuned_model_strategy1)
    
    # Test the tuned model
    print("\nEvaluating block-tuned model:")
    final_score = accuracy_eval_func(tuned_model_strategy1, val_loader, orig_model)
    print(f"accuracy score: {final_score:.4f}")
    print(f"speed score: {speed_eval_func(tuned_model_strategy1):.2f} samples/sec")
    print(f"memory usage: {measure_memory(tuned_model_strategy1, memory_batch):.2f}MB")
    
    # Strategy 2: Layer-specific parameter tuning
    print("\n===== Strategy 2: Layer-specific parameter tuning =====")
    
    # Create more granular configs for each type of layer
    configs_strategy2 = TuningConfigs([
        # Tune 3x3 convolutions
        LayerConfig(
            layer_names={"pattern": "layer.*conv1"},  # 3x3 convs in bottleneck blocks
            params={
                "num_terms": [2, 3],
                "low_rank": [32, 64, 96],
            },
            separate=True  # Tune each layer separately
        ),
        # Tune 1x1 convolutions with different parameters
        LayerConfig(
            layer_names={"pattern": "layer.*conv2"},  # 1x1 convs in bottleneck blocks
            params={
                "num_terms": [1, 2],
                "low_rank": [32, 48],
            },
            separate=True
        ),
        # Tune downsample convolutions
        LayerConfig(
            layer_names={"pattern": "layer.*downsample.0"},
            params={
                "num_terms": [1],
                "low_rank": [32, 64],
            },
            separate=True
        ),
    ])
    
    # Create tuner with proper accuracy threshold
    tuner_strategy2 = SKAutoTuner(
        model=copy.deepcopy(model),
        configs=configs_strategy2,
        accuracy_eval_func=acc_eval_func,
        accuracy_threshold=accuracy_threshold,
        search_algorithm=RandomSearch(max_trials=10),  # Use random search instead of grid search
        verbose=True,
        optmization_eval_func=speed_eval_func  # Still optimize for speed when accuracy is acceptable
    )
    
    # Run tuning
    print("\nRunning layer-specific tuning...")
    best_params = tuner_strategy2.tune()
    print(f"Best parameters: {best_params}")
    
    # Apply best parameters
    tuned_model_strategy2 = tuner_strategy2.apply_best_params()
    
    print("\n===== Tuned Model Structure (Strategy 2) =====")
    ModelVisualizer.print_module_tree(tuned_model_strategy2)
    
    # Test the tuned model
    print("\nEvaluating layer-specific tuned model:")
    final_score = accuracy_eval_func(tuned_model_strategy2, val_loader, orig_model)
    print(f"Final score: {final_score:.4f}")
    print(f"speed score: {speed_eval_func(tuned_model_strategy2):.2f} samples/sec")
    print(f"memory usage: {measure_memory(tuned_model_strategy2, memory_batch):.2f}MB")
    
    # Strategy 3: Memory-constrained tuning
    print("\n===== Strategy 3: Memory-constrained tuning =====")
    
    # Create configs focused on memory reduction for the largest layers
    configs_strategy3 = TuningConfigs([
        LayerConfig(
            # Find the layers with the most parameters
            layer_names={"pattern": "layer4.*", "type": "Conv2d"},
            params={
                "num_terms": [2],
                "low_rank": [16, 32, 48, 64],  # Lower rank = lower memory
            },
            separate=False
        ),
        LayerConfig(
            layer_names={"pattern": "layer3.*", "type": "Conv2d"},
            params={
                "num_terms": [2],
                "low_rank": [16, 32, 48],
            },
            separate=False
        ),
    ])
    
    # Create a memory optimization function
    def memory_optimization_func(model):
        mem_usage = measure_memory(model, memory_batch)
        orig_mem = measure_memory(orig_model, memory_batch)
        
        score = (orig_mem - mem_usage) / max(orig_mem, 1e-8)
        
        print(f"Memory optimization: {mem_usage:.2f}MB (original: {orig_mem:.2f}MB, reduction: {score:.2f}x)")
        return score
    
    # Create memory-aware tuner
    tuner_strategy3 = SKAutoTuner(
        model=copy.deepcopy(model),
        configs=configs_strategy3,
        accuracy_eval_func=acc_eval_func,
        accuracy_threshold=accuracy_threshold,
        search_algorithm=GridSearch(),
        verbose=True,
        optmization_eval_func=memory_optimization_func  # Optimize for memory specifically
    )
    
    # Run tuning
    print("\nRunning memory-constrained tuning...")
    best_params = tuner_strategy3.tune()
    print(f"Best parameters: {best_params}")
    
    # Apply best parameters
    tuned_model_strategy3 = tuner_strategy3.apply_best_params()
    
    print("\n===== Tuned Model Structure (Strategy 3) =====")
    ModelVisualizer.print_module_tree(tuned_model_strategy3)
    
    # Test the tuned model
    print("\nEvaluating memory-optimized model:")
    print(f"Final score: {accuracy_eval_func(tuned_model_strategy3, val_loader, orig_model):.4f}")
    print(f"speed score: {speed_eval_func(tuned_model_strategy3):.2f} samples/sec")
    print(f"memory usage: {measure_memory(tuned_model_strategy3, memory_batch):.2f}MB")

if __name__ == "__main__":
    import copy  # Used for deep copying models
    
    # Run the full test with multiple strategies
    print("\nRunning full test with multiple tuning strategies...")
    test_specific_layers()
    
    print("\nAll tests completed.")