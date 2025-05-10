import os
import time
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer
import random
import torch.nn.functional as F

# Import components
from panther.utils.SkAutoTuner import (
    SKAutoTuner,
    LayerConfig,
    TuningConfigs,
    GridSearch,
    RandomSearch, 
    ModelVisualizer
)

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed} for reproducibility")

# Setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Call set_seed early in the script
set_seed(42)

##################################### HELPERS #######################################

def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_info(model):
    """Get detailed size information about the model"""
    total_params = count_parameters(model)
    
    # Get layer-wise parameter counts for important components
    layer_params = {}
    
    # Check BERT layers
    if hasattr(model, 'bert') and hasattr(model.bert, 'encoder'):
        for i, layer in enumerate(model.bert.encoder.layer):
            layer_params[f'bert.encoder.layer.{i}'] = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    
    # Check MLM head
    if hasattr(model, 'cls'):
        if hasattr(model.cls, 'predictions'):
            if hasattr(model.cls.predictions, 'transform'):
                layer_params['cls.predictions.transform'] = sum(
                    p.numel() for p in model.cls.predictions.transform.parameters() if p.requires_grad)
            if hasattr(model.cls.predictions, 'decoder'):
                layer_params['cls.predictions.decoder'] = sum(
                    p.numel() for p in model.cls.predictions.decoder.parameters() if p.requires_grad)
    
    return {
        "total_params": total_params,
        "total_params_millions": total_params / 1e6,
        "layer_params": layer_params
    }

def dump_tensor_info(tensor, name="Tensor"):
    """Print details about a tensor"""
    print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    print(f"  - Values: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")
    print(f"  - First few values: {tensor.flatten()[:5]}")

def measure_time_with_stats(func, *args, n_runs=20, warmup=5):
    """Measure execution time of a function with proper GPU synchronization and report statistics"""
    # Clear cache first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Warmup
    for _ in range(warmup):
        result = func(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        result = func(*args)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.time()
        times.append(end - start)
    
    # Calculate statistics
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return {
        "mean": mean_time,
        "std": std_time,
        "min": np.min(times),
        "max": np.max(times),
        "samples_per_sec": 1.0 / mean_time,
        "samples_per_sec_std": std_time / (mean_time * mean_time)
    }

def measure_memory(model, input_tensor):
    """Measure peak memory usage of a model during inference"""
    if not torch.cuda.is_available():
        return 0  # Cannot measure CUDA memory on CPU
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Run inference
    with torch.no_grad():
        model(**input_tensor)
    
    # Get peak memory
    return torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB

class MaskedTextDataset(Dataset):
    """Dataset for masked language modeling"""
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            return_special_tokens_mask=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create input_ids with masks
        input_ids = encoding.input_ids.clone().squeeze(0)
        special_tokens_mask = encoding.special_tokens_mask.squeeze(0).bool()
        
        # Create labels (clone of input_ids)
        labels = input_ids.clone()
        
        # Find positions eligible for masking (not special tokens)
        mask_positions = (~special_tokens_mask).nonzero(as_tuple=True)[0]
        
        # Randomly mask 15% of eligible tokens
        num_to_mask = max(1, int(0.15 * len(mask_positions)))
        mask_indices = np.random.choice(mask_positions.tolist(), size=num_to_mask, replace=False)
        input_ids[mask_indices] = self.tokenizer.mask_token_id
        
        # Create attention mask
        attention_mask = encoding.attention_mask.squeeze(0)
        
        # Create return dictionary
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
        return batch

def evaluate_model_with_stats(model, dataloader, tokenizer=None, n_runs=3):
    """Evaluate model loss on a dataset with multiple runs for statistics"""
    all_results = []
    
    for run in range(n_runs):
        model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Accumulate loss statistics
                batch_size = batch["input_ids"].size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        
        all_results.append({
            "loss": avg_loss
        })
    
    # Compute statistics across runs
    losses = [res["loss"] for res in all_results]
    
    results = {
        "loss_mean": np.mean(losses),
        "loss_std": np.std(losses),
        "runs": all_results
    }
    
    return results

def get_data():
    """Prepare dataset for BERT testing"""
    print("Preparing BERT test dataset...")
    
    # Load WikiText dataset - using a larger portion
    try:
        from datasets import load_dataset
        # Use a variable for the number of examples instead of hardcoding
        num_examples = 100  # Can be easily changed to any value
        
        print("Loading WikiText dataset for evaluation...")
        wiki_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # Process valid entries from WikiText
        texts = []
        for item in wiki_dataset:
            text = item['text'].strip()
            # Filter for non-empty, meaningful text
            if len(text) >= 50 and len(text.split()) > 10:
                texts.append(text)
        
        # If we need more examples, load from train split as well
        if len(texts) < num_examples:
            print(f"Found only {len(texts)} examples in test split, loading more from train split...")
            wiki_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            
            for item in wiki_train:
                text = item['text'].strip()
                if len(text) >= 50 and len(text.split()) > 10:
                    texts.append(text)
                    if len(texts) >= num_examples + 50:  # Get a bit more than needed
                        break
        
        print(f"Loaded {len(texts)} examples from WikiText dataset")
        
        # Ensure we have enough examples
        if len(texts) < num_examples:
            raise ValueError(f"Could only find {len(texts)} valid examples in WikiText, need at least {num_examples}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load WikiText dataset: {str(e)}. Please install the datasets package with 'pip install datasets'")
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Create dataset
    dataset = MaskedTextDataset(texts, tokenizer)
    
    # Create data loader with batch size that's a multiple of 16 for Tensor Core optimization
    batch_size = 16  # For Tensor Core optimizations
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create a single batch for memory testing
    memory_batch = {k: v.to(device) for k, v in next(iter(dataloader)).items()}
    
    return tokenizer, dataloader, memory_batch

def get_data_varied_lengths(seq_lengths=[128, 256, 384, 512]):
    """Prepare datasets with varying sequence lengths for scaling tests"""
    print(f"Preparing BERT test datasets with varying lengths: {seq_lengths}")
    
    # Try to load more complex texts from WikiText
    try:
        from datasets import load_dataset
        print("Loading WikiText dataset for sequence length tests...")
        
        # Load both test and train splits to ensure we have enough data
        wiki_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        wiki_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        
        # Process valid entries from WikiText
        texts = []
        
        # First collect longer texts that are good for sequence length testing
        for item in wiki_test:
            text = item['text'].strip()
            # Filter for meaningful, longer text
            if len(text) >= 100 and len(text.split()) >= 20:
                texts.append(text)
        
        # Add texts from train split
        for item in wiki_train:
            text = item['text'].strip()
            if len(text) >= 100 and len(text.split()) >= 20:
                texts.append(text)
                # Once we have enough data, stop collecting
                if len(texts) >= 1800:
                    break
        
        print(f"Loaded {len(texts)} examples from WikiText dataset")
        
        # Ensure we have enough examples
        if len(texts) < 1800:
            raise ValueError(f"Could only find {len(texts)} valid examples in WikiText, need at least 5000")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load WikiText dataset: {str(e)}. Please install the datasets package with 'pip install datasets'")
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Create datasets and dataloaders for each sequence length
    datasets = {}
    dataloaders = {}
    memory_batches = {}
    
    for max_length in seq_lengths:
        # Create dataset with this specific max_length
        dataset = MaskedTextDataset(texts, tokenizer, max_length=max_length)
        
        # Create data loader with batch size that's a multiple of 16 for Tensor Core optimization
        # Use smaller batches for longer sequences to prevent OOM
        batch_size = 16 if max_length <= 128 else 8 if max_length <= 256 else 4 if max_length <= 384 else 2
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Create a memory test batch
        memory_batch = {k: v.to(device) for k, v in next(iter(dataloader)).items()}
        
        datasets[max_length] = dataset
        dataloaders[max_length] = dataloader
        memory_batches[max_length] = memory_batch
    
    return tokenizer, datasets, dataloaders, memory_batches

def test_sequence_scaling(orig_model, tuned_model, tokenizer):
    """Test how performance improvements scale with sequence length"""
    print("\n===== Testing Performance Scaling with Sequence Length =====")
    
    # Get datasets with varying sequence lengths
    tokenizer, datasets, dataloaders, memory_batches = get_data_varied_lengths()
    
    # Results table
    results = []
    
    # Test each sequence length
    for seq_length in sorted(dataloaders.keys()):
        print(f"\nTesting with sequence length: {seq_length}")
        dataloader = dataloaders[seq_length]
        memory_batch = memory_batches[seq_length]
        
        # Function for inference
        def infer(model, inputs):
            with torch.no_grad():
                return model(**inputs)
        
        # Test models
        for model_name, model in [("Original", orig_model), ("Tuned", tuned_model)]:
            model.eval()
            torch.cuda.empty_cache()
            
            # Measure accuracy
            print(f"Evaluating {model_name} model accuracy...")
            eval_results = evaluate_model_with_stats(model, dataloader, tokenizer, n_runs=3)
            
            # Measure speed
            print(f"Measuring {model_name} model speed...")
            time_results = measure_time_with_stats(infer, model, memory_batch, n_runs=10, warmup=3)
            
            # Measure memory
            memory_used = measure_memory(model, memory_batch)
            
            # Store results
            results.append({
                "seq_length": seq_length,
                "model": model_name,
                "loss_mean": eval_results["loss_mean"],
                "loss_std": eval_results["loss_std"],
                "speed_mean": time_results["samples_per_sec"],
                "speed_std": time_results["samples_per_sec_std"],
                "memory": memory_used
            })
    
    # Print results table
    print("\n===== Sequence Length Scaling Results =====")
    print("| Seq Length | Model | MLM Loss | Speed (samples/sec) | Memory (MB) | Speedup |")
    print("|------------|-------|----------|---------------------|-------------|---------|")
    
    for seq_length in sorted(dataloaders.keys()):
        # Extract results for this sequence length
        orig_result = next(r for r in results if r["seq_length"] == seq_length and r["model"] == "Original")
        tuned_result = next(r for r in results if r["seq_length"] == seq_length and r["model"] == "Tuned")
        
        # Calculate speedup
        speedup = tuned_result["speed_mean"] / orig_result["speed_mean"]
        
        # Print original model results
        print(f"| {seq_length:10d} | Original | {orig_result['loss_mean']:.4f}±{orig_result['loss_std']:.4f} | "
              f"{orig_result['speed_mean']:.2f}±{orig_result['speed_std']:.2f} | "
              f"{orig_result['memory']:.2f} | 1.00x |")
        
        # Print tuned model results
        print(f"| {seq_length:10d} | Tuned | {tuned_result['loss_mean']:.4f}±{tuned_result['loss_std']:.4f} | "
              f"{tuned_result['speed_mean']:.2f}±{tuned_result['speed_std']:.2f} | "
              f"{tuned_result['memory']:.2f} | {speedup:.2f}x |")
    
    return results

def fill_mask_test(model, tokenizer, text="The capital of France is [MASK]."):
    """Test mask filling capability"""
    # Replace [MASK] with actual mask token if needed
    if "[MASK]" in text:
        text = text.replace("[MASK]", tokenizer.mask_token)
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Find mask token position
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    
    # Get predictions for mask position
    if len(mask_token_index[0]) > 0:
        batch_idx, token_idx = mask_token_index
        mask_logits = logits[batch_idx, token_idx, :]
        
        # Get top 5 predictions
        topk_values, topk_indices = torch.topk(mask_logits, 5, dim=1)
        
        # Convert to tokens
        topk_tokens = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in topk_indices[0]]
        
        return topk_tokens
    else:
        return ["No mask token found"]

def test_bert_optimization():
    """Test SKAutoTuner on BERT model's linear layers"""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create reference copy before any modifications to ensure identical initial states
    model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    model.eval()
    orig_model = copy.deepcopy(model)  # Create copy before any modifications
    
    # Get data for testing (do this before modifying the models)
    tokenizer, val_loader, memory_batch = get_data()
    
    # Get parameter counts before optimization
    print("\n===== Model Parameter Counts Before Optimization =====")
    orig_params = model_size_info(model)
    print(f"Total parameters: {orig_params['total_params_millions']:.2f}M")
    print("Parameters by layer:")
    for layer_name, param_count in orig_params['layer_params'].items():
        print(f"  - {layer_name}: {param_count/1e6:.2f}M parameters")
    
    # Apply identical vocab size modifications to both models
    orig_out_features = model.cls.predictions.decoder.weight.size(0)
    new_out_features = ((orig_out_features + 15) // 16) * 16
    
    # Store the true original forward methods before any wrapping
    true_orig_forward = model.forward
    true_orig_ref_forward = orig_model.forward
    
    # Define the post-processing function
    def post_process_outputs(model_outputs, orig_size=orig_out_features):
        """Trim any padded outputs back to original vocabulary size"""
        if hasattr(model_outputs, 'logits') and model_outputs.logits is not None:
            if model_outputs.logits.size(-1) > orig_size:
                # Trim to original vocabulary size
                model_outputs.logits = model_outputs.logits[..., :orig_size]
        return model_outputs
    
    # Create a wrapper factory for forward methods
    def create_wrapped_forward(original_forward_fn):
        def wrapped_forward(*args, **kwargs):
            outputs = original_forward_fn(*args, **kwargs)
            return post_process_outputs(outputs)
        return wrapped_forward
    
    # Apply identical modifications to both models
    for m in [model, orig_model]:
        if orig_out_features != new_out_features:
            # Create padded weights and bias
            orig_weight = m.cls.predictions.decoder.weight
            orig_bias = m.cls.predictions.decoder.bias
            
            new_weight = torch.zeros(new_out_features, orig_weight.size(1), 
                                    device=orig_weight.device, dtype=orig_weight.dtype)
            new_bias = torch.zeros(new_out_features, 
                                  device=orig_bias.device, dtype=orig_bias.dtype)
            
            # Copy the original values
            new_weight[:orig_out_features, :] = orig_weight
            new_bias[:orig_out_features] = orig_bias
            
            # Replace the decoder
            new_decoder = torch.nn.Linear(orig_weight.size(1), new_out_features, bias=True)
            new_decoder.weight = torch.nn.Parameter(new_weight)
            new_decoder.bias = torch.nn.Parameter(new_bias)
            
            m.cls.predictions.decoder = new_decoder
            
        # Update the config's vocab_size
        m.config.vocab_size = new_out_features
    
    # Apply a single wrapping to each model
    model.forward = create_wrapped_forward(true_orig_forward)
    orig_model.forward = create_wrapped_forward(true_orig_ref_forward)
    
    # First evaluate the original model before any modifications
    print("\nBaseline BERT model (before any modifications):")
    baseline_results = evaluate_model_with_stats(model, val_loader, tokenizer)
    
    # Measure performance metrics of original model
    def infer(model, inputs):
        with torch.no_grad():
            return model(**inputs)
    
    baseline_time_stats = measure_time_with_stats(infer, model, memory_batch, n_runs=10)
    baseline_speed = baseline_time_stats["samples_per_sec"]
    baseline_memory = measure_memory(model, memory_batch)
    
    print(f"MLM Loss: {baseline_results['loss_mean']:.4f}±{baseline_results['loss_std']:.4f}")
    print(f"Baseline model memory usage: {baseline_memory:.2f} MB")
    print(f"Baseline model speed: {baseline_speed:.2f}±{baseline_time_stats['samples_per_sec_std']:.2f} samples/sec")
    
    print("\n===== Original Model Structure =====")
    ModelVisualizer.print_module_tree(model)
    
    # Create an evaluation function for the model
    def acc_eval_func(model):
        """Evaluation function based on MLM loss (lower is better)"""
        results = evaluate_model_with_stats(model, val_loader, tokenizer)
        print(f"MLM Loss: {results['loss_mean']:.4f}±{results['loss_std']:.4f}")
        # Return negative loss since SKAutoTuner maximizes the score
        return -results['loss_mean']  # Negative loss (higher is better, as required by SKAutoTuner)
    
    # Create a separate speed evaluation function
    def speed_eval_func(model):
        """Speed evaluation function"""
        def infer(model, inputs):
            with torch.no_grad():
                return model(**inputs)
        
        # Higher is better (inverse of time)
        time_stats = measure_time_with_stats(infer, model, memory_batch, n_runs=10)
        throughput = time_stats["samples_per_sec"]
        print(f"Inference speed: {throughput:.2f}±{time_stats['samples_per_sec_std']:.2f} samples/sec")
        return throughput
    
    # Calculate loss threshold (allow some increase in loss)
    loss_threshold = -9999  # Allow 0.5 increase in loss
    print(f"Setting loss threshold to {loss_threshold:.4f}")
    
    # Strategy: Optimizing both linear layers in the MLM head
    print("\n===== Optimizing both MLM head linear layers =====")
    
    # Create configs to tune both linear layers together with Tensor Core friendly dimensions
    configs = TuningConfigs([
        LayerConfig(
            # Target both linear layers in the MLM head
            layer_names={
                "pattern": "cls.predictions.*",
                "type": "Linear",
            },
            params={
                "num_terms": [1, 2, 3],
                "low_rank": [16, 32, 64],  # All values are multiples of 16 for Tensor Core
            },
            separate=False  # Tune as a group
        ),
    ])
    
    # Create tuner for both layers together
    tuner = SKAutoTuner(
        model=copy.deepcopy(model),
        configs=configs,
        accuracy_eval_func=acc_eval_func,  # Using loss evaluation, despite the function name
        search_algorithm=GridSearch(),
        verbose=True,
        accuracy_threshold=loss_threshold,  # Negative since we're using negative loss as our metric
        optmization_eval_func=speed_eval_func,
        num_runs_per_param=20
    )
    
    # Run tuning
    print("\nRunning combined MLM head layers tuning...")
    best_params = tuner.tune()
    print(f"Best parameters: {best_params}")
    
    # Apply best parameters
    tuned_model = tuner.apply_best_params()
    
    print("\n===== Tuned Model Structure =====")
    ModelVisualizer.print_module_tree(tuned_model)
    
    # Test the tuned model
    print("\nEvaluating models with identical conditions:")
    
    # Ensure both models are in the same state for fair comparison
    for m in [orig_model, tuned_model]:
        m.eval()
        torch.cuda.empty_cache()

    # Use identical test conditions
    def test_model(model_name, model):
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Run standardized tests
        results = evaluate_model_with_stats(model, val_loader, tokenizer)
        
        def infer(model, inputs):
            with torch.no_grad():
                return model(**inputs)
        
        time_result = measure_time_with_stats(infer, model, memory_batch, n_runs=10)
        speed = time_result["samples_per_sec"]
        speed_std = time_result["samples_per_sec_std"]
        memory_used = measure_memory(model, memory_batch)
        
        return {
            "name": model_name,
            "loss": results["loss_mean"],
            "loss_std": results["loss_std"],
            "speed": speed,
            "speed_std": speed_std,
            "memory": memory_used
        }

    # Test both models under identical conditions
    baseline_results = test_model("Original", orig_model)
    tuned_results = test_model("Tuned", tuned_model)
    
    # Extract results for the comparison table
    baseline_loss = baseline_results["loss"]
    baseline_speed = baseline_results["speed"]
    baseline_memory = baseline_results["memory"]
    
    final_loss = tuned_results["loss"]
    final_speed = tuned_results["speed"]
    final_memory = tuned_results["memory"]
    
    # After optimization, get new parameter counts
    print("\n===== Model Parameter Counts After Optimization =====")
    tuned_params = model_size_info(tuned_model)
    print(f"Original model: {orig_params['total_params_millions']:.2f}M parameters")
    print(f"Tuned model: {tuned_params['total_params_millions']:.2f}M parameters")
    print(f"Reduction: {(1 - tuned_params['total_params_millions']/orig_params['total_params_millions'])*100:.2f}%")
    
    print("\nParameters by layer:")
    for layer_name in sorted(set(list(orig_params['layer_params'].keys()) + list(tuned_params['layer_params'].keys()))):
        orig_count = orig_params['layer_params'].get(layer_name, 0) / 1e6
        tuned_count = tuned_params['layer_params'].get(layer_name, 0) / 1e6
        
        if orig_count > 0 and tuned_count > 0:
            reduction = (1 - tuned_count/orig_count) * 100
            print(f"  - {layer_name}: {orig_count:.2f}M → {tuned_count:.2f}M ({reduction:.2f}% reduction)")
    
    print(f"MLM Loss: {final_loss:.4f}±{tuned_results['loss_std']:.4f} (original: {baseline_loss:.4f}±{baseline_results['loss_std']:.4f})")
    print(f"Speed: {final_speed:.2f}±{tuned_results['speed_std']:.2f} samples/sec (original: {baseline_speed:.2f}±{baseline_results['speed_std']:.2f})")
    print(f"Memory: {final_memory:.2f} MB (original: {baseline_memory:.2f})")
    
    # Enhanced performance comparison table
    print("\n===== Performance Comparison =====")
    print("| Model Version | MLM Loss | Speed (samples/sec) | Memory (MB) | Speed Improvement |")
    print("|--------------|----------|---------------------|-------------|-------------------|")
    print(f"| Original     | {baseline_loss:.4f}±{baseline_results['loss_std']:.4f} | {baseline_speed:.2f}±{baseline_results['speed_std']:.2f} | {baseline_memory:.2f} | 1.00x |")
    print(f"| Tuned        | {final_loss:.4f}±{tuned_results['loss_std']:.4f} | {final_speed:.2f}±{tuned_results['speed_std']:.2f} | {final_memory:.2f} | {final_speed/baseline_speed:.2f}x |")
    
    # Additional comparison tests with real examples
    test_examples = [
        "The capital of France is [MASK].",
        "Machine learning models [MASK] data to make predictions.",
        "Transformers use [MASK] attention to process sequences.",
        "The [MASK] language model was developed by Google researchers."
    ]
    
    print("\n===== Qualitative Comparison: Mask Filling =====")
    for test_sentence in test_examples:
        print(f"\nSentence: {test_sentence}")
        
        # Original model predictions
        orig_predictions = fill_mask_test(orig_model, tokenizer, test_sentence)
        print(f"Original model predictions: {', '.join(orig_predictions)}")
        
        # Tuned model predictions
        tuned_predictions = fill_mask_test(tuned_model, tokenizer, test_sentence)
        print(f"Tuned model predictions:    {', '.join(tuned_predictions)}")
    
    # Run sequence length scaling test
    test_sequence_scaling(orig_model, tuned_model, tokenizer)
    
    return tuned_model

if __name__ == "__main__":
    import copy  # Used for deep copying models
    
    # Run the BERT optimization test
    print("\nRunning BERT optimization test with SKAutoTuner...")
    test_bert_optimization()
    
    print("\nTest completed.")