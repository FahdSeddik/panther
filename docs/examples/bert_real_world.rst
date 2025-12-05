Real-World Applications
=======================

This guide showcases how to use Panther's AutoTuner to optimize neural network models.

AutoTuner with Custom Models
-----------------------------

This example demonstrates how to use Panther's AutoTuner to automatically find optimal sketching parameters for your models.

**Setup and Imports**

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   from torch.utils.data import DataLoader, TensorDataset
   
   from panther.tuner.SkAutoTuner import (
       SKAutoTuner,
       LayerConfig,
       TuningConfigs,
       GridSearch,
       RandomSearch,
       BayesianOptimization,
       ModelVisualizer
   )
   
   # Set device
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")

**Model Setup and Visualization**

.. code-block:: python

   # Load pretrained BERT
   tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
   model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
   
   # Visualize model architecture
   ModelVisualizer.print_module_tree(model)
   
   # Example output:
   # BertForMaskedLM
   #   ├── bert.encoder.layer.0.attention.self.query: Linear(768, 768)
   #   ├── bert.encoder.layer.0.attention.self.key: Linear(768, 768)
   #   ├── bert.encoder.layer.0.attention.self.value: Linear(768, 768)
   #   ├── bert.encoder.layer.0.intermediate.dense: Linear(768, 3072)
   #   └── ...

**Dataset Preparation**

.. code-block:: python

   class MaskedTextDataset(Dataset):
       """Dataset for masked language modeling."""
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
           
           # Create labels
           labels = input_ids.clone()
           
           # Find positions eligible for masking
           mask_positions = (~special_tokens_mask).nonzero(as_tuple=True)[0]
           
           # Randomly mask 15% of eligible tokens
           num_to_mask = max(1, int(0.15 * len(mask_positions)))
           mask_indices = np.random.choice(
               mask_positions.tolist(), 
               size=num_to_mask, 
               replace=False
           )
           input_ids[mask_indices] = self.tokenizer.mask_token_id
           
           return {
               "input_ids": input_ids,
               "attention_mask": encoding.attention_mask.squeeze(0),
               "labels": labels
           }
   
   # Create dataset
   texts = [
       "The quick brown fox jumps over the lazy dog.",
       "Machine learning is transforming technology.",
       "Natural language processing enables AI understanding.",
       # Add more texts...
   ] * 100  # Repeat for larger dataset
   
   dataset = MaskedTextDataset(texts, tokenizer)
   dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

**AutoTuner Configuration**

.. code-block:: python

   def accuracy_eval_func(model):
       """Evaluate model accuracy on validation set."""
       model.eval()
       total_loss = 0
       total_samples = 0
       
       with torch.no_grad():
           for batch in dataloader:
               batch = {k: v.to(device) for k, v in batch.items()}
               outputs = model(**batch)
               loss = outputs.loss
               
               batch_size = batch["input_ids"].size(0)
               total_loss += loss.item() * batch_size
               total_samples += batch_size
               
               # Limit batches for faster evaluation during tuning
               if total_samples >= 500:
                   break
       
       avg_loss = total_loss / total_samples
       # Return negative loss as we want to maximize accuracy (minimize loss)
       return -avg_loss
   
   # Configure which layers to tune - specify exact layer names
   tuning_config = TuningConfigs([
       LayerConfig(
           layer_names=["bert.encoder.layer.0.intermediate.dense", 
                        "bert.encoder.layer.1.intermediate.dense"],
           params={
               "num_terms": [1, 2, 4],
               "low_rank": [32, 64, 128],
           },
           separate=False,  # Apply same config to all matching layers
           copy_weights=True
       )
   ])
   
   # Create AutoTuner
   tuner = SKAutoTuner(
       model=model,
       configs=tuning_config,
       accuracy_eval_func=accuracy_eval_func,
       search_algorithm=GridSearch(),
       verbose=True
   )
   
   # Run tuning
   print("Starting AutoTuning...")
   tuner.tune()
   
   # Get best parameters
   best_params = tuner.get_best_params()
   
   print(f"\\nBest configuration:")
   for layer_name, params_info in best_params.items():
       print(f"Layer: {layer_name}, Best Params: {params_info['params']}")
   
   # Apply best parameters to get optimized model
   best_model = tuner.apply_best_params()

**Training the Optimized Model**

.. code-block:: python

   def train_model(model, train_loader, num_epochs=3):
       """Train the optimized BERT model."""
       model.train()
       optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
       
       for epoch in range(num_epochs):
           total_loss = 0
           for batch_idx, batch in enumerate(train_loader):
               batch = {k: v.to(device) for k, v in batch.items()}
               
               optimizer.zero_grad()
               outputs = model(**batch)
               loss = outputs.loss
               loss.backward()
               optimizer.step()
               
               total_loss += loss.item()
               
               if batch_idx % 50 == 0:
                   avg_loss = total_loss / (batch_idx + 1)
                   print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {avg_loss:.4f}")
   
   # Train with optimized configuration
   train_model(best_model, dataloader, num_epochs=3)

**Model Comparison**

.. code-block:: python

   def compare_models(original_model, optimized_model, test_loader):
       """Compare original and optimized BERT models."""
       
       def count_parameters(model):
           return sum(p.numel() for p in model.parameters() if p.requires_grad)
       
       def measure_inference_time(model, num_runs=100):
           model.eval()
           sample_batch = next(iter(test_loader))
           sample_batch = {k: v.to(device) for k, v in sample_batch.items()}
           
           # Warmup
           for _ in range(10):
               with torch.no_grad():
                   _ = model(**sample_batch)
           
           # Measure
           torch.cuda.synchronize() if device.type == 'cuda' else None
           start = time.time()
           
           for _ in range(num_runs):
               with torch.no_grad():
                   _ = model(**sample_batch)
               torch.cuda.synchronize() if device.type == 'cuda' else None
           
           return (time.time() - start) / num_runs
       
       print("\\n=== Model Comparison ===")
       print(f"Original parameters: {count_parameters(original_model):,}")
       print(f"Optimized parameters: {count_parameters(optimized_model):,}")
       print(f"Parameter reduction: "
             f"{(1 - count_parameters(optimized_model)/count_parameters(original_model))*100:.1f}%")
       
       orig_time = measure_inference_time(original_model)
       opt_time = measure_inference_time(optimized_model)
       
       print(f"\\nOriginal inference time: {orig_time*1000:.2f}ms")
       print(f"Optimized inference time: {opt_time*1000:.2f}ms")
       print(f"Speedup: {orig_time/opt_time:.2f}x")

ResNet with Conv2D Sketching
-----------------------------

**Converting Conv2D layers to Sketched versions**

.. code-block:: python

   import torch
   import torch.nn as nn
   import torchvision.models as models
   from panther.nn import SKConv2d
   
   # Load a pretrained ResNet model
   resnet = models.resnet50(pretrained=True)
   
   # Example: Replace a specific conv layer with SKConv2d
   # Using fromTorch class method to convert from existing Conv2d
   original_conv = resnet.layer1[0].conv1
   
   sketched_conv = SKConv2d.fromTorch(
       layer=original_conv,
       num_terms=4,
       low_rank=16
   )
   
   # Replace in the model
   resnet.layer1[0].conv1 = sketched_conv
   
   # Test the model
   resnet = resnet.to(device)
   x = torch.randn(8, 3, 224, 224, device=device)
   output = resnet(x)
   print(f"Output shape: {output.shape}")  # (8, 1000)
   
   # For systematic replacement across all layers, use AutoTuner
   # See the ResNet Sketching example for more details

Attention-Based Models
----------------------

**Transformer with Randomized Attention**

.. code-block:: python

   from panther.nn import RandMultiHeadAttention
   from panther.nn.pawXimpl import sinSRPE
   
   class RandomizedTransformerEncoder(nn.Module):
       """Transformer encoder with randomized multi-head attention."""
       
       def __init__(self, d_model=512, n_heads=8, n_layers=6, 
                    d_ff=2048, dropout=0.1, num_random_features=256):
           super().__init__()
           
           self.layers = nn.ModuleList([
               RandomizedTransformerLayer(
                   d_model, n_heads, d_ff, dropout, num_random_features
               )
               for _ in range(n_layers)
           ])
       
       def forward(self, x, mask=None):
           for layer in self.layers:
               x = layer(x, mask)
           return x
   
   class RandomizedTransformerLayer(nn.Module):
       """Single transformer layer with randomized attention."""
       
       def __init__(self, d_model, n_heads, d_ff, dropout, num_random_features):
           super().__init__()
           
           # Sketched Random Positional Encoding
           spre = sinSRPE(
               num_heads=n_heads,
               perHead_in=d_model // n_heads,
               sines=16,
               num_realizations=256,
               device=device,
               dtype=torch.float32
           )
           
           # Randomized multi-head attention
           self.attention = RandMultiHeadAttention(
               embed_dim=d_model,
               num_heads=n_heads,
               num_random_features=num_random_features,
               kernel_fn="softmax",
               SRPE=spre,
               device=device
           )
           
           # Feed-forward network
           self.ff = nn.Sequential(
               nn.Linear(d_model, d_ff),
               nn.ReLU(),
               nn.Dropout(dropout),
               nn.Linear(d_ff, d_model)
           )
           
           self.norm1 = nn.LayerNorm(d_model)
           self.norm2 = nn.LayerNorm(d_model)
           self.dropout = nn.Dropout(dropout)
       
       def forward(self, x, mask=None):
           # Self-attention with residual
           attn_out, _ = self.attention(x, x, x, attention_mask=mask)
           x = self.norm1(x + self.dropout(attn_out))
           
           # Feed-forward with residual
           ff_out = self.ff(x)
           x = self.norm2(x + self.dropout(ff_out))
           
           return x
   
   # Usage example
   model = RandomizedTransformerEncoder(
       d_model=512,
       n_heads=8,
       n_layers=6,
       num_random_features=256
   ).to(device)
   
   # Process sequences
   batch_size, seq_len = 32, 100
   x = torch.randn(batch_size, seq_len, 512, device=device)
   output = model(x)
   print(f"Output shape: {output.shape}")  # (32, 100, 512)

Further Resources
-----------------

For more examples and detailed tutorials, see:

- :doc:`autotuner_guide` - Comprehensive AutoTuner documentation
- :doc:`performance_benchmarks` - Performance comparisons and benchmarks
- :doc:`../benchmarks` - Full benchmark results with visualizations
