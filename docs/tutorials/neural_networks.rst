Neural Networks with Sketching
=============================

This tutorial demonstrates how to build complete neural networks using Panther's sketched layers, from simple MLPs to complex architectures.

Building Your First Sketched Network
-------------------------------------

**From Standard to Sketched**

Let's start by converting a standard neural network to use sketched layers:

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   import panther as pr
   from torch.utils.data import DataLoader, TensorDataset
   
   # Standard neural network
   class StandardMLP(nn.Module):
       def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
           super().__init__()
           
           layers = []
           current_dim = input_dim
           
           for hidden_dim in hidden_dims:
               layers.extend([
                   nn.Linear(current_dim, hidden_dim),
                   nn.ReLU(),
                   nn.Dropout(dropout)
               ])
               current_dim = hidden_dim
           
           layers.append(nn.Linear(current_dim, output_dim))
           self.network = nn.Sequential(*layers)
       
       def forward(self, x):
           return self.network(x)
   
   # Sketched neural network
   class SketchedMLP(nn.Module):
       def __init__(self, input_dim, hidden_dims, output_dim, 
                    num_terms_schedule=None, low_rank_schedule=None, dropout=0.1):
           super().__init__()
           
           # Default parameter schedules
           if num_terms_schedule is None:
               num_terms_schedule = [8] * len(hidden_dims) + [4]
           
           if low_rank_schedule is None:
               base_rank = min(64, min(input_dim, hidden_dims[0]) // 4)
               low_rank_schedule = [base_rank // (i + 1) for i in range(len(hidden_dims))] + [16]
           
           layers = []
           current_dim = input_dim
           
           for i, hidden_dim in enumerate(hidden_dims):
               layers.extend([
                   pr.nn.SKLinear(
                       current_dim, hidden_dim,
                       num_terms=num_terms_schedule[i],
                       low_rank=low_rank_schedule[i]
                   ),
                   nn.ReLU(),
                   nn.Dropout(dropout)
               ])
               current_dim = hidden_dim
           
           # Output layer
           layers.append(pr.nn.SKLinear(
               current_dim, output_dim,
               num_terms=num_terms_schedule[-1],
               low_rank=low_rank_schedule[-1]
           ))
           
           self.network = nn.Sequential(*layers)
       
       def forward(self, x):
           return self.network(x)
   
   # Compare models
   input_dim, hidden_dims, output_dim = 784, [512, 256, 128], 10
   
   standard_model = StandardMLP(input_dim, hidden_dims, output_dim)
   sketched_model = SketchedMLP(input_dim, hidden_dims, output_dim)
   
   # Parameter comparison
   standard_params = sum(p.numel() for p in standard_model.parameters())
   sketched_params = sum(p.numel() for p in sketched_model.parameters())
   
   print(f"Standard model parameters: {standard_params:,}")
   print(f"Sketched model parameters: {sketched_params:,}")
   print(f"Parameter reduction: {(1 - sketched_params/standard_params)*100:.1f}%")

**Training Comparison**

.. code-block:: python

   def create_synthetic_dataset(n_samples=10000, input_dim=784, n_classes=10):
       \"\"\"Create synthetic classification dataset.\"\"\""
       
       # Generate structured data
       class_centers = torch.randn(n_classes, input_dim)
       
       X = []
       y = []
       
       for class_idx in range(n_classes):
           n_class_samples = n_samples // n_classes
           
           # Generate samples around class center
           samples = class_centers[class_idx] + 0.5 * torch.randn(n_class_samples, input_dim)
           labels = torch.full((n_class_samples,), class_idx)
           
           X.append(samples)
           y.append(labels)
       
       X = torch.cat(X, dim=0)
       y = torch.cat(y, dim=0)
       
       # Shuffle data
       perm = torch.randperm(len(X))
       X, y = X[perm], y[perm]
       
       return X, y
   
   def train_and_evaluate(model, train_loader, test_loader, num_epochs=10, lr=0.001):
       \"\"\"Train and evaluate a model.\"\"\""
       
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       model = model.to(device)
       
       criterion = nn.CrossEntropyLoss()
       optimizer = optim.Adam(model.parameters(), lr=lr)
       
       # Training
       model.train()
       train_losses = []
       
       for epoch in range(num_epochs):
           epoch_loss = 0
           for batch_x, batch_y in train_loader:
               batch_x, batch_y = batch_x.to(device), batch_y.to(device)
               
               optimizer.zero_grad()
               outputs = model(batch_x)
               loss = criterion(outputs, batch_y)
               loss.backward()
               optimizer.step()
               
               epoch_loss += loss.item()
           
           avg_loss = epoch_loss / len(train_loader)
           train_losses.append(avg_loss)
           
           if (epoch + 1) % 2 == 0:
               print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
       
       # Evaluation
       model.eval()
       correct = 0
       total = 0
       
       with torch.no_grad():
           for batch_x, batch_y in test_loader:
               batch_x, batch_y = batch_x.to(device), batch_y.to(device)
               outputs = model(batch_x)
               _, predicted = torch.max(outputs.data, 1)
               total += batch_y.size(0)
               correct += (predicted == batch_y).sum().item()
       
       accuracy = correct / total
       return train_losses, accuracy
   
   # Create dataset
   X, y = create_synthetic_dataset(n_samples=5000, input_dim=784, n_classes=10)
   
   # Split into train/test
   split_idx = int(0.8 * len(X))
   X_train, X_test = X[:split_idx], X[split_idx:]
   y_train, y_test = y[:split_idx], y[split_idx:]
   
   # Create data loaders
   train_dataset = TensorDataset(X_train, y_train)
   test_dataset = TensorDataset(X_test, y_test)
   
   train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
   
   # Train both models
   print("Training standard model:")
   standard_losses, standard_acc = train_and_evaluate(
       StandardMLP(784, [512, 256, 128], 10), 
       train_loader, test_loader
   )
   
   print("\\nTraining sketched model:")
   sketched_losses, sketched_acc = train_and_evaluate(
       SketchedMLP(784, [512, 256, 128], 10),
       train_loader, test_loader
   )
   
   print(f"\\nFinal Results:")
   print(f"Standard model accuracy: {standard_acc:.4f}")
   print(f"Sketched model accuracy: {sketched_acc:.4f}")
   print(f"Accuracy difference: {abs(standard_acc - sketched_acc):.4f}")

Advanced Network Architectures
-------------------------------

**Convolutional Networks with Sketched Layers**

.. code-block:: python

   class SketchedCNN(nn.Module):
       \"\"\"CNN with sketched fully connected layers.\"\"\""
       
       def __init__(self, num_classes=10, sketch_params=None):
           super().__init__()
           
           if sketch_params is None:
               sketch_params = {'num_terms': 8, 'low_rank': 64}
           
           # Convolutional layers (standard)
           self.conv_layers = nn.Sequential(
               nn.Conv2d(3, 32, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
               
               nn.Conv2d(32, 64, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
               
               nn.Conv2d(64, 128, 3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(2),
           )
           
           # Calculate flattened size (assuming 32x32 input)
           self.flatten_size = 128 * 4 * 4
           
           # Sketched fully connected layers
           self.fc_layers = nn.Sequential(
               nn.Flatten(),
               pr.nn.SKLinear(
                   self.flatten_size, 512, 
                   num_terms=sketch_params['num_terms'],
                   low_rank=sketch_params['low_rank']
               ),
               nn.ReLU(),
               nn.Dropout(0.2),
               
               pr.nn.SKLinear(
                   512, 256,
                   num_terms=sketch_params['num_terms'] // 2,
                   low_rank=sketch_params['low_rank'] // 2
               ),
               nn.ReLU(),
               nn.Dropout(0.2),
               
               pr.nn.SKLinear(256, num_classes, num_terms=2, low_rank=16)
           )
       
       def forward(self, x):
           x = self.conv_layers(x)
           x = self.fc_layers(x)
           return x

**Residual Networks with Sketching**

.. code-block:: python

   class SketchedResidualBlock(nn.Module):
       \"\"\"Residual block with sketched linear transformations.\"\"\""
       
       def __init__(self, in_features, out_features, num_terms=4, low_rank=32):
           super().__init__()
           
           # Main path with sketched layers
           self.main_path = nn.Sequential(
               pr.nn.SKLinear(in_features, out_features, num_terms=num_terms, low_rank=low_rank),
               nn.BatchNorm1d(out_features),
               nn.ReLU(),
               pr.nn.SKLinear(out_features, out_features, num_terms=num_terms, low_rank=low_rank),
               nn.BatchNorm1d(out_features)
           )
           
           # Shortcut connection
           if in_features != out_features:
               self.shortcut = pr.nn.SKLinear(in_features, out_features, num_terms=2, low_rank=16)
           else:
               self.shortcut = nn.Identity()
           
           self.relu = nn.ReLU()
       
       def forward(self, x):
           main_out = self.main_path(x)
           shortcut_out = self.shortcut(x)
           return self.relu(main_out + shortcut_out)
   
   class SketchedResNet(nn.Module):
       \"\"\"ResNet-style architecture with sketched layers.\"\"\""
       
       def __init__(self, input_dim, block_configs, num_classes):
           super().__init__()
           
           # Input layer
           self.input_layer = nn.Sequential(
               pr.nn.SKLinear(input_dim, block_configs[0]['features'], num_terms=8, low_rank=64),
               nn.BatchNorm1d(block_configs[0]['features']),
               nn.ReLU()
           )
           
           # Residual blocks
           self.blocks = nn.ModuleList()
           current_features = block_configs[0]['features']
           
           for config in block_configs:
               block = SketchedResidualBlock(
                   current_features, 
                   config['features'],
                   num_terms=config.get('num_terms', 4),
                   low_rank=config.get('low_rank', 32)
               )
               self.blocks.append(block)
               current_features = config['features']
           
           # Output layer
           self.output_layer = pr.nn.SKLinear(current_features, num_classes, num_terms=2, low_rank=16)
           
       def forward(self, x):
           x = self.input_layer(x)
           
           for block in self.blocks:
               x = block(x)
           
           x = self.output_layer(x)
           return x
   
   # Example usage
   block_configs = [
       {'features': 256, 'num_terms': 6, 'low_rank': 48},
       {'features': 256, 'num_terms': 6, 'low_rank': 48},
       {'features': 512, 'num_terms': 8, 'low_rank': 64},
       {'features': 512, 'num_terms': 8, 'low_rank': 64},
   ]
   
   resnet_model = SketchedResNet(input_dim=784, block_configs=block_configs, num_classes=10)

**Attention Mechanisms with Sketching**

.. code-block:: python

   class SketchedMultiHeadAttention(nn.Module):
       \"\"\"Multi-head attention with sketched projections.\"\"\""
       
       def __init__(self, d_model, n_heads, num_terms=6, low_rank=48):
           super().__init__()
           
           self.d_model = d_model
           self.n_heads = n_heads
           self.d_k = d_model // n_heads
           
           # Sketched query, key, value projections
           self.query_proj = pr.nn.SKLinear(d_model, d_model, num_terms=num_terms, low_rank=low_rank)
           self.key_proj = pr.nn.SKLinear(d_model, d_model, num_terms=num_terms, low_rank=low_rank)
           self.value_proj = pr.nn.SKLinear(d_model, d_model, num_terms=num_terms, low_rank=low_rank)
           
           # Output projection
           self.output_proj = pr.nn.SKLinear(d_model, d_model, num_terms=num_terms, low_rank=low_rank)
           
           self.dropout = nn.Dropout(0.1)
           
       def forward(self, x, mask=None):
           batch_size, seq_len, d_model = x.shape
           
           # Generate Q, K, V
           Q = self.query_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
           K = self.key_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
           V = self.value_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
           
           # Attention computation
           scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
           
           if mask is not None:
               scores = scores.masked_fill(mask == 0, -1e9)
           
           attention_weights = torch.softmax(scores, dim=-1)
           attention_weights = self.dropout(attention_weights)
           
           context = torch.matmul(attention_weights, V)
           
           # Concatenate heads and project
           context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
           output = self.output_proj(context)
           
           return output, attention_weights
   
   class SketchedTransformerBlock(nn.Module):
       \"\"\"Transformer block with sketched layers.\"\"\""
       
       def __init__(self, d_model, n_heads, d_ff, num_terms=6, low_rank=48):
           super().__init__()
           
           self.attention = SketchedMultiHeadAttention(d_model, n_heads, num_terms, low_rank)
           self.norm1 = nn.LayerNorm(d_model)
           
           # Feed-forward with sketched layers
           self.feed_forward = nn.Sequential(
               pr.nn.SKLinear(d_model, d_ff, num_terms=num_terms*2, low_rank=low_rank*2),
               nn.ReLU(),
               nn.Dropout(0.1),
               pr.nn.SKLinear(d_ff, d_model, num_terms=num_terms, low_rank=low_rank)
           )
           self.norm2 = nn.LayerNorm(d_model)
           self.dropout = nn.Dropout(0.1)
       
       def forward(self, x, mask=None):
           # Self-attention with residual connection
           attn_out, _ = self.attention(x, mask)
           x = self.norm1(x + self.dropout(attn_out))
           
           # Feed-forward with residual connection
           ff_out = self.feed_forward(x)
           x = self.norm2(x + self.dropout(ff_out))
           
           return x

Adaptive Sketching Strategies
-----------------------------

**Dynamic Parameter Adjustment**

.. code-block:: python

   class AdaptiveSketchedLayer(nn.Module):
       \"\"\"Sketched layer that adapts parameters based on input statistics.\"\"\""
       
       def __init__(self, in_features, out_features, 
                    base_terms=4, base_rank=32, adaptation_rate=0.01):
           super().__init__()
           
           self.in_features = in_features
           self.out_features = out_features
           self.base_terms = base_terms
           self.base_rank = base_rank
           self.adaptation_rate = adaptation_rate
           
           # Current parameters
           self.current_terms = base_terms
           self.current_rank = base_rank
           
           # Create initial layer
           self.sketched_layer = pr.nn.SKLinear(
               in_features, out_features, 
               num_terms=self.current_terms, 
               low_rank=self.current_rank
           )
           
           # Statistics tracking
           self.register_buffer('input_variance', torch.ones(1))
           self.register_buffer('gradient_norm', torch.ones(1))
           self.update_counter = 0
           
       def forward(self, x):
           # Track input statistics
           if self.training:
               current_var = torch.var(x, dim=0).mean()
               self.input_variance = (1 - self.adaptation_rate) * self.input_variance + \\
                                   self.adaptation_rate * current_var
               
               self.update_counter += 1
               
               # Adapt parameters periodically
               if self.update_counter % 100 == 0:
                   self.adapt_parameters()
           
           return self.sketched_layer(x)
       
       def adapt_parameters(self):
           \"\"\"Adjust sketching parameters based on observed statistics.\"\"\""
           
           # Simple adaptation rule: increase complexity if high variance
           if self.input_variance > 2.0 and self.current_terms < 16:
               new_terms = min(16, self.current_terms + 2)
               new_rank = min(128, self.current_rank + 16)
               
               # Create new layer with increased parameters
               new_layer = pr.nn.SKLinear(
                   self.in_features, self.out_features,
                   num_terms=new_terms, low_rank=new_rank
               )
               
               # Transfer learned parameters (simplified)
               with torch.no_grad():
                   if self.sketched_layer.bias is not None:
                       new_layer.bias.copy_(self.sketched_layer.bias)
               
               self.sketched_layer = new_layer
               self.current_terms = new_terms
               self.current_rank = new_rank
               
               print(f"Adapted to terms={new_terms}, rank={new_rank}")

**Layer-wise Learning Rate Scheduling**

.. code-block:: python

   class SketchedLayerWithScheduling(nn.Module):
       \"\"\"Sketched layer with built-in learning rate scheduling.\"\"\""
       
       def __init__(self, in_features, out_features, num_terms=8, low_rank=64):
           super().__init__()
           
           self.sketched_layer = pr.nn.SKLinear(in_features, out_features, num_terms, low_rank)
           
           # Separate learning rates for different parameter groups
           self.register_buffer('s_lr_multiplier', torch.tensor(1.0))
           self.register_buffer('u_lr_multiplier', torch.tensor(0.1))  # U matrices are fixed
           
       def get_parameter_groups(self, base_lr):
           \"\"\"Get parameter groups with different learning rates.\"\"\""
           
           s_params = [p for name, p in self.named_parameters() if 'S' in name]
           other_params = [p for name, p in self.named_parameters() if 'S' not in name]
           
           return [
               {'params': s_params, 'lr': base_lr * self.s_lr_multiplier},
               {'params': other_params, 'lr': base_lr}
           ]
       
       def forward(self, x):
           return self.sketched_layer(x)

**Progressive Training Strategy**

.. code-block:: python

   class ProgressiveSketchedNetwork(nn.Module):
       \"\"\"Network that progressively increases sketch complexity during training.\"\"\""
       
       def __init__(self, layer_configs, num_classes):
           super().__init__()
           
           self.layer_configs = layer_configs
           self.num_classes = num_classes
           self.current_phase = 0
           self.max_phases = 3
           
           # Initialize with minimal complexity
           self.build_network()
           
       def build_network(self):
           \"\"\"Build network for current training phase.\"\"\""
           
           layers = []
           phase_multiplier = (self.current_phase + 1) / self.max_phases
           
           for i, config in enumerate(self.layer_configs):
               # Scale parameters based on training phase
               num_terms = max(2, int(config['num_terms'] * phase_multiplier))
               low_rank = max(16, int(config['low_rank'] * phase_multiplier))
               
               layer = pr.nn.SKLinear(
                   config['in_features'], 
                   config['out_features'],
                   num_terms=num_terms,
                   low_rank=low_rank
               )
               
               layers.extend([layer, nn.ReLU(), nn.Dropout(0.1)])
           
           # Output layer
           layers.append(pr.nn.SKLinear(
               self.layer_configs[-1]['out_features'],
               self.num_classes,
               num_terms=2, low_rank=16
           ))
           
           self.network = nn.Sequential(*layers)
       
       def advance_phase(self):
           \"\"\"Move to next training phase with increased complexity.\"\"\""
           
           if self.current_phase < self.max_phases - 1:
               old_state = self.network.state_dict()
               self.current_phase += 1
               self.build_network()
               
               # Transfer compatible parameters
               new_state = self.network.state_dict()
               for key in old_state:
                   if key in new_state and old_state[key].shape == new_state[key].shape:
                       new_state[key] = old_state[key]
               
               self.network.load_state_dict(new_state)
               
               print(f"Advanced to phase {self.current_phase + 1}/{self.max_phases}")
               return True
           
           return False
       
       def forward(self, x):
           return self.network(x)

Memory and Computational Efficiency
------------------------------------

**Memory Monitoring During Training**

.. code-block:: python

   import psutil
   import gc
   
   class MemoryEfficientTrainer:
       \"\"\"Trainer with memory monitoring and optimization.\"\"\""
       
       def __init__(self, model, train_loader, val_loader, device):
           self.model = model.to(device)
           self.train_loader = train_loader
           self.val_loader = val_loader
           self.device = device
           
           self.memory_stats = []
           
       def train_epoch(self, optimizer, criterion, accumulation_steps=1):
           \"\"\"Train one epoch with gradient accumulation.\"\"\""
           
           self.model.train()
           total_loss = 0
           num_batches = 0
           
           optimizer.zero_grad()
           
           for batch_idx, (data, target) in enumerate(self.train_loader):
               data, target = data.to(self.device), target.to(self.device)
               
               # Forward pass
               output = self.model(data)
               loss = criterion(output, target) / accumulation_steps
               
               # Backward pass
               loss.backward()
               
               total_loss += loss.item() * accumulation_steps
               num_batches += 1
               
               # Update weights every accumulation_steps
               if (batch_idx + 1) % accumulation_steps == 0:
                   optimizer.step()
                   optimizer.zero_grad()
               
               # Memory monitoring
               if batch_idx % 50 == 0:
                   self.log_memory_usage(batch_idx)
               
               # Cleanup
               del data, target, output, loss
               if self.device.type == 'cuda':
                   torch.cuda.empty_cache()
           
           return total_loss / num_batches
       
       def log_memory_usage(self, batch_idx):
           \"\"\"Log current memory usage.\"\"\""
           
           if self.device.type == 'cuda':
               gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
               gpu_cached = torch.cuda.memory_reserved() / 1024**2   # MB
           else:
               gpu_memory = gpu_cached = 0
           
           cpu_memory = psutil.virtual_memory().used / 1024**2  # MB
           
           self.memory_stats.append({
               'batch': batch_idx,
               'gpu_allocated': gpu_memory,
               'gpu_cached': gpu_cached,
               'cpu_used': cpu_memory
           })
       
       def validate(self, criterion):
           \"\"\"Validate model performance.\"\"\""
           
           self.model.eval()
           total_loss = 0
           correct = 0
           total = 0
           
           with torch.no_grad():
               for data, target in self.val_loader:
                   data, target = data.to(self.device), target.to(self.device)
                   
                   output = self.model(data)
                   loss = criterion(output, target)
                   
                   total_loss += loss.item()
                   _, predicted = torch.max(output.data, 1)
                   total += target.size(0)
                   correct += (predicted == target).sum().item()
           
           accuracy = correct / total
           avg_loss = total_loss / len(self.val_loader)
           
           return avg_loss, accuracy

**Gradient Checkpointing for Large Models**

.. code-block:: python

   from torch.utils.checkpoint import checkpoint
   
   class CheckpointedSketchedLayer(nn.Module):
       \"\"\"Sketched layer with gradient checkpointing.\"\"\""
       
       def __init__(self, in_features, out_features, num_terms=8, low_rank=64):
           super().__init__()
           
           self.sketched_layer = pr.nn.SKLinear(in_features, out_features, num_terms, low_rank)
           self.activation = nn.ReLU()
           
       def forward(self, x):
           # Use gradient checkpointing to save memory
           return checkpoint(self._forward_impl, x)
       
       def _forward_impl(self, x):
           return self.activation(self.sketched_layer(x))

**Mixed Precision Training**

.. code-block:: python

   def train_with_mixed_precision(model, train_loader, val_loader, num_epochs=10):
       \"\"\"Train model using automatic mixed precision.\"\"\""
       
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       model = model.to(device)
       
       criterion = nn.CrossEntropyLoss()
       optimizer = optim.Adam(model.parameters(), lr=0.001)
       scaler = torch.cuda.amp.GradScaler()
       
       for epoch in range(num_epochs):
           model.train()
           epoch_loss = 0
           
           for batch_idx, (data, target) in enumerate(train_loader):
               data, target = data.to(device), target.to(device)
               
               optimizer.zero_grad()
               
               # Forward pass with autocast
               with torch.cuda.amp.autocast():
                   output = model(data)
                   loss = criterion(output, target)
               
               # Backward pass with gradient scaling
               scaler.scale(loss).backward()
               scaler.step(optimizer)
               scaler.update()
               
               epoch_loss += loss.item()
           
           print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

Real-World Application: Document Classification
-----------------------------------------------

**Complete Document Classification Pipeline**

.. code-block:: python

   class DocumentClassifier(nn.Module):
       \"\"\"Complete document classifier using sketched layers.\"\"\""
       
       def __init__(self, vocab_size, embed_dim=128, hidden_dims=[512, 256], 
                    num_classes=10, max_seq_len=512):
           super().__init__()
           
           # Embedding layer
           self.embedding = nn.Embedding(vocab_size, embed_dim)
           self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
           
           # Transformer-style encoder with sketched layers
           self.encoder = SketchedTransformerBlock(
               d_model=embed_dim, 
               n_heads=8, 
               d_ff=embed_dim*4,
               num_terms=6, 
               low_rank=48
           )
           
           # Global pooling
           self.global_pool = nn.AdaptiveAvgPool1d(1)
           
           # Classification head with sketched layers
           classifier_layers = []
           current_dim = embed_dim
           
           for hidden_dim in hidden_dims:
               classifier_layers.extend([
                   pr.nn.SKLinear(current_dim, hidden_dim, num_terms=8, low_rank=64),
                   nn.ReLU(),
                   nn.Dropout(0.3)
               ])
               current_dim = hidden_dim
           
           classifier_layers.append(pr.nn.SKLinear(current_dim, num_classes, num_terms=4, low_rank=32))
           
           self.classifier = nn.Sequential(*classifier_layers)
       
       def forward(self, input_ids, attention_mask=None):
           # Embedding with positional encoding
           seq_len = input_ids.size(1)
           embeddings = self.embedding(input_ids)
           embeddings = embeddings + self.pos_encoding[:seq_len]
           
           # Encoder
           encoded = self.encoder(embeddings, attention_mask)
           
           # Global pooling
           if attention_mask is not None:
               # Masked average pooling
               masked_encoded = encoded * attention_mask.unsqueeze(-1)
               pooled = masked_encoded.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
           else:
               # Simple average pooling
               pooled = encoded.mean(dim=1)
           
           # Classification
           logits = self.classifier(pooled)
           
           return logits
   
   # Training function for document classification
   def train_document_classifier():
       # Hyperparameters
       vocab_size = 10000
       max_seq_len = 512
       num_classes = 20
       batch_size = 32
       
       # Model
       model = DocumentClassifier(
           vocab_size=vocab_size,
           embed_dim=128,
           hidden_dims=[512, 256],
           num_classes=num_classes,
           max_seq_len=max_seq_len
       )
       
       # Print model statistics
       total_params = sum(p.numel() for p in model.parameters())
       trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
       
       print(f"Model Parameters:")
       print(f"  Total: {total_params:,}")
       print(f"  Trainable: {trainable_params:,}")
       
       # Calculate memory usage for sketched vs standard layers
       sketched_params = sum(p.numel() for name, p in model.named_parameters() 
                            if any(layer_type in name for layer_type in ['S1s', 'S2s']))
       
       print(f"  Sketched layer parameters: {sketched_params:,}")
       
       return model

This comprehensive tutorial covers building neural networks with Panther's sketched layers. The next tutorial will focus on performance optimization techniques.
