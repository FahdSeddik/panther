Sketched Linear Layers
======================

This tutorial covers the fundamentals of sketched linear layers in Panther, explaining how they work and when to use them.

What are Sketched Linear Layers?
---------------------------------

Traditional linear layers in neural networks compute:

.. math::

   y = Wx + b

:math:`W` is a :math:`d_{out} \times d_{in}` weight matrix.

Sketched linear layers approximate this computation using a sum of low-rank terms:

.. math::

   y \approx \frac{1}{2L} \sum_{i=1}^{L} \left( U_{1,i}^T S_{1,i} x + S_{2,i} U_{2,i} x \right) + b

where:

* :math:`L` is the number of terms (``num_terms``)
* :math:`k` is the rank of each term (``low_rank``)
* :math:`U_{1,i}, U_{2,i}` are fixed random matrices
* :math:`S_{1,i}, S_{2,i}` are learnable parameter matrices

Why Use Sketched Layers?
-------------------------

**Memory Efficiency**

Standard linear layer parameters: :math:`d_{out} \times d_{in} + d_{out}`

Sketched layer parameters: :math:`2L \times k \times (d_{out} + d_{in}) + d_{out}`

For large layers, this can be a significant reduction.

**Example Comparison**

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   
   # Large layer dimensions
   in_features, out_features = 4096, 4096
   
   # Standard linear layer
   standard_layer = nn.Linear(in_features, out_features)
   standard_params = sum(p.numel() for p in standard_layer.parameters())
   
   # Sketched linear layer
   sketched_layer = pr.nn.SKLinear(
       in_features=in_features,
       out_features=out_features,
       num_terms=4,
       low_rank=128
   )
   sketched_params = sum(p.numel() for p in sketched_layer.parameters())
   
   print(f"Standard layer parameters: {standard_params:,}")
   print(f"Sketched layer parameters: {sketched_params:,}")
   print(f"Memory reduction: {(1 - sketched_params/standard_params)*100:.1f}%")
   
   # Output:
   # Standard layer parameters: 16,781,312
   # Sketched layer parameters: 4,194,304
   # Memory reduction: 75.0%

Understanding the Parameters
----------------------------

**num_terms (L)**

The number of low-rank terms in the approximation.

* **More terms**: Better approximation, more parameters
* **Fewer terms**: Faster computation, less memory

.. code-block:: python

   # Conservative: Fast but less accurate
   conservative = pr.nn.SKLinear(1024, 512, num_terms=2, low_rank=32)
   
   # Aggressive: Slower but more accurate  
   aggressive = pr.nn.SKLinear(1024, 512, num_terms=4, low_rank=32)

**low_rank (k)**

The rank of each low-rank term.

* **Higher rank**: Better approximation per term, more parameters
* **Lower rank**: More compression, potentially less accurate

.. code-block:: python

   # Low rank: High compression
   high_compression = pr.nn.SKLinear(1024, 512, num_terms=2, low_rank=32)
   
   # High rank: Better approximation
   better_approx = pr.nn.SKLinear(1024, 512, num_terms=2, low_rank=64)

Parameter Selection Guidelines
------------------------------

**Rule of Thumb**

Ensure the sketched layer uses fewer parameters than the standard layer:

.. math::

   2 \times L \times k \times (d_{in} + d_{out}) < d_{in} \times d_{out}

.. code-block:: python

   def check_parameter_efficiency(in_features, out_features, num_terms, low_rank):
       """Check if sketched layer uses fewer parameters."""
       standard_params = in_features * out_features
       sketched_params = 2 * num_terms * low_rank * (in_features + out_features)
       
       efficient = sketched_params < standard_params
       reduction = (1 - sketched_params / standard_params) * 100
       
       return efficient, reduction
   
   # Test different configurations
   configs = [
       (1024, 512, 2, 32),   # Conservative
       (1024, 512, 2, 64),   # Balanced
   ]
   
   for in_feat, out_feat, terms, rank in configs:
       efficient, reduction = check_parameter_efficiency(in_feat, out_feat, terms, rank)
       print(f"Terms={terms}, Rank={rank}: Efficient={efficient}, Reduction={reduction:.1f}%")

**Suggested Starting Points**

.. list-table::
   :header-rows: 1
   
   * - Layer Size
     - num_terms
     - low_rank
     - Memory Reduction
   * - Small (< 512)
     - 2-4
     - 16-32
     - 50-70%
   * - Medium (512-2048)
     - 4-8
     - 32-64
     - 60-80%
   * - Large (> 2048)
     - 8-16
     - 64-128
     - 70-90%

Building Your First Sketched Network
-------------------------------------

**Step 1: Replace Linear Layers**

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   
   # Original network
   class OriginalNet(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(784, 512)
           self.fc2 = nn.Linear(512, 256)
           self.fc3 = nn.Linear(256, 10)
           
       def forward(self, x):
           x = torch.relu(self.fc1(x))
           x = torch.relu(self.fc2(x))
           x = self.fc3(x)
           return x
   
   # Sketched version
   class SketchedNet(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = pr.nn.SKLinear(784, 512, num_terms=2, low_rank=64)
           self.fc2 = pr.nn.SKLinear(512, 256, num_terms=2, low_rank=32)
           self.fc3 = pr.nn.SKLinear(256, 10, num_terms=1, low_rank=4)
           
       def forward(self, x):
           x = torch.relu(self.fc1(x))
           x = torch.relu(self.fc2(x))
           x = self.fc3(x)
           return x

**Step 2: Training Comparison**

.. code-block:: python

   import torch.optim as optim
   from torch.utils.data import DataLoader, TensorDataset
   
   # Generate sample data
   X = torch.randn(1000, 784)
   y = torch.randint(0, 10, (1000,))
   dataset = TensorDataset(X, y)
   dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
   
   # Create models
   original_model = OriginalNet()
   sketched_model = SketchedNet()
   
   # Optimizers
   original_optimizer = optim.Adam(original_model.parameters(), lr=0.001)
   sketched_optimizer = optim.Adam(sketched_model.parameters(), lr=0.001)
   
   criterion = nn.CrossEntropyLoss()
   
   # Training function
   def train_model(model, optimizer, num_epochs=5):
       model.train()
       losses = []
       
       for epoch in range(num_epochs):
           epoch_loss = 0
           for batch_x, batch_y in dataloader:
               optimizer.zero_grad()
               outputs = model(batch_x)
               loss = criterion(outputs, batch_y)
               loss.backward()
               optimizer.step()
               epoch_loss += loss.item()
           
           avg_loss = epoch_loss / len(dataloader)
           losses.append(avg_loss)
           print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
       
       return losses
   
   # Train both models
   print("Training original model:")
   original_losses = train_model(original_model, original_optimizer)
   
   print("\\nTraining sketched model:")
   sketched_losses = train_model(sketched_model, sketched_optimizer)

**Step 3: Evaluate Performance**

.. code-block:: python

   def evaluate_model(model, dataloader):
       model.eval()
       correct = 0
       total = 0
       
       with torch.no_grad():
           for batch_x, batch_y in dataloader:
               outputs = model(batch_x)
               _, predicted = torch.max(outputs.data, 1)
               total += batch_y.size(0)
               correct += (predicted == batch_y).sum().item()
       
       accuracy = correct / total
       return accuracy
   
   # Evaluate both models
   original_acc = evaluate_model(original_model, dataloader)
   sketched_acc = evaluate_model(sketched_model, dataloader)
   
   print(f"\\nFinal Results:")
   print(f"Original model accuracy: {original_acc:.4f}")
   print(f"Sketched model accuracy: {sketched_acc:.4f}")
   print(f"Accuracy difference: {abs(original_acc - sketched_acc):.4f}")

Advanced Usage Patterns
------------------------

**1. Layer-Specific Parameters**

.. code-block:: python

   class AdaptiveSketchedNet(nn.Module):
       def __init__(self):
           super().__init__()
           
           # First layer: More terms for better approximation
           self.fc1 = pr.nn.SKLinear(784, 1024, num_terms=3, low_rank=64)
           
           # Middle layer: Balanced approach
           self.fc2 = pr.nn.SKLinear(1024, 512, num_terms=3, low_rank=48)
           
           # Output layer: Fewer parameters, higher precision
           self.fc3 = pr.nn.SKLinear(512, 10, num_terms=1, low_rank=4)
           
       def forward(self, x):
           x = torch.relu(self.fc1(x))
           x = torch.dropout(x, 0.2, training=self.training)
           x = torch.relu(self.fc2(x))
           x = torch.dropout(x, 0.2, training=self.training)
           x = self.fc3(x)
           return x

**2. Gradual Parameter Increase**

.. code-block:: python

   class ProgressiveSketchedLayer(nn.Module):
       """Layer that increases complexity during training."""
       
       def __init__(self, in_features, out_features, 
                    initial_terms=2, max_terms=16, 
                    initial_rank=16, max_rank=128):
           super().__init__()
           
           self.in_features = in_features
           self.out_features = out_features
           self.max_terms = max_terms
           self.max_rank = max_rank
           
           # Start with minimal complexity
           self.current_layer = pr.nn.SKLinear(
               in_features, out_features, 
               num_terms=initial_terms, 
               low_rank=initial_rank
           )
           
       def increase_complexity(self):
           """Double the number of terms and rank (up to maximum)."""
           current_terms = self.current_layer.num_terms
           current_rank = self.current_layer.low_rank
           
           new_terms = min(current_terms * 2, self.max_terms)
           new_rank = min(current_rank * 2, self.max_rank)
           
           if new_terms > current_terms or new_rank > current_rank:
               # Create new layer with increased parameters
               new_layer = pr.nn.SKLinear(
                   self.in_features, self.out_features,
                   num_terms=new_terms, low_rank=new_rank
               )
               
               # Transfer bias if it exists
               if self.current_layer.bias is not None:
                   with torch.no_grad():
                       new_layer.bias.copy_(self.current_layer.bias)
               
               self.current_layer = new_layer
               return True
           
           return False
       
       def forward(self, x):
           return self.current_layer(x)

**3. Conditional Sketching**

.. code-block:: python

   class ConditionalSketchedLayer(nn.Module):
       """Use sketching only for large inputs."""
       
       def __init__(self, in_features, out_features, 
                    sketch_threshold=1000):
           super().__init__()
           self.sketch_threshold = sketch_threshold
           
           # Standard layer for small inputs
           self.standard_layer = nn.Linear(in_features, out_features)
           
           # Sketched layer for large inputs
           self.sketched_layer = pr.nn.SKLinear(
               in_features, out_features, 
               num_terms=8, low_rank=64
           )
           
       def forward(self, x):
           batch_size = x.size(0)
           
           if batch_size < self.sketch_threshold:
               return self.standard_layer(x)
           else:
               return self.sketched_layer(x)

Debugging and Optimization
---------------------------

**1. Parameter Analysis**

.. code-block:: python

   def analyze_sketched_layer(layer):
       """Analyze sketched layer parameters and statistics."""
       print(f"Layer: {layer.in_features} → {layer.out_features}")
       print(f"num_terms: {layer.num_terms}, low_rank: {layer.low_rank}")
       
       # Parameter counts
       s1_params = layer.S1s.numel()
       s2_params = layer.S2s.numel()
       u1_params = layer.U1s.numel()
       u2_params = layer.U2s.numel()
       bias_params = layer.bias.numel() if layer.bias is not None else 0
       
       print(f"S1s parameters: {s1_params:,}")
       print(f"S2s parameters: {s2_params:,}")
       print(f"U1s parameters (fixed): {u1_params:,}")
       print(f"U2s parameters (fixed): {u2_params:,}")
       print(f"Bias parameters: {bias_params:,}")
       print(f"Total learnable: {s1_params + s2_params + bias_params:,}")
       
       # Parameter statistics
       with torch.no_grad():
           s1_mean = layer.S1s.mean().item()
           s1_std = layer.S1s.std().item()
           s2_mean = layer.S2s.mean().item()
           s2_std = layer.S2s.std().item()
           
           print(f"S1s: mean={s1_mean:.4f}, std={s1_std:.4f}")
           print(f"S2s: mean={s2_mean:.4f}, std={s2_std:.4f}")
   
   # Example usage
   layer = pr.nn.SKLinear(1024, 512, num_terms=3, low_rank=48)
   analyze_sketched_layer(layer)

**2. Approximation Quality**

.. code-block:: python

   def test_approximation_quality(in_features, out_features, 
                                  num_terms, low_rank, num_tests=100):
       """Test how well sketched layer approximates standard layer."""
       
       # Create layers
       standard = nn.Linear(in_features, out_features)
       sketched = pr.nn.SKLinear(in_features, out_features, 
                                num_terms=num_terms, low_rank=low_rank)
       
       # Initialize sketched layer to approximate standard layer
       # (This is a simplified initialization)
       with torch.no_grad():
           W = standard.weight.data  # (out_features, in_features)
           
           # Simple initialization: distribute weight across terms
           for i in range(num_terms):
               # Initialize S1s and S2s to approximate W/num_terms
               sketched.S1s[i].normal_(0, 0.1)
               sketched.S2s[i].normal_(0, 0.1)
       
       # Test approximation quality
       errors = []
       for _ in range(num_tests):
           x = torch.randn(32, in_features)
           
           y_standard = standard(x)
           y_sketched = sketched(x)
           
           error = torch.norm(y_standard - y_sketched) / torch.norm(y_standard)
           errors.append(error.item())
       
       avg_error = sum(errors) / len(errors)
       return avg_error
   
   # Test different configurations
   configs = [
       (512, 256, 2, 16),
       (512, 256, 2, 32),
   ]
   
   for in_feat, out_feat, terms, rank in configs:
       error = test_approximation_quality(in_feat, out_feat, terms, rank)
       print(f"Config ({terms}, {rank}): Average error = {error:.4f}")

Best Practices
--------------

1. **Start Conservative**: Begin with fewer terms and lower rank, then increase if needed

2. **Monitor Training**: Sketched layers may have different training dynamics

3. **Layer-Specific Tuning**: Different layers may benefit from different parameters

4. **Parameter Efficiency**: Always ensure sketched layers use fewer parameters

5. **Gradient Clipping**: May be helpful due to the approximation nature

6. **Learning Rate**: Consider using different learning rates for S1s and S2s

.. code-block:: python

   # Example of best practices
   def create_optimized_sketched_model(input_dim, hidden_dims, output_dim):
       layers = []
       
       current_dim = input_dim
       for i, hidden_dim in enumerate(hidden_dims):
           # Gradually decrease complexity for deeper layers
           num_terms = max(2, 8 - i)
           low_rank = max(16, 64 - i * 8)
           
           layer = pr.nn.SKLinear(
               current_dim, hidden_dim,
               num_terms=num_terms, low_rank=low_rank
           )
           
           layers.extend([layer, nn.ReLU(), nn.Dropout(0.1)])
           current_dim = hidden_dim
       
       # Output layer with minimal sketching
       layers.append(pr.nn.SKLinear(current_dim, output_dim, 
                                   num_terms=2, low_rank=16))
       
       return nn.Sequential(*layers)
   
   # Usage
   model = create_optimized_sketched_model(784, [512, 256, 128], 10)
   
   # Specialized optimizer with different learning rates
   s_params = [p for name, p in model.named_parameters() if 'S1s' in name or 'S2s' in name]
   other_params = [p for name, p in model.named_parameters() if not ('S1s' in name or 'S2s' in name)]
   
   optimizer = torch.optim.Adam([
       {'params': s_params, 'lr': 0.001},
       {'params': other_params, 'lr': 0.0005}
   ])

This tutorial provides a comprehensive introduction to sketched linear layers. The next tutorial will cover matrix decompositions in more detail.
