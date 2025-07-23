AutoTuner Guide
===============

This guide shows how to use Panther's AutoTuner to automatically optimize sketching parameters for your specific use case.

Introduction to AutoTuning
---------------------------

Choosing optimal sketching parameters (``num_terms`` and ``low_rank``) can be challenging. The AutoTuner uses Bayesian optimization to automatically find the best parameters for your model and dataset.

**Why AutoTune?**

- **Better Performance**: Find parameters that work best for your specific data
- **Save Time**: Avoid manual hyperparameter search
- **Principled Approach**: Uses Bayesian optimization instead of random search
- **Multi-Objective**: Can optimize for accuracy, speed, and memory usage

Basic AutoTuner Usage
----------------------

**Simple Linear Layer Optimization**

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   from panther.tuner import SkAutoTuner
   
   # Define your evaluation function
   def evaluate_linear_layer(num_terms, low_rank):
       \"\"\"Evaluate a sketched linear layer configuration.\"\"\""
       
       # Create model with given parameters
       model = nn.Sequential(
           pr.nn.SKLinear(784, 512, num_terms=int(num_terms), low_rank=int(low_rank)),
           nn.ReLU(),
           pr.nn.SKLinear(512, 10, num_terms=max(1, int(num_terms)//2), low_rank=max(8, int(low_rank)//2))
       )
       
       # Simple evaluation (replace with your actual training/validation)
       criterion = nn.CrossEntropyLoss()
       optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
       
       # Quick training
       for epoch in range(3):
           for _ in range(50):  # Limited batches for speed
               x = torch.randn(32, 784)
               y = torch.randint(0, 10, (32,))
               
               optimizer.zero_grad()
               output = model(x)
               loss = criterion(output, y)
               loss.backward()
               optimizer.step()
       
       # Simple validation
       model.eval()
       correct = 0
       total = 0
       with torch.no_grad():
           for _ in range(20):
               x = torch.randn(32, 784)
               y = torch.randint(0, 10, (32,))
               output = model(x)
               _, predicted = torch.max(output.data, 1)
               total += y.size(0)
               correct += (predicted == y).sum().item()
       
       accuracy = correct / total
       return accuracy
   
   # Create AutoTuner
   tuner = SkAutoTuner(
       parameter_bounds={
           'num_terms': (2, 16),    # Search between 2 and 16 terms
           'low_rank': (16, 128)    # Search between 16 and 128 rank
       },
       objective_function=evaluate_linear_layer,
       n_initial_points=5,          # Start with 5 random evaluations
       n_iterations=20              # Then do 20 optimization steps
   )
   
   # Run optimization
   best_params, best_score = tuner.optimize()
   
   print(f"Best parameters found:")
   print(f"  num_terms: {best_params['num_terms']}")
   print(f"  low_rank: {best_params['low_rank']}")
   print(f"  Best accuracy: {best_score:.4f}")

Real-World Example: MNIST Classification
-----------------------------------------

**Complete MNIST AutoTuning Pipeline**

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torch.utils.data import DataLoader
   import torchvision
   import torchvision.transforms as transforms
   from panther.tuner import SkAutoTuner
   import panther as pr
   
   class SketchedMNISTNet(nn.Module):
       def __init__(self, layer1_terms, layer1_rank, layer2_terms, layer2_rank):
           super().__init__()
           self.flatten = nn.Flatten()
           self.layer1 = pr.nn.SKLinear(784, 512, 
                                       num_terms=int(layer1_terms), 
                                       low_rank=int(layer1_rank))
           self.relu1 = nn.ReLU()
           self.dropout1 = nn.Dropout(0.2)
           
           self.layer2 = pr.nn.SKLinear(512, 256,
                                       num_terms=int(layer2_terms),
                                       low_rank=int(layer2_rank))
           self.relu2 = nn.ReLU()
           self.dropout2 = nn.Dropout(0.2)
           
           self.layer3 = pr.nn.SKLinear(256, 10,
                                       num_terms=2, low_rank=16)  # Fixed for output
       
       def forward(self, x):
           x = self.flatten(x)
           x = self.dropout1(self.relu1(self.layer1(x)))
           x = self.dropout2(self.relu2(self.layer2(x)))
           x = self.layer3(x)
           return x
   
   def prepare_mnist_data():
       \"\"\"Prepare MNIST dataset.\"\"\""
       transform = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
       ])
       
       train_dataset = torchvision.datasets.MNIST(
           root='./data', train=True, download=True, transform=transform
       )
       test_dataset = torchvision.datasets.MNIST(
           root='./data', train=False, download=True, transform=transform
       )
       
       # Use subset for faster tuning
       train_subset = torch.utils.data.Subset(train_dataset, range(0, 5000))
       test_subset = torch.utils.data.Subset(test_dataset, range(0, 1000))
       
       train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
       test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
       
       return train_loader, test_loader
   
   def evaluate_mnist_model(layer1_terms, layer1_rank, layer2_terms, layer2_rank):
       \"\"\"Evaluate MNIST model with given parameters.\"\"\""
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
       # Create model
       model = SketchedMNISTNet(layer1_terms, layer1_rank, layer2_terms, layer2_rank)
       model = model.to(device)
       
       # Prepare data
       train_loader, test_loader = prepare_mnist_data()
       
       # Training setup
       criterion = nn.CrossEntropyLoss()
       optimizer = optim.Adam(model.parameters(), lr=0.001)
       
       # Quick training (2 epochs for speed)
       model.train()
       for epoch in range(2):
           for batch_idx, (data, target) in enumerate(train_loader):
               data, target = data.to(device), target.to(device)
               
               optimizer.zero_grad()
               output = model(data)
               loss = criterion(output, target)
               loss.backward()
               optimizer.step()
               
               # Limit batches for speed during tuning
               if batch_idx > 30:
                   break
       
       # Evaluation
       model.eval()
       correct = 0
       total = 0
       
       with torch.no_grad():
           for data, target in test_loader:
               data, target = data.to(device), target.to(device)
               output = model(data)
               _, predicted = torch.max(output.data, 1)
               total += target.size(0)
               correct += (predicted == target).sum().item()
       
       accuracy = correct / total
       
       # Clean up GPU memory
       del model
       if device.type == 'cuda':
           torch.cuda.empty_cache()
       
       return accuracy
   
   # Run AutoTuning
   def tune_mnist_model():
       \"\"\"Run complete AutoTuning for MNIST.\"\"\""
       
       tuner = SkAutoTuner(
           parameter_bounds={
               'layer1_terms': (2, 12),
               'layer1_rank': (16, 96),
               'layer2_terms': (2, 8),
               'layer2_rank': (12, 64)
           },
           objective_function=evaluate_mnist_model,
           n_initial_points=8,
           n_iterations=25
       )
       
       print("Starting AutoTuning for MNIST...")
       best_params, best_score = tuner.optimize()
       
       print(f"\\nAutoTuning completed!")
       print(f"Best parameters:")
       for param, value in best_params.items():
           print(f"  {param}: {value}")
       print(f"Best accuracy: {best_score:.4f}")
       
       return best_params, best_score
   
   # Run the tuning
   if __name__ == "__main__":
       best_params, best_score = tune_mnist_model()

Multi-Objective Optimization
-----------------------------

**Optimizing for Accuracy and Speed**

.. code-block:: python

   import time
   
   def multi_objective_evaluation(num_terms, low_rank):
       \"\"\"Evaluate model considering both accuracy and inference speed.\"\"\""
       
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
       # Create model
       model = nn.Sequential(
           pr.nn.SKLinear(1024, 512, num_terms=int(num_terms), low_rank=int(low_rank)),
           nn.ReLU(),
           pr.nn.SKLinear(512, 10, num_terms=max(1, int(num_terms)//2), low_rank=max(8, int(low_rank)//2))
       ).to(device)
       
       # Measure accuracy (simplified)
       accuracy = train_and_evaluate_model(model)  # Your training function
       
       # Measure inference speed
       model.eval()
       dummy_input = torch.randn(64, 1024, device=device)
       
       # Warmup
       with torch.no_grad():
           for _ in range(10):
               _ = model(dummy_input)
       
       # Time inference
       if device.type == 'cuda':
           torch.cuda.synchronize()
       
       start_time = time.time()
       with torch.no_grad():
           for _ in range(100):
               _ = model(dummy_input)
               if device.type == 'cuda':
                   torch.cuda.synchronize()
       
       inference_time = (time.time() - start_time) / 100
       
       # Combine objectives (accuracy more important)
       speed_score = 1.0 / (inference_time + 1e-6)  # Higher is better
       normalized_speed = min(speed_score / 1000, 1.0)  # Normalize to [0, 1]
       
       combined_score = 0.7 * accuracy + 0.3 * normalized_speed
       
       return combined_score
   
   # AutoTune with multi-objective
   multi_obj_tuner = SkAutoTuner(
       parameter_bounds={
           'num_terms': (1, 20),
           'low_rank': (8, 256)
       },
       objective_function=multi_objective_evaluation,
       n_initial_points=10,
       n_iterations=30
   )
   
   best_params, best_score = multi_obj_tuner.optimize()

Memory-Constrained Optimization
--------------------------------

**Optimizing Under Memory Limits**

.. code-block:: python

   import psutil
   import gc
   
   def memory_constrained_evaluation(num_terms, low_rank, max_memory_gb=4.0):
       \"\"\"Evaluate model with memory constraints.\"\"\""
       
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
       # Estimate memory usage
       estimated_params = estimate_parameter_count(int(num_terms), int(low_rank))
       estimated_memory = estimated_params * 4 / (1024**3)  # 4 bytes per float32
       
       # Early rejection if estimated memory too high
       if estimated_memory > max_memory_gb:
           return 0.0
       
       try:
           # Monitor initial memory
           if device.type == 'cuda':
               torch.cuda.reset_peak_memory_stats()
               initial_memory = torch.cuda.memory_allocated()
           else:
               process = psutil.Process()
               initial_memory = process.memory_info().rss
           
           # Create and train model
           model = create_model(int(num_terms), int(low_rank))
           model = model.to(device)
           
           accuracy = quick_train_and_evaluate(model)
           
           # Check actual memory usage
           if device.type == 'cuda':
               peak_memory = torch.cuda.max_memory_allocated()
               memory_used = (peak_memory - initial_memory) / (1024**3)
           else:
               current_memory = process.memory_info().rss
               memory_used = (current_memory - initial_memory) / (1024**3)
           
           # Penalize if memory exceeded
           if memory_used > max_memory_gb:
               return 0.0
           
           # Reward memory efficiency
           memory_efficiency = max_memory_gb / (memory_used + 0.1)
           adjusted_score = accuracy * min(memory_efficiency, 2.0)
           
           return adjusted_score
           
       except torch.cuda.OutOfMemoryError:
           return 0.0  # Penalize OOM configurations
       
       finally:
           # Clean up
           if 'model' in locals():
               del model
           if device.type == 'cuda':
               torch.cuda.empty_cache()
           gc.collect()
   
   def estimate_parameter_count(num_terms, low_rank):
       \"\"\"Estimate total parameter count for given configuration.\"\"\""
       # Simplified estimation for a typical model
       layer_configs = [(1024, 512), (512, 256), (256, 10)]
       total_params = 0
       
       for in_feat, out_feat in layer_configs:
           sketched_params = 2 * num_terms * low_rank * (in_feat + out_feat)
           total_params += sketched_params
       
       return total_params
   
   # Memory-constrained tuning
   memory_tuner = SkAutoTuner(
       parameter_bounds={
           'num_terms': (1, 32),
           'low_rank': (8, 512)
       },
       objective_function=lambda nt, lr: memory_constrained_evaluation(nt, lr, max_memory_gb=2.0),
       n_initial_points=12,
       n_iterations=40
   )

Progressive Tuning Strategy
---------------------------

**Phase 1: Broad Search, Phase 2: Fine-Tuning**

.. code-block:: python

   def progressive_autotuning():
       \"\"\"Two-phase AutoTuning: broad search then refinement.\"\"\""
       
       # Phase 1: Broad exploration
       print("Phase 1: Broad parameter exploration...")
       
       broad_tuner = SkAutoTuner(
           parameter_bounds={
               'num_terms': (1, 32),
               'low_rank': (8, 512)
           },
           objective_function=evaluate_model_function,
           n_initial_points=15,
           n_iterations=30
       )
       
       broad_best_params, broad_best_score = broad_tuner.optimize()
       
       print(f"Phase 1 best: {broad_best_params} (score: {broad_best_score:.4f})")
       
       # Phase 2: Refined search around best region
       print("\\nPhase 2: Refined search...")
       
       # Define refined bounds around Phase 1 best
       margin_terms = 4
       margin_rank = 32
       
       refined_bounds = {
           'num_terms': (
               max(1, broad_best_params['num_terms'] - margin_terms),
               broad_best_params['num_terms'] + margin_terms
           ),
           'low_rank': (
               max(8, broad_best_params['low_rank'] - margin_rank),
               broad_best_params['low_rank'] + margin_rank
           )
       }
       
       refined_tuner = SkAutoTuner(
           parameter_bounds=refined_bounds,
           objective_function=evaluate_model_function,
           n_initial_points=8,
           n_iterations=20
       )
       
       refined_best_params, refined_best_score = refined_tuner.optimize()
       
       print(f"Phase 2 best: {refined_best_params} (score: {refined_best_score:.4f})")
       
       # Return the better of the two
       if refined_best_score > broad_best_score:
           return refined_best_params, refined_best_score
       else:
           return broad_best_params, broad_best_score

Integration with Popular Frameworks
------------------------------------

**PyTorch Lightning Integration**

.. code-block:: python

   import pytorch_lightning as pl
   from pytorch_lightning.callbacks import EarlyStopping
   from panther.tuner import SkAutoTuner
   
   class SketchedLightningModule(pl.LightningModule):
       def __init__(self, num_terms=8, low_rank=64):
           super().__init__()
           self.save_hyperparameters()
           
           self.model = nn.Sequential(
               pr.nn.SKLinear(784, 512, num_terms=num_terms, low_rank=low_rank),
               nn.ReLU(),
               nn.Dropout(0.2),
               pr.nn.SKLinear(512, 10, num_terms=max(1, num_terms//2), low_rank=max(8, low_rank//2))
           )
           
           self.criterion = nn.CrossEntropyLoss()
           self.train_acc = pl.metrics.Accuracy()
           self.val_acc = pl.metrics.Accuracy()
       
       def forward(self, x):
           return self.model(x.view(x.size(0), -1))
       
       def training_step(self, batch, batch_idx):
           x, y = batch
           logits = self(x)
           loss = self.criterion(logits, y)
           
           preds = torch.argmax(logits, dim=1)
           self.train_acc(preds, y)
           
           self.log('train_loss', loss, on_step=True, on_epoch=True)
           self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
           
           return loss
       
       def validation_step(self, batch, batch_idx):
           x, y = batch
           logits = self(x)
           loss = self.criterion(logits, y)
           
           preds = torch.argmax(logits, dim=1)
           self.val_acc(preds, y)
           
           self.log('val_loss', loss)
           self.log('val_acc', self.val_acc)
           
           return loss
       
       def configure_optimizers(self):
           optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
           return optimizer
   
   def evaluate_lightning_model(num_terms, low_rank):
       \"\"\"Evaluate model using PyTorch Lightning.\"\"\""
       
       # Create model
       model = SketchedLightningModule(int(num_terms), int(low_rank))
       
       # Data module (implement your own)
       dm = MNISTDataModule(batch_size=64)
       
       # Trainer with early stopping
       trainer = pl.Trainer(
           max_epochs=5,
           callbacks=[EarlyStopping(monitor='val_loss', patience=2)],
           enable_progress_bar=False,
           enable_model_summary=False,
           logger=False
       )
       
       # Fit model
       trainer.fit(model, dm)
       
       # Return validation accuracy
       return trainer.callback_metrics['val_acc'].item()
   
   # AutoTune with Lightning
   lightning_tuner = SkAutoTuner(
       parameter_bounds={'num_terms': (2, 16), 'low_rank': (16, 128)},
       objective_function=evaluate_lightning_model,
       n_initial_points=6,
       n_iterations=20
   )

Best Practices for AutoTuning
------------------------------

**1. Start with Quick Evaluations**

.. code-block:: python

   def quick_evaluation_for_tuning(num_terms, low_rank):
       \"\"\"Fast evaluation suitable for initial tuning phases.\"\"\""
       
       # Use smaller models/datasets during tuning
       model = create_small_model(int(num_terms), int(low_rank))
       
       # Train for fewer epochs
       quick_train(model, max_epochs=3, max_batches_per_epoch=50)
       
       # Evaluate on smaller validation set
       accuracy = quick_evaluate(model, validation_subset_size=500)
       
       return accuracy

**2. Use Early Stopping**

.. code-block:: python

   def evaluation_with_early_stopping(num_terms, low_rank):
       \"\"\"Evaluation with early stopping to save time.\"\"\""
       
       model = create_model(int(num_terms), int(low_rank))
       
       best_val_acc = 0
       patience = 3
       patience_counter = 0
       
       for epoch in range(20):  # Max epochs
           train_one_epoch(model)
           val_acc = validate_model(model)
           
           if val_acc > best_val_acc:
               best_val_acc = val_acc
               patience_counter = 0
           else:
               patience_counter += 1
           
           if patience_counter >= patience:
               break  # Early stopping
       
       return best_val_acc

**3. Cache Expensive Computations**

.. code-block:: python

   from functools import lru_cache
   
   @lru_cache(maxsize=128)
   def cached_evaluation(num_terms, low_rank):
       \"\"\"Cache results to avoid re-evaluating same parameters.\"\"\""
       return expensive_evaluation_function(num_terms, low_rank)

**4. Parallel Evaluation (Advanced)**

.. code-block:: python

   from joblib import Parallel, delayed
   
   def parallel_autotuning():
       \"\"\"Use parallel evaluation for faster tuning.\"\"\""
       
       def parallel_objective_wrapper(params_dict):
           return evaluate_model_function(**params_dict)
       
       # This requires a custom implementation
       # Standard SkAutoTuner doesn't support parallel evaluation yet
       # But you can implement batch evaluation
       
       # Generate parameter combinations
       param_combinations = generate_parameter_grid()
       
       # Evaluate in parallel
       results = Parallel(n_jobs=4)(
           delayed(parallel_objective_wrapper)(params) 
           for params in param_combinations
       )
       
       # Find best
       best_idx = np.argmax(results)
       return param_combinations[best_idx], results[best_idx]

Interpreting AutoTuning Results
-------------------------------

**Analyzing Parameter Sensitivity**

.. code-block:: python

   def analyze_tuning_results(tuner):
       \"\"\"Analyze which parameters are most important.\"\"\""
       
       # Get all evaluated points (if tuner stores them)
       evaluations = tuner.get_evaluation_history()  # Hypothetical method
       
       # Analyze parameter correlations
       import pandas as pd
       import matplotlib.pyplot as plt
       
       df = pd.DataFrame(evaluations)
       
       # Correlation matrix
       correlation_matrix = df.corr()
       print("Parameter correlations:")
       print(correlation_matrix)
       
       # Plot parameter vs. performance
       fig, axes = plt.subplots(2, 2, figsize=(12, 10))
       
       axes[0, 0].scatter(df['num_terms'], df['score'])
       axes[0, 0].set_xlabel('num_terms')
       axes[0, 0].set_ylabel('Score')
       
       axes[0, 1].scatter(df['low_rank'], df['score'])
       axes[0, 1].set_xlabel('low_rank')
       axes[0, 1].set_ylabel('Score')
       
       # Parameter interaction
       scatter = axes[1, 0].scatter(df['num_terms'], df['low_rank'], c=df['score'], cmap='viridis')
       axes[1, 0].set_xlabel('num_terms')
       axes[1, 0].set_ylabel('low_rank')
       plt.colorbar(scatter, ax=axes[1, 0])
       
       plt.tight_layout()
       plt.show()

This comprehensive guide covers all aspects of using Panther's AutoTuner effectively. Remember to start with quick evaluations and progressively refine your search for the best results.
