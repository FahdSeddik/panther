AutoTuner API
=============

The :mod:`panther.tuner` module provides automatic hyperparameter optimization for sketching parameters using Bayesian optimization.

.. currentmodule:: panther.tuner

.. note::
   The AutoTuner module is experimental and requires additional dependencies. Install with: ``pip install botorch``

AutoTuner Class
---------------

.. automodule:: panther.tuner.SkAutoTuner
   :members:
   :special-members: __init__
   :no-index:

Examples
--------

**Basic AutoTuning for Linear Layers**

.. code-block:: python

   import torch
   import torch.nn as nn
   import panther as pr
   from panther.tuner import SkAutoTuner
   
   # Define your model architecture
   def create_model(num_terms, low_rank):
       return nn.Sequential(
           pr.nn.SKLinear(784, 512, num_terms=num_terms, low_rank=low_rank),
           nn.ReLU(),
           pr.nn.SKLinear(512, 256, num_terms=num_terms, low_rank=low_rank//2),
           nn.ReLU(),
           pr.nn.SKLinear(256, 10, num_terms=num_terms//2, low_rank=low_rank//4)
       )
   
   # Define evaluation function
   def evaluate_model(num_terms, low_rank):
       model = create_model(int(num_terms), int(low_rank))
       
       # Your training/validation logic here
       train_loader = get_train_loader()  # Your data loader
       val_loader = get_val_loader()      # Your validation loader
       
       # Train for a few epochs
       optimizer = torch.optim.Adam(model.parameters())
       criterion = nn.CrossEntropyLoss()
       
       model.train()
       for epoch in range(3):  # Quick training
           for batch_idx, (data, target) in enumerate(train_loader):
               optimizer.zero_grad()
               output = model(data)
               loss = criterion(output, target)
               loss.backward()
               optimizer.step()
               
               if batch_idx > 50:  # Limit batches for speed
                   break
       
       # Evaluate
       model.eval()
       correct = 0
       total = 0
       with torch.no_grad():
           for data, target in val_loader:
               output = model(data)
               _, predicted = torch.max(output.data, 1)
               total += target.size(0)
               correct += (predicted == target).sum().item()
               
               if total > 1000:  # Limit evaluation for speed
                   break
       
       accuracy = correct / total
       return accuracy
   
   # Create AutoTuner
   tuner = SkAutoTuner(
       parameter_bounds={
           'num_terms': (2, 16),      # Search between 2 and 16 terms
           'low_rank': (16, 128)      # Search between 16 and 128 rank
       },
       objective_function=evaluate_model,
       n_initial_points=5,            # Start with 5 random evaluations
       n_iterations=20               # Then do 20 Bayesian optimization steps
   )
   
   # Run optimization
   best_params, best_score = tuner.optimize()
   
   print(f"Best parameters: {best_params}")
   print(f"Best accuracy: {best_score:.4f}")

**Advanced AutoTuning with Memory Constraints**

.. code-block:: python

   import psutil
   import torch
   from panther.tuner import SkAutoTuner
   
   def evaluate_with_constraints(num_terms, low_rank, max_memory_gb=8):
       # Check if parameters would exceed memory limit
       estimated_params = 2 * num_terms * low_rank * (784 + 512 + 512 + 256)
       estimated_memory_gb = estimated_params * 4 / (1024**3)  # 4 bytes per float32
       
       if estimated_memory_gb > max_memory_gb:
           return 0.0  # Penalize configurations that use too much memory
       
       # Create and evaluate model
       model = create_model(int(num_terms), int(low_rank))
       
       # Monitor actual memory usage
       process = psutil.Process()
       initial_memory = process.memory_info().rss / (1024**3)
       
       try:
           # Your training code here
           accuracy = train_and_evaluate(model)
           
           # Check memory usage
           peak_memory = process.memory_info().rss / (1024**3)
           memory_used = peak_memory - initial_memory
           
           if memory_used > max_memory_gb:
               return 0.0  # Penalize if actual memory exceeded limit
           
           # Optionally weight accuracy by memory efficiency
           memory_efficiency = max_memory_gb / (memory_used + 0.1)
           return accuracy * min(memory_efficiency, 1.0)
           
       except torch.cuda.OutOfMemoryError:
           return 0.0  # Penalize OOM configurations
   
   # AutoTune with memory constraints
   tuner = SkAutoTuner(
       parameter_bounds={
           'num_terms': (1, 32),
           'low_rank': (8, 256)
       },
       objective_function=lambda nt, lr: evaluate_with_constraints(nt, lr, max_memory_gb=4),
       n_initial_points=10,
       n_iterations=30
   )
   
   best_params, best_score = tuner.optimize()

**Multi-Layer AutoTuning**

.. code-block:: python

   def evaluate_multilayer_config(layer1_terms, layer1_rank, 
                                 layer2_terms, layer2_rank,
                                 layer3_terms, layer3_rank):
       """Tune each layer independently."""
       
       model = nn.Sequential(
           pr.nn.SKLinear(784, 512, 
                         num_terms=int(layer1_terms), 
                         low_rank=int(layer1_rank)),
           nn.ReLU(),
           pr.nn.SKLinear(512, 256,
                         num_terms=int(layer2_terms), 
                         low_rank=int(layer2_rank)),
           nn.ReLU(),
           pr.nn.SKLinear(256, 10,
                         num_terms=int(layer3_terms), 
                         low_rank=int(layer3_rank))
       )
       
       return train_and_evaluate(model)
   
   # Tune multiple layers
   tuner = SkAutoTuner(
       parameter_bounds={
           'layer1_terms': (2, 16), 'layer1_rank': (32, 128),
           'layer2_terms': (2, 12), 'layer2_rank': (16, 64),
           'layer3_terms': (1, 8),  'layer3_rank': (8, 32)
       },
       objective_function=evaluate_multilayer_config,
       n_initial_points=15,  # More initial points for higher-dimensional space
       n_iterations=50
   )
   
   best_params, best_score = tuner.optimize()

**AutoTuning with Custom Objectives**

.. code-block:: python

   def multi_objective_evaluation(num_terms, low_rank):
       """Optimize for both accuracy and speed."""
       
       model = create_model(int(num_terms), int(low_rank))
       
       # Measure accuracy
       accuracy = train_and_evaluate(model)
       
       # Measure inference speed
       model.eval()
       dummy_input = torch.randn(32, 784)
       
       import time
       start_time = time.time()
       for _ in range(100):
           with torch.no_grad():
               _ = model(dummy_input)
       inference_time = (time.time() - start_time) / 100
       
       # Combine accuracy and speed (prefer faster models)
       speed_score = 1.0 / (inference_time + 0.001)  # Higher is better
       
       # Weighted combination
       combined_score = 0.7 * accuracy + 0.3 * min(speed_score / 1000, 1.0)
       
       return combined_score
   
   # Optimize for accuracy and speed
   tuner = SkAutoTuner(
       parameter_bounds={
           'num_terms': (1, 20),
           'low_rank': (8, 256)
       },
       objective_function=multi_objective_evaluation,
       n_initial_points=8,
       n_iterations=25
   )
   
   best_params, best_score = tuner.optimize()

Integration with Training Pipelines
------------------------------------

**PyTorch Lightning Integration**

.. code-block:: python

   import pytorch_lightning as pl
   from panther.tuner import SkAutoTuner
   
   class SketchedLightningModule(pl.LightningModule):
       def __init__(self, num_terms=8, low_rank=64):
           super().__init__()
           self.model = nn.Sequential(
               pr.nn.SKLinear(784, 512, num_terms=num_terms, low_rank=low_rank),
               nn.ReLU(),
               pr.nn.SKLinear(512, 10, num_terms=num_terms//2, low_rank=low_rank//2)
           )
           self.criterion = nn.CrossEntropyLoss()
       
       def training_step(self, batch, batch_idx):
           x, y = batch
           y_hat = self.model(x)
           loss = self.criterion(y_hat, y)
           return loss
       
       def validation_step(self, batch, batch_idx):
           x, y = batch
           y_hat = self.model(x)
           loss = self.criterion(y_hat, y)
           acc = (y_hat.argmax(dim=1) == y).float().mean()
           self.log('val_acc', acc)
           return loss
       
       def configure_optimizers(self):
           return torch.optim.Adam(self.parameters())
   
   def evaluate_lightning(num_terms, low_rank):
       model = SketchedLightningModule(int(num_terms), int(low_rank))
       trainer = pl.Trainer(max_epochs=3, enable_progress_bar=False, 
                           enable_model_summary=False, logger=False)
       
       # Your data modules
       train_loader = get_train_loader()
       val_loader = get_val_loader()
       
       trainer.fit(model, train_loader, val_loader)
       
       # Return validation accuracy
       return trainer.callback_metrics['val_acc'].item()
   
   # AutoTune with PyTorch Lightning
   tuner = SkAutoTuner(
       parameter_bounds={'num_terms': (2, 16), 'low_rank': (16, 128)},
       objective_function=evaluate_lightning,
       n_initial_points=5,
       n_iterations=15
   )

Hyperparameter Search Strategies
---------------------------------

**Grid Search Alternative**

.. code-block:: python

   # Instead of manual grid search, use intelligent search
   def grid_search_alternative():
       # Manual grid would be:
       # for num_terms in [2, 4, 8, 16]:
       #     for low_rank in [16, 32, 64, 128]:
       #         # 16 evaluations
       
       # AutoTuner does this more efficiently:
       tuner = SkAutoTuner(
           parameter_bounds={
               'num_terms': (2, 16),
               'low_rank': (16, 128)
           },
           objective_function=evaluate_model,
           n_initial_points=4,   # Covers parameter space
           n_iterations=8        # Focuses on promising regions
       )
       # Total: 12 evaluations vs 16 for grid search
       return tuner.optimize()

**Random Search Alternative**

.. code-block:: python

   # More efficient than random search
   def random_search_alternative():
       # Random search would evaluate random points
       # AutoTuner uses information from previous evaluations
       
       tuner = SkAutoTuner(
           parameter_bounds={
               'num_terms': (1, 32),
               'low_rank': (8, 256)
           },
           objective_function=evaluate_model,
           n_initial_points=10,  # Random initialization
           n_iterations=40       # Guided search
       )
       return tuner.optimize()

Best Practices
--------------

**1. Start with wide bounds, then narrow**

.. code-block:: python

   # Phase 1: Explore wide range
   wide_tuner = SkAutoTuner(
       parameter_bounds={
           'num_terms': (1, 32),
           'low_rank': (8, 512)
       },
       objective_function=evaluate_model,
       n_initial_points=10,
       n_iterations=20
   )
   best_params_wide, _ = wide_tuner.optimize()
   
   # Phase 2: Refine around best region
   narrow_tuner = SkAutoTuner(
       parameter_bounds={
           'num_terms': (max(1, best_params_wide['num_terms'] - 4),
                        best_params_wide['num_terms'] + 4),
           'low_rank': (max(8, best_params_wide['low_rank'] - 32),
                       best_params_wide['low_rank'] + 32)
       },
       objective_function=evaluate_model,
       n_initial_points=5,
       n_iterations=15
   )
   best_params_refined, best_score = narrow_tuner.optimize()

**2. Use early stopping for faster iterations**

.. code-block:: python

   def fast_evaluate_model(num_terms, low_rank):
       model = create_model(int(num_terms), int(low_rank))
       
       # Train with early stopping
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

**3. Cache expensive computations**

.. code-block:: python

   import functools
   
   @functools.lru_cache(maxsize=128)
   def cached_evaluate_model(num_terms, low_rank):
       # This will cache results for repeated parameter combinations
       return evaluate_model(num_terms, low_rank)
   
   tuner = SkAutoTuner(
       parameter_bounds={'num_terms': (2, 16), 'low_rank': (16, 128)},
       objective_function=cached_evaluate_model,
       n_initial_points=5,
       n_iterations=20
   )
