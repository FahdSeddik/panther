AutoTuner API
=============

The :mod:`panther.tuner` module provides automatic hyperparameter optimization for sketching parameters using Bayesian optimization.

.. currentmodule:: panther.tuner

.. note::
   The AutoTuner module is experimental and requires additional dependencies. Install with: ``pip install botorch``

AutoTuner Class
---------------

.. autoclass:: SKAutoTuner
   :members:
   :special-members: __init__
   :no-index:

Configuration Classes
---------------------

.. autoclass:: LayerConfig
   :members:
   :special-members: __init__
   :no-index:

.. autoclass:: TuningConfigs
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
   from panther.tuner.SkAutoTuner import SKAutoTuner, LayerConfig, TuningConfigs, GridSearch
   
   # Define your model with regular PyTorch layers
   class SimpleModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.fc1 = nn.Linear(784, 512)
           self.relu1 = nn.ReLU()
           self.fc2 = nn.Linear(512, 256)
           self.relu2 = nn.ReLU()
           self.fc3 = nn.Linear(256, 10)
       
       def forward(self, x):
           x = x.view(-1, 784)
           x = self.relu1(self.fc1(x))
           x = self.relu2(self.fc2(x))
           x = self.fc3(x)
           return x
   
   model = SimpleModel()
   
   # Define evaluation function - assesses model performance
   def accuracy_eval_func(model):
       model.eval()
       correct = 0
       total = 0
       
       with torch.no_grad():
           # Use your validation data here
           for data, target in val_loader:
               output = model(data)
               _, predicted = torch.max(output.data, 1)
               total += target.size(0)
               correct += (predicted == target).sum().item()
       
       accuracy = correct / total
       return accuracy
   
   # Configure which layers to tune
   config1 = LayerConfig(
       layer_names=["fc1"],
       params={
           "num_terms": [2, 4, 8],
           "low_rank": [32, 64, 128]
       },
       separate=True,
       copy_weights=True
   )
   config2 = LayerConfig(
       layer_names=["fc2"],
       params={
           "num_terms": [2, 4],
           "low_rank": [32, 64]
       },
       separate=True,
       copy_weights=True
   )
   
   tuning_configs = TuningConfigs(configs=[config1, config2])
   
   # Create AutoTuner
   tuner = SKAutoTuner(
       model=model,
       configs=tuning_configs,
       accuracy_eval_func=accuracy_eval_func,
       search_algorithm=GridSearch(),
       verbose=True
   )
   
   # Run tuning
   tuner.tune()
   
   # Get and apply best parameters
   best_params = tuner.get_best_params()
   optimized_model = tuner.apply_best_params()
   
   print(f"Best parameters: {best_params}")

Using Different Search Algorithms
----------------------------------

