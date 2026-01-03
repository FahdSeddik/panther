AutoTuner API
=============

The :mod:`panther.tuner` module provides automatic hyperparameter optimization for sketching parameters using `Optuna <https://optuna.org/>`_, a modern hyperparameter optimization framework.

.. currentmodule:: panther.tuner

.. note::
   The AutoTuner uses Optuna by default. Optional dependencies for visualization:
   ``pip install matplotlib pandas``

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

Parameter Specification Types
-----------------------------

The tuner supports modern parameter specification types for flexible search spaces:

- ``Categorical(choices)``: Discrete choices from a list
- ``Int(low, high, step=1)``: Integer range with optional step
- ``Float(low, high, log=False)``: Continuous float range

Examples
--------

**Basic AutoTuning with Optuna (Recommended)**

.. code-block:: python

   import torch
   import torch.nn as nn
   from panther.tuner import SKAutoTuner, LayerConfig, TuningConfigs, Int, Categorical
   
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
           for data, target in val_loader:
               output = model(data)
               _, predicted = torch.max(output.data, 1)
               total += target.size(0)
               correct += (predicted == target).sum().item()
       
       return correct / total
   
   # Configure which layers to tune using ParamSpec types
   config = LayerConfig(
       layer_names=["fc1", "fc2"],
       params={
           "num_terms": Categorical([2, 4, 8]),
           "low_rank": Int(32, 128, step=16)
       },
       separate=True,
       copy_weights=True
   )
   
   tuning_configs = TuningConfigs(configs=[config])
   
   # Create AutoTuner (uses OptunaSearch by default with TPE sampler)
   tuner = SKAutoTuner(
       model=model,
       configs=tuning_configs,
       accuracy_eval_func=accuracy_eval_func,
       accuracy_threshold=0.90,  # Minimum acceptable accuracy
       verbose=True
   )
   
   # Run tuning
   tuner.tune()
   
   # Get and apply best parameters
   best_params = tuner.get_best_params()
   optimized_model = tuner.apply_best_params()
   
   print(f"Best parameters: {best_params}")

Using Different Search Samplers
-------------------------------

The AutoTuner uses ``OptunaSearch`` by default with the TPE sampler. You can customize the search strategy by using different Optuna samplers:

**Grid Search (exhaustive)**

.. code-block:: python

   from panther.tuner import SKAutoTuner, OptunaSearch
   from optuna.samplers import GridSampler
   
   # Define the full search space for grid search
   search_space = {
       "num_terms": [2, 4, 8],
       "low_rank": [32, 64, 128]
   }
   
   tuner = SKAutoTuner(
       model=model,
       configs=tuning_configs,
       accuracy_eval_func=accuracy_eval_func,
       search_algorithm=OptunaSearch(
           sampler=GridSampler(search_space)
       )
   )

**Random Search**

.. code-block:: python

   from panther.tuner import SKAutoTuner, OptunaSearch
   from optuna.samplers import RandomSampler
   
   tuner = SKAutoTuner(
       model=model,
       configs=tuning_configs,
       accuracy_eval_func=accuracy_eval_func,
       search_algorithm=OptunaSearch(
           n_trials=50,
           sampler=RandomSampler(seed=42)
       )
   )

**CMA-ES (for continuous parameters)**

.. code-block:: python

   from panther.tuner import SKAutoTuner, OptunaSearch
   from optuna.samplers import CmaEsSampler
   
   tuner = SKAutoTuner(
       model=model,
       configs=tuning_configs,
       accuracy_eval_func=accuracy_eval_func,
       search_algorithm=OptunaSearch(
           n_trials=100,
           sampler=CmaEsSampler(seed=42)
       )
   )