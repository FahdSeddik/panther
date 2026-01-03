AutoTuner API
=============

The :mod:`panther.tuner` module provides automatic hyperparameter optimization for sketching parameters using `Optuna <https://optuna.org/>`_, a modern hyperparameter optimization framework.

.. currentmodule:: panther.tuner

Optional Dependencies
---------------------

+---------------+----------------------------+---------------------------------------+
| Package       | Install Command            | Required For                          |
+===============+============================+=======================================+
| pandas        | ``pip install pandas``     | ``get_results_dataframe()``           |
+---------------+----------------------------+---------------------------------------+

.. tip::
   For tuning result visualization, use Optuna's built-in plots on ``tuner.search_algorithm.study``.
   See the :ref:`external-tools` section for examples.

Public Exports
--------------

All recommended imports are available directly from ``panther.tuner``:

.. code-block:: python

   from panther.tuner import (
       # Core tuner
       SKAutoTuner,
       # Configuration
       LayerConfig,
       TuningConfigs,
       # Parameter specification types
       Categorical,
       Int,
       Float,
       # Search algorithms
       SearchAlgorithm,
       OptunaSearch,
       # Visualization (optional)
       ModelVisualizer,
   )

For advanced/internal APIs not re-exported at the top level, use:

.. code-block:: python

   from panther.tuner.SkAutoTuner.Configs import LayerNameResolver, ParamsResolver

SKAutoTuner Class
-----------------

.. py:class:: SKAutoTuner(model, configs, accuracy_eval_func, search_algorithm=None, verbose=False, accuracy_threshold=None, optmization_eval_func=None, num_runs_per_param=1)

   Auto-tuner for sketched neural network layers.

   :param model: The neural network model to tune (``nn.Module``)
   :param configs: Configuration for the layers to tune (``TuningConfigs``)
   :param accuracy_eval_func: Evaluation function that takes a model and returns an accuracy score (higher is better)
   :param search_algorithm: Search algorithm to use (default: ``OptunaSearch()``)
   :param verbose: Whether to print progress during tuning
   :param accuracy_threshold: Minimum acceptable accuracy. If set along with ``optmization_eval_func``, the tuner maximizes speed while maintaining accuracy ≥ threshold
   :param optmization_eval_func: Function to maximize (e.g., throughput) after reaching the accuracy threshold
   :param num_runs_per_param: Number of runs per parameter combination (for averaging noisy evaluations)

   .. note::
      The parameter is spelled ``optmization_eval_func`` (not "optimization") in the current implementation.

Tuning Workflow Methods
~~~~~~~~~~~~~~~~~~~~~~~

.. py:method:: SKAutoTuner.tune() -> Dict[str, Dict[str, Any]]

   Run the tuning process for all configured layer groups.

   :returns: Dictionary mapping layer names to their best parameters

.. py:method:: SKAutoTuner.apply_best_params() -> nn.Module

   Apply the best found parameters to the model, replacing layers with their sketched versions.

   :returns: The model with optimized sketched layers

.. py:method:: SKAutoTuner.replace_without_tuning() -> nn.Module

   Replace layers with sketched versions using the first parameter value from each specification, without running any tuning trials. Useful for quick testing.

   :returns: The model with layers replaced

Results and Analysis Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:method:: SKAutoTuner.get_best_params() -> Dict[str, Dict[str, Any]]

   Get the best parameters found for each layer.

   :returns: Dictionary mapping layer names to their best parameter configurations

.. py:method:: SKAutoTuner.get_results_dataframe() -> pandas.DataFrame

   Get all tuning results as a pandas DataFrame.

   :returns: DataFrame with columns: ``layer_name``, parameter columns, ``score``, ``accuracy``, ``speed``
   :raises ImportError: If pandas is not installed

   .. code-block:: python

      results_df = tuner.get_results_dataframe()
      print(results_df[["layer_name", "num_terms", "low_rank", "accuracy", "speed", "score"]])

Persistence Methods
~~~~~~~~~~~~~~~~~~~

.. py:method:: SKAutoTuner.save_tuning_results(file_path: str)

   Save tuning results to a pickle file.

   :param file_path: Path to save the results

.. py:method:: SKAutoTuner.load_tuning_results(file_path: str)

   Load tuning results from a pickle file.

   :param file_path: Path to load results from
   :raises FileNotFoundError: If the file does not exist
   :raises ValueError: If the file contains invalid data

Configuration Accessors
~~~~~~~~~~~~~~~~~~~~~~~

.. py:method:: SKAutoTuner.getResolver() -> LayerNameResolver

   Get the layer name resolver for advanced layer inspection.

.. py:method:: SKAutoTuner.getConfigs() -> TuningConfigs

   Get the current (resolved) tuning configurations.

.. py:method:: SKAutoTuner.setConfigs(configs: TuningConfigs)

   Set new tuning configurations. Automatically resolves layer names and parameters.

Configuration Classes
---------------------

LayerConfig
~~~~~~~~~~~

.. py:class:: LayerConfig(layer_names, params, separate=True, copy_weights=True)

   Configuration for a single layer or group of layers to tune.

   :param layer_names: Layer selector (see :ref:`layer-selectors`)
   :param params: Dictionary of parameter specifications (see :ref:`param-specs`)
   :param separate: If ``True``, tune each layer independently. If ``False``, tune all layers jointly with shared parameters
   :param copy_weights: Whether to copy weights from original layers when creating sketched versions

   .. code-block:: python

      from panther.tuner import LayerConfig, Categorical, Int

      # Tune fc1 and fc2 separately with the same parameter space
      config = LayerConfig(
          layer_names=["fc1", "fc2"],
          params={
              "num_terms": Categorical([1, 2, 3]),
              "low_rank": Int(16, 128, step=16)
          },
          separate=True,
          copy_weights=True
      )

TuningConfigs
~~~~~~~~~~~~~

.. py:class:: TuningConfigs(configs: List[LayerConfig])

   Collection of ``LayerConfig`` objects for tuning multiple layer groups.

   :param configs: List of ``LayerConfig`` objects

   **Container Methods:**

   - ``__len__()`` - Number of configs
   - ``__getitem__(index)`` - Get config by index
   - ``__iter__()`` - Iterate over configs
   - ``add(config)`` - Add a config (returns new ``TuningConfigs``)
   - ``remove(index)`` - Remove config at index (returns new ``TuningConfigs``)
   - ``replace(index, config)`` - Replace config at index (returns new ``TuningConfigs``)
   - ``clone()`` - Deep copy
   - ``merge(other)`` - Combine with another ``TuningConfigs``
   - ``filter(predicate)`` - Filter configs by predicate function
   - ``map(transform)`` - Transform each config

   .. code-block:: python

      from panther.tuner import TuningConfigs, LayerConfig, Categorical

      configs = TuningConfigs([
          LayerConfig(
              layer_names=["encoder.*"],
              params={"num_terms": Categorical([1, 2, 3])},
          ),
          LayerConfig(
              layer_names=["decoder.*"],
              params={"num_terms": Categorical([2, 4, 8])},
          ),
      ])

      # Access individual configs
      print(len(configs))  # 2
      print(configs[0])    # First LayerConfig

.. _layer-selectors:

Layer Selectors
---------------

The ``layer_names`` parameter in ``LayerConfig`` supports multiple selector formats for flexible layer targeting.

String Pattern
~~~~~~~~~~~~~~

A single string is interpreted as a regex pattern or substring:

.. code-block:: python

   # Match all layers containing "encoder"
   LayerConfig(layer_names="encoder", ...)

   # Match layers like "encoder.layer0.attention", "encoder.layer1.attention"
   LayerConfig(layer_names="encoder.*attention", ...)

List of Patterns
~~~~~~~~~~~~~~~~

A list of strings matches multiple patterns:

.. code-block:: python

   # Match layers containing "encoder" OR "decoder"
   LayerConfig(layer_names=["encoder", "decoder"], ...)

   # Match specific layer names exactly
   LayerConfig(layer_names=["fc1", "fc2", "fc3"], ...)

Dictionary Selector
~~~~~~~~~~~~~~~~~~~

A dictionary enables advanced selection with multiple criteria:

.. code-block:: python

   # Select by regex pattern
   LayerConfig(
       layer_names={"pattern": "encoder\\.layer[0-5]\\..*"},
       ...
   )

   # Select by layer type
   LayerConfig(
       layer_names={"type": "Linear"},  # All nn.Linear layers
       ...
   )

   # Multiple types
   LayerConfig(
       layer_names={"type": ["Conv2d", "ConvTranspose2d"]},
       ...
   )

   # Select by substring
   LayerConfig(
       layer_names={"contains": "attention"},
       ...
   )

   # Select specific indices from matched layers
   LayerConfig(
       layer_names={
           "pattern": "encoder.*",
           "indices": [0, 2, 4]  # First, third, and fifth matched layers
       },
       ...
   )

   # Select a range of layers
   LayerConfig(
       layer_names={
           "type": "Linear",
           "range": [0, 6]      # First 6 matched Linear layers
       },
       ...
   )

   # Range with step
   LayerConfig(
       layer_names={
           "type": "Linear",
           "range": [0, 12, 2]  # Every other layer from first 12
       },
       ...
   )

   # Combine multiple criteria (intersection)
   LayerConfig(
       layer_names={
           "pattern": "encoder.*",
           "type": "Linear",
           "indices": [0, 1, 2]
       },
       ...
   )

**Selector Keys:**

+---------------+----------------------------------+-------------------------------------------+
| Key           | Type                             | Description                               |
+===============+==================================+===========================================+
| ``pattern``   | ``str`` or ``List[str]``         | Regex patterns to match layer names       |
+---------------+----------------------------------+-------------------------------------------+
| ``type``      | ``str`` or ``List[str]``         | Layer type names (e.g., ``"Linear"``)     |
+---------------+----------------------------------+-------------------------------------------+
| ``contains``  | ``str`` or ``List[str]``         | Substrings that layer names must contain  |
+---------------+----------------------------------+-------------------------------------------+
| ``indices``   | ``int`` or ``List[int]``         | Specific indices from matched layers      |
+---------------+----------------------------------+-------------------------------------------+
| ``range``     | ``[start, end]`` or              | Range of indices (exclusive end)          |
|               | ``[start, end, step]``           |                                           |
+---------------+----------------------------------+-------------------------------------------+

.. note::
   ``indices`` and ``range`` cannot be used together. Multiple criteria (pattern, type, contains) are combined with AND logic (intersection).

.. _param-specs:

Parameter Specification Types
-----------------------------

Modern first-class parameter specifications for expressing search spaces:

Categorical
~~~~~~~~~~~

.. py:class:: Categorical(choices: Sequence[Any])

   A categorical parameter that takes values from a fixed set of choices.

   :param choices: Sequence of possible values (any type)
   :raises ValueError: If choices is empty

   .. code-block:: python

      from panther.tuner import Categorical

      # Integer choices
      Categorical([1, 2, 3, 4])

      # String choices
      Categorical(["relu", "gelu", "silu"])

      # Boolean choices
      Categorical([True, False])

Int
~~~

.. py:class:: Int(low: int, high: int, step: int = 1, log: bool = False)

   An integer parameter within a range ``[low, high]``.

   :param low: Lower bound (inclusive)
   :param high: Upper bound (inclusive)
   :param step: Step size for discrete values (default: 1)
   :param log: Whether to sample in log scale
   :raises ValueError: If low > high, step < 1, or log=True with low ≤ 0

   .. code-block:: python

      from panther.tuner import Int

      # Integer from 1 to 100
      Int(1, 100)

      # Multiples of 8 from 8 to 512
      Int(8, 512, step=8)

      # Log-scale integer sampling
      Int(1, 1000, log=True)

Float
~~~~~

.. py:class:: Float(low: float, high: float, step: float = None, log: bool = False)

   A floating-point parameter within a range ``[low, high]``.

   :param low: Lower bound (inclusive)
   :param high: Upper bound (inclusive)
   :param step: Step size for discrete values (``None`` for continuous)
   :param log: Whether to sample in log scale
   :raises ValueError: If low > high, step ≤ 0, or log=True with low ≤ 0

   .. code-block:: python

      from panther.tuner import Float

      # Continuous float from 0 to 1
      Float(0.0, 1.0)

      # Log-scale float (e.g., learning rate)
      Float(1e-5, 1e-1, log=True)

      # Discrete float values
      Float(0.1, 1.0, step=0.1)

Legacy List Format
~~~~~~~~~~~~~~~~~~

For backward compatibility, plain lists are also supported and are interpreted as ``Categorical``:

.. code-block:: python

   # Legacy format (still works)
   params = {
       "num_terms": [1, 2, 3],
       "low_rank": [8, 16, 32, 64],
   }

   # Equivalent modern format
   params = {
       "num_terms": Categorical([1, 2, 3]),
       "low_rank": Categorical([8, 16, 32, 64]),
   }

.. _auto-params:

Automatic Parameter Generation
------------------------------

The special value ``"auto"`` for the ``params`` field enables automatic parameter space generation based on layer dimensions:

.. code-block:: python

   from panther.tuner import LayerConfig, TuningConfigs

   config = LayerConfig(
       layer_names={"type": "Linear"},
       params="auto",  # Automatically determine parameter ranges
       separate=True
   )

How "auto" Works
~~~~~~~~~~~~~~~~

When ``params="auto"`` is specified, the tuner intelligently analyzes your model and generates optimal parameter search spaces:

- **Smart Grouping**: Layers are automatically grouped by type, shape, and size to enable efficient tuning
- **Efficiency-Aware**: Parameter ranges are computed using layer-specific efficiency equations that consider the mathematical properties of each layer type
- **Adaptive Ranges**: The generated Parameter values are tailored to each layer's dimensions to ensure only efficient configurations are explored
- **Layer-Specific Logic**: Different layer types (Linear, Conv2d, etc.) use their own specialized parameter generation strategies

The automatic parameter generation considers:

- Layer type and its specific implementation details
- Input/output dimensions and their relationships  
- Theoretical efficiency bounds for sketched approximations
- Practical constraints to avoid degenerate configurations

.. code-block:: python

   # Let the tuner figure out the best parameter ranges
   config = LayerConfig(
       layer_names={"type": "Linear"},
       params="auto",
       separate=True
   )

   # The tuner will:
   # 1. Analyze each layer's dimensions
   # 2. Group similar layers together
   # 3. Generate efficient parameter ranges for each group
   # 4. Create appropriate LayerConfigs automatically

.. note::
   ``"auto"`` currently supports ``nn.Linear`` and ``nn.Conv2d`` layers. Unsupported layer types are skipped with a warning if ``verbose=True``.

Search Algorithms
-----------------

SearchAlgorithm Interface
~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: SearchAlgorithm

   Abstract base class for search algorithms. Implement this interface to create custom search strategies.

   **Required Methods:**

   .. py:method:: initialize(param_space: Dict[str, Any])

      Initialize with the parameter space.

   .. py:method:: get_next_params() -> Optional[Dict[str, Any]]

      Get the next parameters to try, or ``None`` if finished.

   .. py:method:: update(params: Dict[str, Any], score: float)

      Update with trial results.

   .. py:method:: save_state(filepath: str)

      Save algorithm state to file.

   .. py:method:: load_state(filepath: str)

      Load algorithm state from file.

   .. py:method:: get_best_params() -> Optional[Dict[str, Any]]

      Get the best parameters found.

   .. py:method:: get_best_score() -> Optional[float]

      Get the best score achieved.

   .. py:method:: reset()

      Reset to initial state.

   .. py:method:: is_finished() -> bool

      Check if search is complete.

   **Optional Methods:**

   .. py:method:: get_resource() -> Any

      Return a resource value for resource-aware algorithms (e.g., Hyperband). The ``SKAutoTuner`` will pass this to evaluation functions if present.

OptunaSearch
~~~~~~~~~~~~

.. py:class:: OptunaSearch(n_trials=100, sampler=None, direction="maximize", study_name=None, storage=None, load_if_exists=False, seed=None)

   Optuna-backed search algorithm with state-of-the-art samplers.

   :param n_trials: Maximum number of trials (default: 100)
   :param sampler: Optuna sampler (default: ``TPESampler``)
   :param direction: ``"maximize"`` or ``"minimize"`` (default: ``"maximize"``)
   :param study_name: Name for the Optuna study (useful for persistence)
   :param storage: Optuna storage URL (e.g., ``"sqlite:///study.db"``) for persistence
   :param load_if_exists: Whether to resume an existing study from storage
   :param seed: Random seed for reproducibility

   **Properties:**

   .. py:attribute:: study

      Access the underlying ``optuna.Study`` for advanced operations.

   **Additional Methods:**

   .. py:method:: get_trials_dataframe() -> pandas.DataFrame

      Get a DataFrame of all trials (requires pandas).

   .. py:method:: set_user_attr(key: str, value: Any)

      Set a user attribute on the current pending trial.

   .. py:method:: report_intermediate(value: float, step: int)

      Report intermediate objective value for pruning.

   .. py:method:: should_prune() -> bool

      Check if current trial should be pruned.

Search Sampler Examples
~~~~~~~~~~~~~~~~~~~~~~~

**TPE Sampler (Default)**

Tree-structured Parzen Estimator - excellent for most use cases:

.. code-block:: python

   from panther.tuner import SKAutoTuner, OptunaSearch

   tuner = SKAutoTuner(
       model=model,
       configs=tuning_configs,
       accuracy_eval_func=accuracy_eval_func,
       search_algorithm=OptunaSearch(n_trials=100, seed=42),
   )

**Random Search**

Simple random sampling:

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
       ),
   )

**Grid Search**

Exhaustive search over all combinations (use for small parameter spaces):

.. code-block:: python

   from panther.tuner import SKAutoTuner, OptunaSearch
   from optuna.samplers import GridSampler

   # Define the full search space for GridSampler
   search_space = {
       "num_terms": [1, 2, 3],
       "low_rank": [8, 16, 32, 64],
   }

   tuner = SKAutoTuner(
       model=model,
       configs=tuning_configs,
       accuracy_eval_func=accuracy_eval_func,
       search_algorithm=OptunaSearch(
           sampler=GridSampler(search_space)
       ),
   )

**CMA-ES**

Covariance Matrix Adaptation Evolution Strategy - excellent for continuous parameters:

.. code-block:: python

   from panther.tuner import SKAutoTuner, OptunaSearch
   from optuna.samplers import CmaEsSampler

   tuner = SKAutoTuner(
       model=model,
       configs=tuning_configs,
       accuracy_eval_func=accuracy_eval_func,
       search_algorithm=OptunaSearch(
           n_trials=200,
           sampler=CmaEsSampler(seed=42)
       ),
   )

Study Persistence
~~~~~~~~~~~~~~~~~

For long-running tuning jobs, persist the Optuna study to a database:

.. code-block:: python

   from panther.tuner import SKAutoTuner, OptunaSearch

   # Create tuner with SQLite persistence
   tuner = SKAutoTuner(
       model=model,
       configs=tuning_configs,
       accuracy_eval_func=accuracy_eval_func,
       search_algorithm=OptunaSearch(
           n_trials=1000,
           study_name="my_tuning_study",
           storage="sqlite:///tuning_results.db",
           load_if_exists=True  # Resume if study exists
       ),
   )

   # Run tuning (can be interrupted and resumed)
   tuner.tune()

   # Access study for advanced analysis
   study = tuner.search_algorithm.study
   print(study.best_params)
   print(study.best_value)

   # Get trials as DataFrame
   trials_df = tuner.search_algorithm.get_trials_dataframe()

ModelVisualizer
---------------

.. py:class:: ModelVisualizer

   Utility class for discovering layer names in PyTorch models. Use this to craft
   correct layer selectors for ``LayerConfig``. All methods are static.

.. py:staticmethod:: ModelVisualizer.print_module_tree(model: nn.Module, root_name: str = "model")

   Print the module hierarchy as an ASCII tree with layer types.

   :param model: The PyTorch model to inspect
   :param root_name: Display name for the root module

   .. code-block:: python

      from panther.tuner import ModelVisualizer

      ModelVisualizer.print_module_tree(model)
      # Output:
      # model (MyModel)/
      # └─ encoder (Encoder)/
      #     ├─ layer0 (TransformerLayer)/
      #     │   ├─ attention (MultiheadAttention)/
      #     │   └─ fc (Linear)/
      #     └─ layer1 (TransformerLayer)/
      #         ...

   Use the printed layer names directly in your ``LayerConfig``:

   .. code-block:: python

      config = LayerConfig(
          layer_names=["encoder.layer0.fc", "encoder.layer1.fc"],
          params={...}
      )

.. _external-tools:

External Tools for Model Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For detailed model summaries with accurate parameter counts, use `torchinfo <https://github.com/TylerYep/torchinfo>`_:

.. code-block:: bash

   pip install torchinfo

.. code-block:: python

   from torchinfo import summary

   # Get detailed model summary with parameter counts and layer shapes
   summary(model, input_size=(1, 3, 224, 224))

   # Or get summary as a string
   model_stats = summary(model, input_size=(1, 3, 224, 224), verbose=0)
   print(f"Total parameters: {model_stats.total_params:,}")
   print(f"Trainable parameters: {model_stats.trainable_params:,}")

For tuning result visualization, use Optuna's built-in visualization on the study object:

.. code-block:: python

   import optuna

   # After tuning, access the Optuna study
   study = tuner.search_algorithm.study

   # Plot optimization history
   optuna.visualization.plot_optimization_history(study).show()

   # Plot parameter importances
   optuna.visualization.plot_param_importances(study).show()

   # Plot parallel coordinate plot of trials
   optuna.visualization.plot_parallel_coordinate(study).show()

   # Plot slice plot for each parameter
   optuna.visualization.plot_slice(study).show()

   # For Jupyter notebooks, use matplotlib backend
   from optuna.visualization.matplotlib import plot_optimization_history
   plot_optimization_history(study)

.. tip::
   Optuna provides many more visualizations. See the
   `Optuna Visualization Documentation <https://optuna.readthedocs.io/en/stable/reference/visualization/index.html>`_
   for the full list.

Complete Examples
-----------------

Basic Tuning with Accuracy Threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn
   from panther.tuner import (
       SKAutoTuner, LayerConfig, TuningConfigs,
       Categorical, Int, OptunaSearch
   )

   # Define model
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
           return self.fc3(x)

   model = SimpleModel()

   # Define evaluation functions
   def accuracy_eval_func(model):
       model.eval()
       correct = total = 0
       with torch.no_grad():
           for data, target in val_loader:
               output = model(data)
               _, predicted = torch.max(output, 1)
               total += target.size(0)
               correct += (predicted == target).sum().item()
       return correct / total

   def speed_eval_func(model):
       # Measure throughput (higher is better)
       import time
       model.eval()
       x = torch.randn(128, 784)
       start = time.time()
       for _ in range(100):
           with torch.no_grad():
               model(x)
       elapsed = time.time() - start
       return 100 * 128 / elapsed  # samples per second

   # Configure layers to tune
   config = LayerConfig(
       layer_names=["fc1", "fc2"],
       params={
           "num_terms": Categorical([1, 2, 3]),
           "low_rank": Int(16, 128, step=16)
       },
       separate=True,
       copy_weights=True
   )

   tuning_configs = TuningConfigs([config])

   # Create tuner with constraint-based optimization
   tuner = SKAutoTuner(
       model=model,
       configs=tuning_configs,
       accuracy_eval_func=accuracy_eval_func,
       accuracy_threshold=0.90,              # Must maintain 90% accuracy
       optmization_eval_func=speed_eval_func,  # Maximize speed
       search_algorithm=OptunaSearch(n_trials=50, seed=42),
       verbose=True
   )

   # Run tuning
   tuner.tune()

   # Apply best parameters
   optimized_model = tuner.apply_best_params()

   # Analyze results
   results_df = tuner.get_results_dataframe()
   print(results_df[["layer_name", "num_terms", "low_rank", "accuracy", "speed", "score"]])

   # Visualize with Optuna (see External Tools section)
   import optuna
   study = tuner.search_algorithm.study
   optuna.visualization.plot_optimization_history(study).show()

Using Automatic Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from panther.tuner import SKAutoTuner, LayerConfig, TuningConfigs

   # Auto-generate parameter spaces based on layer dimensions
   config = LayerConfig(
       layer_names={"type": "Linear"},  # All Linear layers
       params="auto",                    # Automatic parameter generation
       separate=True
   )

   tuner = SKAutoTuner(
       model=model,
       configs=TuningConfigs([config]),
       accuracy_eval_func=accuracy_eval_func,
       verbose=True
   )

   tuner.tune()

Custom Search Algorithm
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from panther.tuner import SearchAlgorithm
   from typing import Any, Dict, Optional
   import random

   class MyCustomSearch(SearchAlgorithm):
       def __init__(self, n_trials=50):
           self.n_trials = n_trials
           self._param_space = {}
           self._trial_count = 0
           self._best_params = None
           self._best_score = float("-inf")

       def initialize(self, param_space: Dict[str, Any]):
           self._param_space = param_space
           self._trial_count = 0
           self._best_params = None
           self._best_score = float("-inf")

       def get_next_params(self) -> Optional[Dict[str, Any]]:
           if self.is_finished():
               return None
           self._trial_count += 1
           # Custom sampling logic
           params = {}
           for name, choices in self._param_space.items():
               if isinstance(choices, list):
                   params[name] = random.choice(choices)
           return params

       def update(self, params: Dict[str, Any], score: float):
           if score > self._best_score:
               self._best_score = score
               self._best_params = params.copy()

       def get_best_params(self) -> Optional[Dict[str, Any]]:
           return self._best_params

       def get_best_score(self) -> Optional[float]:
           return self._best_score

       def is_finished(self) -> bool:
           return self._trial_count >= self.n_trials

       def reset(self):
           self._trial_count = 0
           self._best_params = None
           self._best_score = float("-inf")

       def save_state(self, filepath: str):
           import pickle
           with open(filepath, "wb") as f:
               pickle.dump({
                   "trial_count": self._trial_count,
                   "best_params": self._best_params,
                   "best_score": self._best_score,
               }, f)

       def load_state(self, filepath: str):
           import pickle
           with open(filepath, "rb") as f:
               state = pickle.load(f)
           self._trial_count = state["trial_count"]
           self._best_params = state["best_params"]
           self._best_score = state["best_score"]

   # Use custom search
   tuner = SKAutoTuner(
       model=model,
       configs=tuning_configs,
       accuracy_eval_func=accuracy_eval_func,
       search_algorithm=MyCustomSearch(n_trials=100),
   )

Joint vs Separate Tuning
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from panther.tuner import LayerConfig, TuningConfigs, Categorical

   # Separate tuning: each layer gets its own optimal parameters
   separate_config = LayerConfig(
       layer_names=["fc1", "fc2", "fc3"],
       params={"num_terms": Categorical([1, 2, 3])},
       separate=True  # Each layer tuned independently
   )

   # Joint tuning: all layers share the same parameters
   joint_config = LayerConfig(
       layer_names=["fc1", "fc2", "fc3"],
       params={"num_terms": Categorical([1, 2, 3])},
       separate=False  # All layers use same parameter values
   )

   # Joint tuning is faster (fewer trials) but may be suboptimal
   # Separate tuning finds optimal per-layer parameters but takes longer