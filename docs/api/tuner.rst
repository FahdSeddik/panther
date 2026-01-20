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