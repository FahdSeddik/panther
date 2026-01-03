Autotuning for Maximum Performance
===================================

This tutorial shows you how to use Panther's AutoTuner to automatically find the best sketching parameters for your neural network, maximizing throughput while maintaining accuracy.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

When sketching neural network layers, choosing the right parameters e.g. (``num_terms``, ``low_rank``) is crucial:

- **Too aggressive**: Fast but inaccurate
- **Too conservative**: Accurate but slow
- **Just right**: Maximum speedup with acceptable accuracy loss

The ``SKAutoTuner`` automates this process using state-of-the-art hyperparameter optimization.

Quick Start
-----------

Here's the fastest way to tune your model:

.. code-block:: python

   import torch.nn as nn
   from panther.tuner import SKAutoTuner, LayerConfig, TuningConfigs

   model = YourModel()

   # Use "auto" for automatic parameter ranges
   config = LayerConfig(
       layer_names={"type": "Linear"},
       params="auto",
       separate=True
   )

   tuner = SKAutoTuner(
       model=model,
       configs=TuningConfigs([config]),
       accuracy_eval_func=your_accuracy_function,
       verbose=True
   )

   tuner.tune()
   optimized_model = tuner.apply_best_params()

Constraint-Based Optimization
-----------------------------

The most powerful feature is **constraint-based optimization**: maximize speed while maintaining a minimum accuracy threshold.

Setting Up Evaluation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need two functions:

1. **Accuracy function**: Returns a score where higher is better (e.g., accuracy, F1)
2. **Speed function**: Returns throughput where higher is better (e.g., samples/second)

.. code-block:: python

   import torch
   import time

   def accuracy_eval_func(model):
       """Evaluate model accuracy on validation set."""
       model.eval()
       correct = total = 0
       
       with torch.no_grad():
           for inputs, targets in val_loader:
               inputs, targets = inputs.cuda(), targets.cuda()
               outputs = model(inputs)
               _, predicted = outputs.max(1)
               total += targets.size(0)
               correct += predicted.eq(targets).sum().item()
       
       return correct / total

   def speed_eval_func(model):
       """Measure inference throughput (samples per second)."""
       model.eval()
       batch_size = 64
       x = torch.randn(batch_size, *input_shape).cuda()
       
       # Warmup
       for _ in range(10):
           with torch.no_grad():
               model(x)
       
       torch.cuda.synchronize()
       
       # Measure
       iterations = 100
       start = time.perf_counter()
       for _ in range(iterations):
           with torch.no_grad():
               model(x)
       torch.cuda.synchronize()
       elapsed = time.perf_counter() - start
       
       return (iterations * batch_size) / elapsed

Running Constrained Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from panther.tuner import (
       SKAutoTuner, LayerConfig, TuningConfigs,
       Categorical, Int, OptunaSearch
   )

   # Define which layers to tune and their parameter search space
   config = LayerConfig(
       layer_names={"type": "Linear"},  # All Linear layers
       params={
           "num_terms": Categorical([1, 2, 3, 4]),
           "low_rank": Int(8, 128, step=8)
       },
       separate=True,      # Tune each layer independently
       copy_weights=True   # Preserve trained weights
   )

   # Create tuner with constraint
   tuner = SKAutoTuner(
       model=model,
       configs=TuningConfigs([config]),
       accuracy_eval_func=accuracy_eval_func,
       accuracy_threshold=0.95,              # Must maintain 95% accuracy
       optmization_eval_func=speed_eval_func,  # Maximize this
       search_algorithm=OptunaSearch(n_trials=100, seed=42),
       verbose=True
   )

   # Run the search
   best_params = tuner.tune()

   # Apply winning configuration
   optimized_model = tuner.apply_best_params()

   print(f"Best parameters found: {best_params}")

Understanding the Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When both ``accuracy_threshold`` and ``optmization_eval_func`` are set:

1. Each trial evaluates accuracy first
2. If accuracy ≥ threshold, the speed score becomes the objective
3. If accuracy < threshold, the trial gets score ``-inf`` (rejected)
4. The tuner maximizes speed among configurations that meet the accuracy bar

Tuning Strategies
-----------------

Strategy 1: Quick Exploration with "auto"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let the tuner figure out good parameter ranges:

.. code-block:: python

   config = LayerConfig(
       layer_names={"type": "Linear"},
       params="auto",  # Intelligent automatic ranges
       separate=True
   )

This is great for initial exploration when you don't know what ranges work.

Strategy 2: Targeted Layer Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Focus tuning on the most impactful layers:

.. code-block:: python

   from panther.tuner import ModelVisualizer

   # First, inspect your model
   ModelVisualizer.print_module_tree(model)

   # Tune only the expensive layers (e.g., large linear layers in transformer)
   config = LayerConfig(
       layer_names={
           "pattern": "encoder\\.layers\\.[0-5]\\..*",
           "type": "Linear"
       },
       params={
           "num_terms": Categorical([1, 2, 3]),
           "low_rank": Int(16, 256, step=16)
       },
       separate=True
   )

Strategy 3: Joint vs Separate Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Separate tuning** (default): Each layer gets its own optimal parameters.

.. code-block:: python

   # More trials, but finds per-layer optimal settings
   config = LayerConfig(
       layer_names=["layer1", "layer2", "layer3"],
       params={"num_terms": Categorical([1, 2, 3])},
       separate=True  # 3 independent searches
   )

**Joint tuning**: All specified layers share the same parameters.

.. code-block:: python

   # Fewer trials, layers use identical settings
   config = LayerConfig(
       layer_names=["layer1", "layer2", "layer3"],
       params={"num_terms": Categorical([1, 2, 3])},
       separate=False  # 1 search, shared parameters
   )

Use joint tuning when:

- Layers are similar (same size, same role)
- You want faster tuning
- Memory constraints require uniform compression

Strategy 4: Progressive Refinement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For very large search spaces, you can tune in stages—first exploring broadly, then refining around the best regions. This is optional since TPE already balances exploration and exploitation, but can be useful when:

- Your search space has many parameters with wide ranges
- You want to quickly eliminate bad regions before investing in fine-tuning
- You're tuning different layer groups and want to lock in some before tuning others

.. code-block:: python

   # Stage 1: Broad exploration with sparse sampling
   coarse_config = LayerConfig(
       layer_names={"type": "Linear"},
       params={
           "num_terms": Categorical([1, 2, 4, 8]),      # Wide range
           "low_rank": Categorical([16, 64, 128, 256])  # Sparse values
       }
   )

   tuner = SKAutoTuner(
       model=model,
       configs=TuningConfigs([coarse_config]),
       accuracy_eval_func=accuracy_eval_func,
       accuracy_threshold=0.95,
       optmization_eval_func=speed_eval_func,
       search_algorithm=OptunaSearch(n_trials=30),  # Quick exploration
   )
   tuner.tune()

   # Analyze: which regions performed best?
   df = tuner.get_results_dataframe()
   print(df.sort_values("score", ascending=False).head(10))
   
   # Suppose we find num_terms=2 and low_rank around 64-128 work best
   
   # Stage 2: Fine-grained search in the promising region
   fine_config = LayerConfig(
       layer_names={"type": "Linear"},
       params={
           "num_terms": Categorical([2, 3]),           # Narrow: around best
           "low_rank": Int(48, 144, step=8)            # Dense sampling in [64-128] region
       }
   )

   # Create fresh tuner with refined ranges
   fine_tuner = SKAutoTuner(
       model=model,  # Use original model, not the one from stage 1
       configs=TuningConfigs([fine_config]),
       accuracy_eval_func=accuracy_eval_func,
       accuracy_threshold=0.95,
       optmization_eval_func=speed_eval_func,
       search_algorithm=OptunaSearch(n_trials=50),  # More trials in smaller space
   )
   fine_tuner.tune()
   
   # Apply the refined best parameters
   optimized_model = fine_tuner.apply_best_params()

.. note::
   For most use cases, a single tuning run with enough trials (50-200) and the default TPE sampler is sufficient. Use multi-stage only when dealing with extremely large search spaces.

Advanced Configuration
----------------------

Using Different Samplers
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler

   # TPE (default) - good general-purpose sampler
   search = OptunaSearch(n_trials=100, sampler=TPESampler(seed=42))

   # CMA-ES - excellent for continuous parameters
   search = OptunaSearch(n_trials=200, sampler=CmaEsSampler(seed=42))

   # Random - useful as a baseline
   search = OptunaSearch(n_trials=50, sampler=RandomSampler(seed=42))

Resumable Tuning with Persistence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For long tuning runs, persist progress to a database:

.. code-block:: python

   tuner = SKAutoTuner(
       model=model,
       configs=TuningConfigs([config]),
       accuracy_eval_func=accuracy_eval_func,
       accuracy_threshold=0.95,
       optmization_eval_func=speed_eval_func,
       search_algorithm=OptunaSearch(
           n_trials=500,
           study_name="my_model_tuning",
           storage="sqlite:///tuning.db",
           load_if_exists=True  # Resume if interrupted
       ),
   )

   # Can be interrupted and resumed
   tuner.tune()

Handling Noisy Evaluations
~~~~~~~~~~~~~~~~~~~~~~~~~~

If your evaluations are noisy (common with randomized/sketched layers, GPU timing variance, or small validation sets), increase ``num_runs_per_param`` to re-evaluate the *same* parameter set multiple times.

.. note::
   ``SKAutoTuner`` does **not** compute an average. For each parameter set it runs ``num_runs_per_param`` evaluations and keeps the **best** (highest) score observed for that parameter set.

.. code-block:: python

   tuner = SKAutoTuner(
       model=model,
       configs=TuningConfigs([config]),
       accuracy_eval_func=accuracy_eval_func,
       optmization_eval_func=speed_eval_func,
       num_runs_per_param=3,  # Re-try each param set; best-of-N is used
       verbose=True
   )

Analyzing Results
-----------------

DataFrame Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd

   # Get all trial results
   df = tuner.get_results_dataframe()

   # Best configurations per layer
   best_per_layer = df.loc[df.groupby("layer_name")["score"].idxmax()]
   print(best_per_layer)

   # Configurations meeting accuracy threshold
   valid = df[df["accuracy"] >= 0.95]
   print(valid.sort_values("speed", ascending=False))

   # Parameter importance (which parameters affect score most)
   for param in ["num_terms", "low_rank"]:
       if param in df.columns:
           print(f"\n{param} vs Score:")
           print(df.groupby(param)["score"].agg(["mean", "std", "count"]))

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   # Built-in visualization
   tuner.visualize_tuning_results(
       save_path="tuning_results.png",
       show_plot=True
   )

   # Or use Optuna's visualization
   import optuna.visualization as vis

   study = tuner.search_algorithm.study
   
   # Parameter importances
   fig = vis.plot_param_importances(study)
   fig.show()
   
   # Optimization history
   fig = vis.plot_optimization_history(study)
   fig.show()
   
   # Parameter relationships
   fig = vis.plot_parallel_coordinate(study)
   fig.show()

Best Practices
--------------

1. **Start with "auto"**: Let the tuner determine reasonable ranges first
2. **Use a representative validation set**: Accuracy evaluation should reflect real performance
3. **Warm up GPU**: Include warmup iterations in speed measurements
4. **Set realistic thresholds**: A 0.99 accuracy threshold might be impossible to meet
5. **Save results**: Use database storage for long tuning runs
6. **Visualize early**: Check results after ~20 trials to catch issues
7. **Consider batch size**: Measure speed at your deployment batch size

Complete Example: Transformer Tuning
------------------------------------

.. code-block:: python

   import torch
   import torch.nn as nn
   import time
   from panther.tuner import (
       SKAutoTuner, LayerConfig, TuningConfigs,
       Categorical, Int, OptunaSearch, ModelVisualizer
   )

   # Load your trained transformer
   model = load_my_transformer()
   model.cuda()
   model.eval()

   # Inspect the model structure
   ModelVisualizer.print_module_tree(model)

   # Prepare evaluation data
   val_loader = get_validation_loader()
   
   def accuracy_eval_func(model):
       model.eval()
       correct = total = 0
       with torch.no_grad():
           for x, y in val_loader:
               x, y = x.cuda(), y.cuda()
               out = model(x)
               _, pred = out.max(-1)
               correct += (pred == y).sum().item()
               total += y.numel()
       return correct / total

   def speed_eval_func(model):
       model.eval()
       x = torch.randn(32, 128).long().cuda()  # batch_size=32, seq_len=128
       
       # Warmup
       for _ in range(20):
           with torch.no_grad():
               model(x)
       torch.cuda.synchronize()
       
       # Measure
       start = time.perf_counter()
       for _ in range(100):
           with torch.no_grad():
               model(x)
       torch.cuda.synchronize()
       elapsed = time.perf_counter() - start
       
       return (100 * 32) / elapsed  # samples/sec

   # Configure tuning for transformer layers
   configs = TuningConfigs([
       # Tune attention projections
       LayerConfig(
           layer_names={"contains": "attn", "type": "Linear"},
           params={
               "num_terms": Categorical([1, 2, 3]),
               "low_rank": Int(32, 256, step=32)
           },
           separate=True
       ),
       # Tune FFN layers
       LayerConfig(
           layer_names={"contains": "mlp", "type": "Linear"},
           params={
               "num_terms": Categorical([1, 2, 4]),
               "low_rank": Int(64, 512, step=64)
           },
           separate=True
       ),
   ])

   # Run constrained optimization
   tuner = SKAutoTuner(
       model=model,
       configs=configs,
       accuracy_eval_func=accuracy_eval_func,
       accuracy_threshold=0.98,  # Keep 98% of original accuracy
       optmization_eval_func=speed_eval_func,
       search_algorithm=OptunaSearch(
           n_trials=200,
           study_name="transformer_tuning",
           storage="sqlite:///transformer_tuning.db",
           seed=42
       ),
       verbose=True
   )

   tuner.tune()

   # Apply best configuration
   optimized_model = tuner.apply_best_params()

   # Final verification
   orig_acc = accuracy_eval_func(model)
   opt_acc = accuracy_eval_func(optimized_model)
   orig_speed = speed_eval_func(model)
   opt_speed = speed_eval_func(optimized_model)

   print(f"Original:  accuracy={orig_acc:.4f}, speed={orig_speed:.1f} samples/sec")
   print(f"Optimized: accuracy={opt_acc:.4f}, speed={opt_speed:.1f} samples/sec")
   print(f"Speedup: {opt_speed/orig_speed:.2f}x")

   # Save results
   tuner.save_tuning_results("best_config.pkl")
   tuner.visualize_tuning_results(save_path="tuning_analysis.png")