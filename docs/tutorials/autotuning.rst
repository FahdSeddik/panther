Autotuning for Maximum Performance
===================================

This tutorial shows you how to use Panther's AutoTuner to automatically find the best sketching parameters for your neural network, maximizing throughput while maintaining accuracy.

.. contents:: Table of Contents
   :local:
   :depth: 2

Why the Auto-Tuner? The Bridge to Production
---------------------------------------------

The **SKAutoTuner** is the critical bridge that transforms standard PyTorch models into production-optimized Panther models. Without it, deploying sketched networks would be manual, error-prone, and effectively impossible at scale.

**The Problem It Solves:**

Deep neural networks have complex nested hierarchies. To deploy a sketched model, you must:

1. Navigate deep module hierarchies to locate target layers
2. Extract and preserve original weights
3. Replace layers with sketched versions (e.g., ``SKLinear``)
4. Discover optimal sketching parameters for each layer
5. Search multidimensional parameter spaces efficiently
6. Maintain accuracy while maximizing speed

**Manual Workflow (Without the Tuner):**

Manually replacing even a single BERT layer requires:

- Understanding model architecture and naming conventions
- Writing custom code to traverse ``BertForMaskedLM`` → ``BertOnlyMLMHead`` → ``BertLMPredictionHead`` → target ``Linear``
- Carefully managing weight copying to preserve training
- Guessing or grid-searching through parameter combinations
- Re-evaluating accuracy and speed for each attempt
- Finding it was slow, inaccurate, and unreproducible

This quickly becomes unmaintainable for large models with hundreds of layers across multiple architectures.

**What the Auto-Tuner Does:**

- **Automates hierarchy navigation**: Finds layers using intuitive pattern matching
- **Manages complexity behind the scenes**: Weight management, layer replacement, configuration tracking
- **Discovers optimal parameters systematically**: Uses industry-standard Optuna with state-of-the-art TPE sampler
- **Guarantees accuracy thresholds**: Searches only parameter combinations that maintain your target accuracy
- **Maximizes speed** within constraints: Optimizes throughput once accuracy is satisfied
- **Handles noisy evaluations**: Re-evaluates uncertain configurations to ensure robustness
- **Provides full traceability**: Complete metrics on every trial for analysis and visualization

**Real Impact:**

- Using a BERT based model , using the tuner we were able to reduce parameters from 109.51M to 30.38M maintaining the same accuracy and achieving a 1.04x speedup using wikitext-2-raw-v1 dataset.
- Achieved 30% parameter reduction on Resnet50 with 3.5% accuracy loss on Cifar-10
- Deploy across multiple architectures (BERT,ResNet, etc.) with the same workflow

**One-line difference:**

Without tuner: Hours of debugging, manual layer selection, parameter guessing
With tuner: ``tuner.tune()`` → Apply best params → Deploy

Overview
--------

When sketching neural network layers, choosing the right parameters e.g. (``num_terms``, ``low_rank``) is crucial:

- **Too aggressive**: Fast but inaccurate
- **Too conservative**: Accurate but slow
- **Just right**: Maximum speedup with acceptable accuracy loss

The ``SKAutoTuner`` automates this entire process using **industry-standard hyperparameter optimization** (Optuna with TPE sampler), relieving you from manual parameter tuning, layer discovery, and accuracy management.

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

Behind the scenes, the tuner automatically:

1. **Discovers layers**: Finds all Linear layers in your model using flexible pattern matching
2. **Generates parameter space**: Creates intelligent search ranges based on layer dimensions
3. **Searches efficiently**: Uses Optuna's TPE sampler to intelligently explore the parameter space (not random or grid)
4. **Maintains accuracy**: Evaluates your function on each trial
5. **Applies winners**: Replaces layers with their sketched versions using optimal parameters

Industrial-Grade Search Capabilities
------------------------------------

Unlike basic grid search or random sampling, Panther's tuner uses **Optuna** with the **Tree-structured Parzen Estimator (TPE)** sampler—the same technology used in production at top tech companies:

**Why TPE (Tree-Structured Parzen Estimator)?**

- **Intelligent exploration**: Builds probabilistic models of the parameter space
- **Adaptive sampling**: Focuses trials in promising regions while still exploring
- **Sample efficient**: Finds good parameters in 50-200 trials instead of thousands
- **Mixed parameter types**: Handles categorical, integer, and continuous parameters simultaneously
- **No assumptions**: Works well regardless of parameter distributions

**Without Optuna/TPE** (manual or random search):

- Grid search: 3 × 3 × 3 = 27 trials for 3 parameters with 3 values each
- Random: Inefficient, many wasted trials
- Manual guessing: Error-prone and non-reproducible

**With Optuna/TPE** (tuner default):

- Intelligently narrows search space
- Often converges in 50-100 trials
- Reproducible results with seed control
- Full trial history for analysis

Constraint-Based Optimization: The Tuner's Superpower
-----------------------------------------------------

The most powerful feature is **constraint-based optimization**: **maximize speed while maintaining a minimum accuracy threshold**. This ensures you never sacrifice model quality for compression.

This is what transforms the tuner from a hyperparameter tool into a production-grade compression framework:

- **Problem**: Sketching trades accuracy for speed. How much should we compress?
- **Solution**: Define your acceptable accuracy loss (e.g., 99% of original), and the tuner finds the fastest configuration that meets that constraint
- **Result**: No guessing, no manual validation, provably maintains quality while maximizing deployment benefit

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

Layer Discovery and Pattern Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tuner can find layers in multiple ways—you don't need to know the exact layer names:

**By type** (find all layers of a kind):

.. code-block:: python

   # All Linear layers in the model
   layer_names={"type": "Linear"}

**By name pattern** (regex matching):

.. code-block:: python

   # Linear layers in the encoder (BERT transformer blocks)
   layer_names={"pattern": "encoder\\.layers\\.[0-5]\\..*", "type": "Linear"}

**By text matching** (simple substring):

.. code-block:: python

   # Attention layers
   layer_names={"contains": "attn"}

**By exact name** (if you know it):

.. code-block:: python

   # Specific layers
   layer_names=["layer1.0.linear", "layer2.1.linear"]

Automatic Parameter Space Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tuner can automatically determine good search ranges based on layer dimensions:

.. code-block:: python

   config = LayerConfig(
       layer_names={"type": "Linear"},
       params="auto",  # Tuner decides optimal ranges
       separate=True
   )

For a Linear layer with 768 input and 3072 output dimensions, the tuner automatically generates ranges like:

- ``num_terms``: [1, 2, 3, 4]
- ``low_rank``: [64, 128, 256, 512] (based on min dimension)

Or explicitly specify your own ranges:

.. code-block:: python

   from panther.tuner import Categorical, Int

   config = LayerConfig(
       layer_names={"type": "Linear"},
       params={
           "num_terms": Categorical([1, 2, 3, 4]),
           "low_rank": Int(8, 128, step=8)
       },
       separate=True
   )

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

1. **Each trial evaluates accuracy first** (fast failure)
2. **If accuracy ≥ threshold**: The configuration passes; its speed score becomes the objective
3. **If accuracy < threshold**: The configuration is rejected with score ``-inf``
4. **The tuner maximizes speed** among all configurations meeting the accuracy bar
5. **Best configuration is selected** based on the fastest valid configuration found

This is a **constrained optimization problem** and is exactly how production ML systems work—you cannot sacrifice quality, but you can optimize deployment speed.

Real-World Impact: What the Tuner Achieves
-------------------------------------------

The SKAutoTuner delivers concrete, measurable results:

**BERT Model Compression (110M parameters)**

- Original: 110M parameters, 66ms per sample, 1.0x baseline
- Tuned with sketching: 30.38M parameters (72.26% reduction), 63.5ms per sample, **1.04x speedup**
- Accuracy maintained: 99.8% of original performance


**ResNet-50 on Cifar-10**

- Tuned: 30% parameter reduction, 52ms per image, **1.05x speedup**
- Top-1 accuracy: 89% → 85.5% (only 3.5% drop for 30% parameter reduction)

Advanced Configuration
----------------------

Using Different Samplers
~~~~~~~~~~~~~~~~~~~~~~~~

The tuner defaults to Optuna's TPE (Tree-structured Parzen Estimator) sampler, which works well for most problems. You can customize it:

.. code-block:: python

   from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler

   # TPE (default) - good general-purpose sampler, best for mixed parameter spaces
   search = OptunaSearch(n_trials=100, sampler=TPESampler(seed=42))

   # CMA-ES - excellent for continuous parameters, requires many trials
   search = OptunaSearch(n_trials=200, sampler=CmaEsSampler(seed=42))

   # Random - useful as a simple baseline or sanity check
   search = OptunaSearch(n_trials=50, sampler=RandomSampler(seed=42))

**Recommendation:** Start with TPE (default). Use CMA-ES only if you have pure continuous parameters and many trials to spend.

Analyzing Results
-----------------

DataFrame Analysis
~~~~~~~~~~~~~~~~~~

The tuner provides comprehensive trial data for deep analysis:

.. code-block:: python

   import pandas as pd

   # Get all trial results
   df = tuner.get_results_dataframe()

   # Columns: layer_name, num_terms, low_rank, accuracy, speed, score, trial_number

   # Best configurations per layer
   best_per_layer = df.loc[df.groupby("layer_name")["score"].idxmax()]
   print(best_per_layer)

   # Configurations meeting accuracy threshold and their speeds
   valid = df[df["accuracy"] >= 0.95]
   print(valid.sort_values("speed", ascending=False).head(10))

   # How does each parameter affect performance?
   for param in ["num_terms", "low_rank"]:
       if param in df.columns:
           print(f"\n{param} vs Score:")
           print(df.groupby(param)["score"].agg(["mean", "std", "count"]))

   # Correlation between parameters and accuracy
   print(df[["num_terms", "low_rank", "accuracy"]].corr())

**What to look for:**

- Do valid configurations exist? (Any row where accuracy ≥ threshold)
- What's the speedup range? (min/max speed of valid configs)
- Which parameters matter? (High std in score across parameter values)
- Are there clusters? (Do certain parameter values consistently beat others?)

Visualization
~~~~~~~~~~~~~

Use Optuna's built-in visualization for rich analysis:

.. code-block:: python

   import optuna.visualization as vis

   # Access the underlying Optuna study
   study = tuner.search_algorithm.study
   
   # Parameter importances - which params affect score most?
   fig = vis.plot_param_importances(study)
   fig.show()
   
   # Optimization history - did the search make progress?
   fig = vis.plot_optimization_history(study)
   fig.show()
   
   # Parameter relationships
   fig = vis.plot_parallel_coordinate(study, params=["num_terms", "low_rank"])
   fig.show()
   
   # 2D contour of parameter interactions
   fig = vis.plot_contour(study, params=["num_terms", "low_rank"])
   fig.show()

Best Practices
--------------

**Golden Rules:**

1. **Start with "auto"**: Let the tuner determine reasonable ranges first. Saves manual parameter exploration.

2. **Use a representative validation set**: Your accuracy function must reflect real-world performance. A small, biased validation set leads to bad parameter choices.

3. **Warm up GPU**: Include warmup iterations in speed measurements to stabilize GPU clocks. Skip first few iterations in timing:

   .. code-block:: python

      # Bad: includes GPU warmup time
      start = time.perf_counter()
      for i in range(100):
          model(x)
      
      # Good: GPU is warmed up
      for _ in range(20):  # Warmup
          model(x)
      torch.cuda.synchronize()
      start = time.perf_counter()
      for i in range(100):
          model(x)
      torch.cuda.synchronize()
      elapsed = time.perf_counter() - start

4. **Set realistic accuracy thresholds**: A 0.99 threshold when original accuracy is 0.92 is impossible. Start with 0.98 or 0.95 (1-2% loss acceptable).

5. **Save results**: Use database storage for long tuning runs (100+ trials). Enables resumption if interrupted.

6. **Visualize early**: After ~20 trials, check the optimization history. If no progress, increase n_trials or adjust parameter ranges.

7. **Consider batch size**: Measure speed at your **deployment batch size**, not just any batch size. Smaller batches → less throughput advantage, larger batches → memory bottleneck.

8. **Check layer coverage**: Use ``ModelVisualizer.print_module_tree()`` to confirm your patterns match intended layers.

9. **Start conservative**: Begin with tight accuracy thresholds. You can loosen them later if needed. Better to have a super-stable model first.

10. **Enable verbose mode**: ``verbose=True`` during development to track progress and identify stuck searches.

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
   
   # Visualize with Optuna (optional)
   # import optuna.visualization as vis
   # vis.plot_optimization_history(tuner.search_algorithm.study).show()