import copy
import os
import random

import torch
import torch.nn as nn

from panther.utils import *

# For reproducibility
torch.manual_seed(0)
random.seed(0)


# 1. Define a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=1024, kernel_size=3, padding=1
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        # Calculate flattened size: 1024 channels, image H/2, W/2. Assume 32x32 input -> 16x16 after pool.
        # (1024 * 16 * 16)
        self.fc1 = nn.Linear(1024 * 16 * 16, 2048)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 1024)  # Output layer

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x


# 2. Define evaluation functions
def dummy_accuracy_eval_func(model: nn.Module) -> float:
    """
    A dummy accuracy evaluation function.
    In a real scenario, this would evaluate the model on a validation dataset.
    This function gives slightly higher accuracy if layers are sketched.
    """
    base_accuracy = 0.6
    sketched_bonus = 0.0
    num_sketched = 0
    for module in model.modules():
        if "SK" in type(module).__name__:  # Check if it's a sketched layer
            sketched_bonus += 0.05
            num_sketched += 1
            # Example: Favor specific sketch parameters for variety in results
            if hasattr(module, "num_terms") and hasattr(module, "low_rank"):
                if module.num_terms > 15:  # Arbitrary condition for demo
                    sketched_bonus += 0.02
                if module.low_rank < 10:  # Arbitrary condition for demo
                    sketched_bonus += 0.01

    # Simulate some noise or dependency on parameters
    if num_sketched > 0:
        # Small random factor to make tuning non-deterministic if not for seed
        return min(1.0, base_accuracy + sketched_bonus + random.uniform(-0.01, 0.01))
    return base_accuracy + random.uniform(-0.01, 0.01)


def dummy_optimization_eval_func(model: nn.Module) -> float:
    """
    A dummy optimization evaluation function (e.g., inference speed).
    Higher is better. This function simulates that sketched layers are faster.
    """
    simulated_latency = 0.05  # Base latency
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            params = sum(p.numel() for p in module.parameters())
            if "SK" in type(module).__name__:  # Sketched layer
                simulated_latency += 0.0000005 * params  # Sketched layers are faster
            else:  # Original layer
                simulated_latency += 0.0000025 * params

    # Score is inverse of latency (higher score = faster)
    return 1.0 / simulated_latency if simulated_latency > 0 else 0.0


if __name__ == "__main__":
    print("SKAutoTuner Example Script")
    print("==========================")

    # Create dummy input for model (batch_size=1, 3 channels, 32x32 image)
    dummy_input = torch.randn(1, 3, 32, 32)

    # --- Initial Model ---
    original_model = SimpleModel()
    print("\n--- Original Model Summary ---")
    # To use print_model_summary, we need a tuner instance with the model
    temp_tuner_orig = SKAutoTuner(original_model, TuningConfigs([]), lambda m: 0.0)
    temp_tuner_orig.print_model_summary()

    # --- Configuration for Tuning ---
    # Define which layers to tune and with what parameters
    # Note: Keep parameter ranges small for quick example execution.
    config1 = LayerConfig(
        layer_names=["conv1"],
        params={"num_terms": [10, 20], "low_rank": [5, 8]},
        separate=True,  # Tune this layer group separately
        copy_weights=True,
    )
    config2 = LayerConfig(
        layer_names=["fc1"],
        params={"num_terms": [15, 25], "low_rank": [6, 10]},
        separate=True,
        copy_weights=True,
    )
    tuning_configs = TuningConfigs(configs=[config1, config2])

    # Create a copy of the model for the main tuning process
    model_for_tuning = copy.deepcopy(original_model)

    # --- Instantiate SKAutoTuner ---
    print("\n--- Initializing SKAutoTuner for Tuning ---")
    tuner = SKAutoTuner(
        model=model_for_tuning,
        configs=tuning_configs,
        accuracy_eval_func=dummy_accuracy_eval_func,
        optmization_eval_func=dummy_optimization_eval_func,
        accuracy_threshold=0.65,  # Aim for at least this accuracy
        verbose=True,
        num_runs_per_param=1,  # For faster example execution
    )

    # --- 1. Tune the model ---
    print("\n--- Starting Tuning Process (tune) ---")
    tuner.tune()
    print("Tuning finished.")

    # --- 2. Get Best Parameters ---
    print("\n--- Best Parameters Found (get_best_params) ---")
    best_params = tuner.get_best_params()
    for layer_name, params_info in best_params.items():
        print(f"Layer: {layer_name}, Best Params: {params_info['params']}")

    # --- 3. Get Results DataFrame ---
    print("\n--- Tuning Results DataFrame (get_results_dataframe) ---")
    # This requires pandas to be installed.
    try:
        results_df = tuner.get_results_dataframe()
        print(results_df.to_string())
    except ImportError:
        print("Pandas not installed. Skipping get_results_dataframe().")
    except Exception as e:
        print(f"Could not generate DataFrame: {e}")

    # --- 4. Apply Best Parameters ---
    print("\n--- Applying Best Parameters to the Model (apply_best_params) ---")
    # tuner.model is modified in-place by apply_best_params()
    tuned_model_explicit_return = tuner.apply_best_params()
    print("Best parameters applied. Model summary after tuning:")
    tuner.print_model_summary()  # Shows the state of tuner.model

    # --- 5. Visualize Tuning Results ---
    print("\n--- Visualizing Tuning Results (visualize_tuning_results) ---")
    # This requires matplotlib and pandas.
    # Create a directory for plots if it doesn't exist.
    viz_dir = "tuning_visualizations"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    viz_path = os.path.join(viz_dir, "tuning_visualization.png")
    try:
        tuner.visualize_tuning_results(save_path=viz_path, show_plot=False)
        print(f"Tuning visualization saved to {viz_path}")
        print("To view the plot, open the saved image file.")
    except ImportError:
        print(
            "Matplotlib or Pandas not installed. Skipping visualize_tuning_results()."
        )
    except Exception as e:
        print(f"Could not visualize results: {e}")
        if "No variable parameters found to visualize" in str(e):
            print(
                "This can happen if all parameter combinations resulted in the same score or only one combination was tried."
            )

    # --- 7. Get Model Summary (Explicitly from Tuned Model) ---
    print("\n--- Explicit call to get_model_summary (on tuned model) ---")
    # tuner.print_model_summary() was already called after apply_best_params,
    # this shows how to get the raw dictionary.
    model_summary_dict = tuner.get_model_summary()
    print(f"Total parameters in tuned model: {model_summary_dict['total_params']}")
    print(f"Number of sketched layers: {model_summary_dict['sketched_layers']}")
    # print(f"Full summary dict: {model_summary_dict}") # Uncomment to see full structure

    # --- 8. Save Tuning Results ---
    print("\n--- Saving Tuning Results (save_tuning_results) ---")
    results_dir = "tuning_results_data"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file_path = os.path.join(results_dir, "tuning_session.pkl")
    tuner.save_tuning_results(results_file_path)
    print(f"Tuning results (including best_params) saved to {results_file_path}")

    # --- 9. Load Tuning Results (Example on a new Tuner instance) ---
    print("\n--- Loading Tuning Results (load_tuning_results) ---")
    # Create a new model and tuner instance to load into
    model_for_loading = copy.deepcopy(original_model)
    tuner_for_loading = SKAutoTuner(
        model=model_for_loading,
        configs=tuning_configs,  # Important: Configs should match the saved session for meaningful application
        accuracy_eval_func=dummy_accuracy_eval_func,  # Required by constructor
        verbose=False,  # Keep output clean for this demo part
    )
    try:
        tuner_for_loading.load_tuning_results(results_file_path)
        print(
            f"Successfully loaded tuning results from {results_file_path} into a new tuner instance."
        )
        loaded_best_params = tuner_for_loading.get_best_params()
        print("Best parameters from loaded results:")
        if loaded_best_params:
            for layer_name, params_info in loaded_best_params.items():
                # Check if params_info is not None and 'params' key exists
                if params_info and "params" in params_info:
                    print(
                        f"  Layer: {layer_name}, Best Params: {params_info['params']}"
                    )
                else:
                    print(
                        f"  Layer: {layer_name}, No best params data found in loaded results for this layer."
                    )
        else:
            print(
                "  No best parameters were loaded (or best_params was empty in the file)."
            )

        # Optionally, apply these loaded parameters
        # tuner_for_loading.apply_best_params()
        # print("Model summary after loading results and applying them:")
        # tuner_for_loading.print_model_summary()

    except FileNotFoundError:
        print(f"ERROR: Could not load results - File not found: {results_file_path}")
    except (
        Exception
    ) as e:  # Catch other potential errors like pickle issues or invalid format
        print(f"ERROR: Could not load results: {e}")

    # --- 10. Export Tuned Model State ---
    # The 'tuner' instance still holds the model that was tuned and had best params applied
    print("\n--- Exporting Tuned Model State (export_model) ---")
    model_export_dir = "exported_models"
    if not os.path.exists(model_export_dir):
        os.makedirs(model_export_dir)
    tuned_model_path = os.path.join(model_export_dir, "tuned_simple_model.pth")
    tuner.export_model(tuned_model_path)  # Exports tuner.model.state_dict()
    print(f"Tuned model state_dict exported to {tuned_model_path}")

    # --- 11. Visualize Parameter Distribution of Tuned Model ---
    print(
        "\n--- Visualizing Parameter Distribution of Tuned Model (visualize_parameter_distribution) ---"
    )
    # This uses the model currently in the 'tuner' instance (which is the tuned one)
    param_dist_path = os.path.join(viz_dir, "tuned_model_parameter_distribution.png")
    try:
        tuner.visualize_parameter_distribution(
            save_path=param_dist_path, show_plot=False
        )
        print(f"Parameter distribution visualization saved to {param_dist_path}")
    except ImportError:
        print(
            "Matplotlib or Pandas not installed. Skipping visualize_parameter_distribution()."
        )
    except Exception as e:
        print(f"Could not visualize parameter distribution: {e}")

    # --- 12. Get Inference Benchmark ---
    print("\n--- Getting Inference Benchmark (get_inference_benchmark) ---")
    # Prepare a dummy input tensor for benchmarking
    # Ensure it's on the same device as the model if using GPU later
    benchmark_input = torch.randn(1, 3, 32, 32)  # Same as dummy_input earlier

    print("\nBenchmarking original model:")
    original_model_for_bench = copy.deepcopy(original_model)  # Use a fresh copy
    # Need a tuner instance to call get_inference_benchmark
    tuner_for_original_bench = SKAutoTuner(
        original_model_for_bench, TuningConfigs([]), lambda m: 0.0
    )
    original_benchmark_results = None  # Initialize for robust access later
    try:
        original_benchmark_results = tuner_for_original_bench.get_inference_benchmark(
            benchmark_input, num_runs=50, warm_up=5
        )
        print(
            f"  Original Model Benchmark (avg time): {original_benchmark_results.get('mean_time', 'N/A'):.6f} s, FPS: {original_benchmark_results.get('fps', 'N/A'):.2f}"
        )
    except Exception as e:
        print(f"  Could not benchmark original model: {e}")

    print("\nBenchmarking tuned model (from main 'tuner' instance):")
    tuned_benchmark_results = None  # Initialize for robust access later
    try:
        # tuner.model is already the tuned one
        tuned_benchmark_results = tuner.get_inference_benchmark(
            benchmark_input, num_runs=50, warm_up=5
        )
        print(
            f"  Tuned Model Benchmark (avg time): {tuned_benchmark_results.get('mean_time', 'N/A'):.6f} s, FPS: {tuned_benchmark_results.get('fps', 'N/A'):.2f}"
        )

        # Compare benchmark results
        if (
            original_benchmark_results
            and tuned_benchmark_results
            and original_benchmark_results.get("mean_time") is not None
            and tuned_benchmark_results.get("mean_time") is not None
        ):
            original_time = original_benchmark_results["mean_time"]
            tuned_time = tuned_benchmark_results["mean_time"]
            if original_time > 0:  # Avoid division by zero
                speedup = (original_time - tuned_time) / original_time * 100
                print(
                    f"  Potential speedup: {speedup:.2f}% (based on mean inference time)"
                )
    except Exception as e:
        print(f"  Could not benchmark tuned model: {e}")

    # --- 6. Demonstrate replace_without_tuning ---
    print("\n--- Demonstrating Replace Without Tuning (replace_without_tuning) ---")
    # Use a fresh copy of the original model for this demonstration
    model_for_replace = copy.deepcopy(original_model)

    print("Model summary before replace_without_tuning:")
    temp_tuner_replace_before = SKAutoTuner(
        model_for_replace,
        TuningConfigs([]),
        lambda m: 0.0,
        # ^ This tuner is just for print_model_summary, uses empty configs
    )
    temp_tuner_replace_before.print_model_summary()

    # Re-initialize tuner with the fresh model and original configs
    # No need for eval funcs or accuracy_threshold for replace_without_tuning
    tuner_for_replace = SKAutoTuner(
        model=model_for_replace,
        configs=tuning_configs,  # <--- Key point: The full 'tuning_configs' IS PROVIDED HERE
        accuracy_eval_func=dummy_accuracy_eval_func,  # Required by __init__, though not used by this specific method
        verbose=True,
    )

    # replace_without_tuning uses the *first* parameter from the lists in LayerConfig
    replaced_model_explicit_return = (
        tuner_for_replace.replace_without_tuning()
    )  # This calls the method on tuner_for_replace
    print("\nModel summary after replace_without_tuning:")
    tuner_for_replace.print_model_summary()  # Shows the state of tuner_for_replace.model

    print("\n--- Additional SKAutoTuner features demonstrated ---")
    # print_comparison_summary was already called, it uses compare_models internally.
    # This shows how to get the raw dictionary from compare_models.
    print(
        "\n- Model Comparison (print_comparison_summary - uses compare_models internally):"
    )
    # Ensure original_model is pristine.
    tuner.print_comparison_summary(original_model=original_model)

    print("\n- Explicit call to compare_models to get the raw dictionary:")
    try:
        # The 'tuner' instance holds the tuned model.
        # 'original_model' is the one from the beginning of the script.
        comparison_dict = tuner.compare_models(original_model=original_model)
        print(
            f"  Parameter reduction from compare_models dict: {comparison_dict.get('param_reduction_percent', 0):.2f}%"
        )
        # print(f"  Full comparison_dict: {comparison_dict}") # Uncomment to see all details
    except Exception as e:
        print(f"  Could not get comparison dictionary: {e}")

    print("\n--- SKAutoTuner Example Script Finished ---")
    print("Note: This script uses dummy evaluation functions.")
    print("In a real application, provide actual model evaluation logic.")
    print(
        "Ensure 'matplotlib' and 'pandas' are installed to see visualizations and dataframes."
    )

    # note that in this example the search_algorithm is not used which will be gridSearch by default.
    # you can change it to randomSearch or bayesianSearch by passing the search_algorithm parameter to SKAutoTuner.
    # and note that some searching algorithms like hyperband will require the eval and optmization function
    # to take 'resource' parameter which can be used as number of epochs for example.

    # note that when using POS the params first 2 values would be used
    # as the min and max values for the search space for this param.
    # and it will search across all the range with step of 1.
