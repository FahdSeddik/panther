# Example usage
import torch.nn as nn
from panther.utils.SKAutoTuner import SKAutoTuner, LayerConfig, TuningConfigs, GridSearch

# Define a model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Define configurations for tuning
layer_config = LayerConfig(
    layer_names=["0", "2"],  # Tune the first and third layers (Linear layers)
    params={
        "num_terms": [1, 2, 4],
        "low_rank": [16, 32, 64]
    },
    separate=True  # Tune each layer separately
)

configs = TuningConfigs([layer_config])

# Define an evaluation function
def eval_func(model):
    # for now let it be the time of inference
    import time
    import torch
    dummy_input = torch.randn(1, 784)  # Example input size for the model
    start_time = time.time()
    with torch.no_grad():
        model(dummy_input)  # Forward pass
    end_time = time.time()
    return end_time - start_time

# Create the autotuner with grid search
tuner = SKAutoTuner(
    model=model,
    configs=configs,
    eval_func=eval_func,
    search_algorithm=GridSearch(),
    verbose=True
)

# Run the tuning process
best_params = tuner.tune()

# Apply the best parameters to the model
optimized_model = tuner.apply_best_params()

# print the optimized model structure
print(optimized_model)

# Save the tuned model
tuner.save("tuned_model.pkl")