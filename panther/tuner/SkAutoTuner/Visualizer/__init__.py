"""
Visualizer - Layer name discovery for SkAutoTuner.

Use ModelVisualizer.print_module_tree() to discover layer names for your
LayerConfig selectors.

For model summaries with accurate parameter counts, use torchinfo:
    pip install torchinfo
    from torchinfo import summary
    summary(model, input_size=(1, 3, 224, 224))

For tuning result visualization, use Optuna's built-in visualization:
    import optuna
    study = tuner.search_algorithm.study
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()
"""

from .ModelVisualizer import ModelVisualizer

__all__ = [
    "ModelVisualizer",
]
