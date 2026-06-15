"""
Search algorithm implementations for the SkAutoTuner.

This module provides search algorithms for hyperparameter tuning:
- SearchAlgorithm: Abstract base class that defines the interface
- OptunaSearch: Industry-standard HPO using Optuna (RECOMMENDED)
- GridSearch: Simple exhaustive search reference implementation

For random search, use OptunaSearch with optuna.samplers.RandomSampler.
For exhaustive grid search, use GridSearch or implement SearchAlgorithm directly.
"""

from .GridSearch import GridSearch
from .OptunaSearch import OptunaSearch
from .SearchAlgorithm import SearchAlgorithm

__all__ = [
    "SearchAlgorithm",
    "OptunaSearch",
    "GridSearch",
]
