"""
Search algorithm implementations for the SkAutoTuner.

This module provides search algorithms for hyperparameter tuning:
- SearchAlgorithm: Abstract base class that defines the interface
- OptunaSearch: Industry-standard HPO using Optuna (RECOMMENDED)

For grid or random search, use OptunaSearch with the appropriate sampler:
- optuna.samplers.GridSampler for grid search
- optuna.samplers.RandomSampler for random search
"""

from .OptunaSearch import OptunaSearch
from .SearchAlgorithm import SearchAlgorithm

__all__ = [
    "SearchAlgorithm",
    "OptunaSearch",
]
