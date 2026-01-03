"""
ModelVisualizer - Utility for discovering layer names in PyTorch models.

Prints a tree of module names and types to help users craft correct layer
selectors for SkAutoTuner. For model summaries with accurate parameter counts,
use torchinfo. For tuning result visualization, use Optuna's built-in
visualization functions on the study object.
"""

from typing import Any, Dict, Iterable, Tuple

import torch.nn as nn


class ModelVisualizer:
    """
    A utility class for discovering layer names in PyTorch models.

    This class helps users inspect model structure to craft correct layer
    selectors for SkAutoTuner. For accurate parameter counts and model
    summaries, use torchinfo instead.
    """

    @staticmethod
    def _build_module_tree(
        named_modules: Iterable,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        tree: Dict[str, Any] = {}
        module_types: Dict[str, str] = {}

        for full_name, module in named_modules:
            module_types[full_name] = type(module).__name__
            parts = full_name.split(".") if full_name else []
            current_level = tree
            for part in parts:
                current_level = current_level.setdefault(part, {})

        return tree, module_types

    @staticmethod
    def _print_tree(
        subtree: Dict[str, Any],
        module_types: Dict[str, str],
        full_path: str = "",
        prefix: str = "",
        is_last: bool = True,
    ) -> None:
        branch = "└─ " if is_last else "├─ "
        for idx, (name, child) in enumerate(sorted(subtree.items())):
            is_child_last = idx == len(subtree) - 1
            current_path = f"{full_path}.{name}" if full_path else name
            module_type = (
                f" ({module_types.get(current_path, 'UnknownType')})"
                if current_path in module_types
                else ""
            )
            print(prefix + branch + name + module_type + ("/" if child else ""))
            if child:
                extension = "    " if is_child_last else "│   "
                ModelVisualizer._print_tree(
                    child, module_types, current_path, prefix + extension, is_child_last
                )

    @staticmethod
    def print_module_tree(model: nn.Module, root_name: str = "model") -> None:
        """
        Prints the modules of a PyTorch model in a tree structure with their types.

        Use this to discover layer names for SkAutoTuner layer selectors.

        Args:
            model: The PyTorch model to visualize.
            root_name: Name to display for the root module.

        Example:
            >>> ModelVisualizer.print_module_tree(my_model)
            model (MyModel)/
            └─ encoder (Encoder)/
                └─ layers (ModuleList)/
                    └─ 0 (Linear)
                    └─ 1 (Linear)
        """
        tree, module_types = ModelVisualizer._build_module_tree(model.named_modules())
        module_types[""] = type(model).__name__
        print(f"{root_name} ({module_types.get('', 'UnknownType')})/")
        ModelVisualizer._print_tree(tree, module_types, full_path="")
