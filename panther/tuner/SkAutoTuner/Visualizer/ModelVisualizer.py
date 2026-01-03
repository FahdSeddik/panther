"""
ModelVisualizer - Utility for inspecting PyTorch model structures.

Provides metadata extraction, tree printing, JSON export, and summary
generation. For tuning result visualization, use Optuna's built-in
visualization functions on the study object.
"""

import json
import logging
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """
    A utility class for inspecting and summarizing PyTorch model structures.

    This class provides:
    - Tree-based module hierarchy printing
    - Metadata extraction (parameter counts, layer types, shapes)
    - Text-based model summaries
    - Model comparison utilities
    - JSON export of model structure
    """

    # -------------------------------------------------------------------------
    # Tree Building & Printing
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_module_tree(
        named_modules: Iterable,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Builds a nested dict representing the module hierarchy and collects module types.
        """
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
        """
        Recursively prints the nested dict as an ASCII tree with module types.
        """
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

        Args:
            model: The PyTorch model to visualize
            root_name: Name to display for the root module
        """
        tree, module_types = ModelVisualizer._build_module_tree(model.named_modules())
        module_types[""] = type(model).__name__
        print(f"{root_name} ({module_types.get('', 'UnknownType')})/")
        ModelVisualizer._print_tree(tree, module_types, full_path="")

    # -------------------------------------------------------------------------
    # Metadata Collection
    # -------------------------------------------------------------------------

    @staticmethod
    def _collect_module_info(model: nn.Module) -> Dict[str, Dict[str, Any]]:
        """
        Collects detailed information about each module in the model.

        Returns:
            Dictionary mapping module paths to their metadata
        """
        module_info: Dict[str, Dict[str, Any]] = {}

        # Root model info
        try:
            root_param_count = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            root_is_trainable = any(p.requires_grad for p in model.parameters())
        except AttributeError:
            root_param_count = 0
            root_is_trainable = False

        module_info["root"] = {
            "type": type(model).__name__,
            "parameters": root_param_count,
            "trainable": root_is_trainable,
            "class": str(type(model)),
            "docstring": (
                model.__doc__.strip().split("\n")[0] if model.__doc__ else "N/A"
            ),
        }

        for name, module in model.named_modules():
            if not name:
                continue

            try:
                param_count = sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )
                is_trainable = any(p.requires_grad for p in module.parameters())
            except (AttributeError, RuntimeError):
                param_count = 0
                is_trainable = False

            info: Dict[str, Any] = {
                "type": type(module).__name__,
                "parameters": param_count,
                "trainable": is_trainable,
                "class": str(type(module)),
            }

            # Add layer-specific details
            if isinstance(module, nn.Conv2d):
                info.update(
                    {
                        "in_channels": module.in_channels,
                        "out_channels": module.out_channels,
                        "kernel_size": module.kernel_size,
                        "stride": module.stride,
                        "padding": module.padding,
                        "groups": module.groups,
                    }
                )
            elif isinstance(module, nn.Linear):
                info.update(
                    {
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                        "bias": module.bias is not None,
                    }
                )
            elif isinstance(module, nn.BatchNorm2d):
                info.update(
                    {
                        "num_features": module.num_features,
                        "eps": module.eps,
                        "momentum": module.momentum,
                    }
                )
            elif isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
                info.update(
                    {
                        "input_size": module.input_size,
                        "hidden_size": module.hidden_size,
                        "num_layers": module.num_layers,
                        "bidirectional": module.bidirectional,
                    }
                )
            elif isinstance(module, nn.MultiheadAttention):
                info.update(
                    {
                        "embed_dim": module.embed_dim,
                        "num_heads": module.num_heads,
                        "dropout": module.dropout,
                    }
                )
            elif isinstance(module, nn.Embedding):
                info.update(
                    {
                        "num_embeddings": module.num_embeddings,
                        "embedding_dim": module.embedding_dim,
                    }
                )
            elif isinstance(module, nn.LayerNorm):
                info.update(
                    {
                        "normalized_shape": module.normalized_shape,
                        "eps": module.eps,
                    }
                )

            module_info[name] = info

        return module_info

    # -------------------------------------------------------------------------
    # JSON Export
    # -------------------------------------------------------------------------

    @staticmethod
    def export_model_structure(
        model: nn.Module, output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Export the model structure as a JSON-serializable dictionary.

        Args:
            model: The PyTorch model to export
            output_path: Optional path to save the JSON file

        Returns:
            Dictionary containing the model structure
        """
        tree, module_types = ModelVisualizer._build_module_tree(model.named_modules())
        module_info = ModelVisualizer._collect_module_info(model)

        export_data = {
            "model_type": type(model).__name__,
            "total_parameters": module_info["root"]["parameters"],
            "tree": tree,
            "module_types": module_types,
            "module_info": module_info,
        }

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)
            logger.info(f"Model structure exported to: {output_path}")

        return export_data

    # -------------------------------------------------------------------------
    # Text Summaries
    # -------------------------------------------------------------------------

    @staticmethod
    def get_model_summary_data(
        model: nn.Module, is_sketched_func: Optional[Callable[[nn.Module], bool]] = None
    ) -> Dict[str, Any]:
        """
        Get a summary of the model architecture, including sketched layers if a check function is provided.

        Args:
            model: The PyTorch model to summarize.
            is_sketched_func: Optional function to check if a module is sketched.

        Returns:
            Dictionary with model summary information
        """
        total_params = 0
        sketched_layers = 0
        layer_info = []

        for name, module in model.named_modules():
            if name == "":
                continue

            params = sum(p.numel() for p in module.parameters() if p.requires_grad)

            is_sketched = False
            if is_sketched_func and is_sketched_func(module):
                is_sketched = True
                sketched_layers += 1

            layer_info.append(
                {
                    "name": name,
                    "type": type(module).__name__,
                    "params": params,
                    "is_sketched": is_sketched,
                }
            )

            total_params += params

        return {
            "total_params": total_params,
            "sketched_layers": sketched_layers,
            "layers": layer_info,
        }

    @staticmethod
    def print_model_summary(
        model: nn.Module, is_sketched_func: Optional[Callable[[nn.Module], bool]] = None
    ) -> None:
        """
        Print a summary of the model architecture, highlighting sketched layers.

        Args:
            model: The PyTorch model to summarize.
            is_sketched_func: Optional function to check if a module is sketched.
        """
        summary = ModelVisualizer.get_model_summary_data(model, is_sketched_func)

        print("=" * 80)
        print(
            f"Model Summary (Total trainable parameters: {summary['total_params']:,})"
        )
        if is_sketched_func:
            print(f"Sketched layers: {summary['sketched_layers']}")
        print("-" * 80)
        header = f"{'Layer Name':<40} {'Layer Type':<25} {'Parameters':<15}"
        if is_sketched_func:
            header += " Sketched"
        print(header)
        print("-" * 80)

        for layer in summary["layers"]:
            row = f"{layer['name']:<40} {layer['type']:<25} {layer['params']:<15,}"
            if is_sketched_func:
                row += f" {'✓' if layer['is_sketched'] else ''}"
            print(row)
        print("=" * 80)

    @staticmethod
    def generate_model_summary(
        model: nn.Module,
        depth: int = -1,
        file_path: Optional[str] = None,
    ) -> str:
        """
        Generates a comprehensive text summary of a PyTorch model's architecture.

        Args:
            model: The PyTorch model to summarize
            depth: Maximum depth to display (-1 for unlimited)
            file_path: Optional file path to save the summary to

        Returns:
            A string containing the model summary
        """

        def count_parameters(m: nn.Module) -> Tuple[int, int]:
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            total = sum(p.numel() for p in m.parameters())
            return trainable, total

        summary_data = []
        total_trainable, total_params = count_parameters(model)

        for name, module in model.named_modules():
            if not name:
                continue

            if depth > 0 and name.count(".") >= depth:
                continue

            trainable, total = count_parameters(module)
            if total == 0:
                continue

            module_type = type(module).__name__
            param_str = f"{trainable:,} / {total:,}"

            shape_info = "N/A"
            if hasattr(module, "in_features") and hasattr(module, "out_features"):
                shape_info = f"{module.in_features} → {module.out_features}"
            elif hasattr(module, "in_channels") and hasattr(module, "out_channels"):
                kernel_str = (
                    f"{module.kernel_size}" if hasattr(module, "kernel_size") else ""
                )
                shape_info = (
                    f"{module.in_channels} → {module.out_channels} {kernel_str}"
                )

            indent = "  " * name.count(".")
            display_name = name.split(".")[-1]

            summary_data.append(
                [
                    indent + display_name,
                    module_type,
                    param_str,
                    shape_info,
                    f"{(trainable / total_trainable) * 100:.2f}%"
                    if total_trainable > 0
                    else "0.00%",
                ]
            )

        summary_lines = []
        summary_lines.append(f"Model Summary: {type(model).__name__}")
        summary_lines.append("=" * 80)
        summary_lines.append(
            f"Total Parameters: {total_params:,} ({total_trainable:,} trainable)"
        )
        summary_lines.append(
            f"Model Size: {total_params * 4 / (1024 * 1024):.2f} MB (assuming float32)"
        )
        summary_lines.append("=" * 80)

        headers = ["Layer", "Type", "Parameters", "Shape", "% of Params"]
        max_widths = [
            max(len(str(row[i])) for row in summary_data + [headers])
            for i in range(len(headers))
        ]

        header_row = " | ".join(h.ljust(max_widths[i]) for i, h in enumerate(headers))
        summary_lines.append(header_row)
        summary_lines.append("-" * len(header_row))

        for row in summary_data:
            summary_lines.append(
                " | ".join(str(col).ljust(max_widths[i]) for i, col in enumerate(row))
            )

        summary_text = "\n".join(summary_lines)

        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
            logger.info(f"Model summary saved to: {file_path}")

        return summary_text

    # -------------------------------------------------------------------------
    # Model Comparison
    # -------------------------------------------------------------------------

    @staticmethod
    def compare_models_data(
        model1: nn.Module,
        model2: nn.Module,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2",
        is_sketched_func_model1: Optional[Callable[[nn.Module], bool]] = None,
        is_sketched_func_model2: Optional[Callable[[nn.Module], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Compare two models and return detailed comparison data.

        Args:
            model1: The first PyTorch model.
            model2: The second PyTorch model.
            model1_name: Name for the first model.
            model2_name: Name for the second model.
            is_sketched_func_model1: Optional sketch check for model1.
            is_sketched_func_model2: Optional sketch check for model2.

        Returns:
            Dictionary with comparison metrics.
        """
        summary1 = ModelVisualizer.get_model_summary_data(
            model1, is_sketched_func_model1
        )
        summary2 = ModelVisualizer.get_model_summary_data(
            model2, is_sketched_func_model2
        )

        params1 = summary1["total_params"]
        params2 = summary2["total_params"]
        param_reduction = params1 - params2
        param_reduction_percent = (
            (param_reduction / params1) * 100 if params1 > 0 else 0
        )

        layer_comparisons = []
        layers1 = {layer["name"]: layer for layer in summary1["layers"]}
        layers2 = {layer["name"]: layer for layer in summary2["layers"]}
        all_layer_names = set(layers1.keys()) | set(layers2.keys())

        for name in sorted(list(all_layer_names)):
            info1 = layers1.get(name)
            info2 = layers2.get(name)
            comp_entry: Dict[str, Any] = {"name": name}
            changed = False

            if info1 and info2:
                comp_entry.update(
                    {
                        "model1_type": info1["type"],
                        "model2_type": info2["type"],
                        "model1_params": info1["params"],
                        "model2_params": info2["params"],
                        "param_diff": info1["params"] - info2["params"],
                    }
                )
                if info1["type"] != info2["type"] or info1["params"] != info2["params"]:
                    changed = True
                if is_sketched_func_model1 and is_sketched_func_model2:
                    comp_entry["model1_sketched"] = info1["is_sketched"]
                    comp_entry["model2_sketched"] = info2["is_sketched"]
                    if info1["is_sketched"] != info2["is_sketched"]:
                        changed = True
            elif info1:
                comp_entry.update(
                    {
                        "model1_type": info1["type"],
                        "model2_type": "N/A",
                        "model1_params": info1["params"],
                        "model2_params": "N/A",
                        "param_diff": info1["params"],
                    }
                )
                changed = True
            elif info2:
                comp_entry.update(
                    {
                        "model1_type": "N/A",
                        "model2_type": info2["type"],
                        "model1_params": "N/A",
                        "model2_params": info2["params"],
                        "param_diff": -info2["params"],
                    }
                )
                changed = True

            if changed:
                layer_comparisons.append(comp_entry)

        return {
            "model1_name": model1_name,
            "model2_name": model2_name,
            "model1_total_params": params1,
            "model2_total_params": params2,
            "param_reduction": param_reduction,
            "param_reduction_percent": param_reduction_percent,
            "layer_comparisons": layer_comparisons,
            "model1_sketched_layers": summary1.get("sketched_layers", 0),
            "model2_sketched_layers": summary2.get("sketched_layers", 0),
        }

    @staticmethod
    def print_comparison_summary(
        model1: nn.Module,
        model2: nn.Module,
        model1_name: str = "Original Model",
        model2_name: str = "Tuned Model",
        is_sketched_func_model1: Optional[Callable[[nn.Module], bool]] = None,
        is_sketched_func_model2: Optional[Callable[[nn.Module], bool]] = None,
    ) -> None:
        """
        Print a summary comparing two models.

        Args:
            model1: The first PyTorch model.
            model2: The second PyTorch model.
            model1_name: Name for the first model.
            model2_name: Name for the second model.
            is_sketched_func_model1: Optional sketch check for model1.
            is_sketched_func_model2: Optional sketch check for model2.
        """
        comp = ModelVisualizer.compare_models_data(
            model1,
            model2,
            model1_name,
            model2_name,
            is_sketched_func_model1,
            is_sketched_func_model2,
        )

        print("=" * 80)
        print(
            f"Model Comparison Summary: {comp['model1_name']} vs {comp['model2_name']}"
        )
        print("-" * 80)
        print(f"{comp['model1_name']} parameters: {comp['model1_total_params']:,}")
        if is_sketched_func_model1:
            print(
                f"{comp['model1_name']} sketched layers: {comp['model1_sketched_layers']}"
            )
        print(f"{comp['model2_name']} parameters: {comp['model2_total_params']:,}")
        if is_sketched_func_model2:
            print(
                f"{comp['model2_name']} sketched layers: {comp['model2_sketched_layers']}"
            )

        print(
            f"Parameter reduction: {comp['param_reduction']:,} ({comp['param_reduction_percent']:.2f}%)"
        )
        print("-" * 80)

        if not comp["layer_comparisons"]:
            print("No differences found in layer structure or parameters.")
        else:
            print("Layer Differences:")
            for layer in comp["layer_comparisons"]:
                m1_params = layer["model1_params"]
                m2_params = layer["model2_params"]
                m1_str = f"{m1_params:,}" if isinstance(m1_params, int) else m1_params
                m2_str = f"{m2_params:,}" if isinstance(m2_params, int) else m2_params
                diff = layer["param_diff"]
                diff_str = f"{diff:,}" if isinstance(diff, int) else diff
                print(
                    f"  {layer['name']}: {layer['model1_type']} → {layer['model2_type']} "
                    f"({m1_str} → {m2_str}, diff: {diff_str})"
                )
        print("=" * 80)

    # -------------------------------------------------------------------------
    # Matplotlib Visualization (optional dependency)
    # -------------------------------------------------------------------------

    @staticmethod
    def visualize_parameter_distribution(
        model: nn.Module,
        is_sketched_func: Optional[Callable[[nn.Module], bool]] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True,
    ) -> None:
        """
        Visualize the distribution of parameters across different layer types.

        Requires matplotlib to be installed.

        Args:
            model: The PyTorch model.
            is_sketched_func: Optional function to check if a module is sketched.
            save_path: Path to save the visualization.
            show_plot: Whether to display the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error(
                "Matplotlib is required for visualization. Install with: pip install matplotlib"
            )
            return

        summary = ModelVisualizer.get_model_summary_data(model, is_sketched_func)

        layer_types: Dict[str, int] = {}
        for layer in summary["layers"]:
            layer_type = layer["type"]
            layer_types[layer_type] = layer_types.get(layer_type, 0) + layer["params"]

        plt.figure(figsize=(12, 7) if is_sketched_func else (7, 7))

        ax1 = plt.subplot(1, 2, 1) if is_sketched_func else plt.subplot(1, 1, 1)
        labels = list(layer_types.keys())
        sizes = list(layer_types.values())

        # Filter small slices
        threshold = sum(sizes) * 0.01 if sum(sizes) > 0 else 0
        other_size = sum(s for s in sizes if 0 < s < threshold)
        filtered = [(label, s) for label, s in zip(labels, sizes) if s >= threshold]
        if other_size > 0:
            filtered.append(("Other (<1%)", other_size))

        if filtered:
            ax1.pie(
                [s for _, s in filtered],
                labels=[label for label, _ in filtered],
                autopct="%1.1f%%",
                startangle=90,
            )
        else:
            ax1.text(0.5, 0.5, "No parameters to display", ha="center", va="center")
        ax1.axis("equal")
        ax1.set_title("Parameter Distribution by Layer Type")

        if is_sketched_func:
            plt.subplot(1, 2, 2)
            sketched_params = sum(
                layer["params"] for layer in summary["layers"] if layer["is_sketched"]
            )
            non_sketched_params = sum(
                layer["params"]
                for layer in summary["layers"]
                if not layer["is_sketched"]
            )

            if sketched_params == 0 and non_sketched_params == 0:
                plt.text(0.5, 0.5, "No parameters to display", ha="center", va="center")
            else:
                plt.bar(
                    ["Non-Sketched", "Sketched"],
                    [non_sketched_params, sketched_params],
                    color=["skyblue", "lightcoral"],
                )
            plt.ylabel("Number of Parameters")
            plt.title("Parameters in Sketched vs Non-Sketched Layers")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Parameter distribution visualization saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()
