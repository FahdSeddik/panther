import json
import logging
import os
import re
import tempfile
import webbrowser
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

try:
    from graphviz import Digraph, ExecutableNotFound
except ImportError:
    Digraph = None
    ExecutableNotFound = Exception

# Setup a logger for this module
logger = logging.getLogger(__name__)


class ModelVisualizer:
    """
    A utility class for visualizing PyTorch model structures interactively.
    It generates an HTML file with an SVG representation of the model,
    allowing users to explore module hierarchy, view details, and search.
    """

    # Get the path to the visualization assets
    # Use a simpler approach that doesn't rely on importlib.resources
    _ASSETS_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "visualization_assets"
    )

    @staticmethod
    def _build_module_tree(
        named_modules: Iterable,
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Builds a nested dict representing the module hierarchy and collects module types.
        """
        tree: Dict[str, Any] = {}
        module_types = {}

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
    ):
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
    def print_module_tree(model: nn.Module, root_name: str = "model"):
        """
        Prints the modules of a PyTorch model in a tree structure with their types.
        """
        tree, module_types = ModelVisualizer._build_module_tree(model.named_modules())
        module_types[""] = type(model).__name__
        print(f"{root_name} ({module_types.get('', 'UnknownType')})/")
        ModelVisualizer._print_tree(tree, module_types, full_path="")

    @staticmethod
    def _collect_module_info(model: nn.Module) -> Dict[str, Dict[str, Any]]:
        """
        Collects detailed information about each module in the model.
        """
        module_info = {}

        try:
            root_param_count = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            root_is_trainable = any(p.requires_grad for p in model.parameters())
        except AttributeError:
            root_param_count = 0
            root_is_trainable = False
            logger.warning("Could not retrieve parameter info for the root model.")

        module_info["root"] = {
            "type": type(model).__name__,
            "parameters": root_param_count,
            "trainable": root_is_trainable,
            "class": str(type(model)),
            "docstring": model.__doc__.strip().split("\n")[0]
            if model.__doc__
            else "N/A",
        }

        for name, module in model.named_modules():
            if not name:
                continue

            try:
                param_count = sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )
                is_trainable = any(p.requires_grad for p in module.parameters())
            except AttributeError:
                param_count = 0
                is_trainable = False
                logger.warning(
                    f"Module {name} ({type(module).__name__}) does not have 'parameters' attribute or it's not iterable."
                )
            except RuntimeError:
                param_count = 0
                is_trainable = False
                logger.warning(
                    f"Could not count parameters for module {name} ({type(module).__name__})."
                )

            info = {
                "type": type(module).__name__,
                "parameters": param_count,
                "trainable": is_trainable,
                "class": str(type(module)),
                "docstring": module.__doc__.strip().split("\n")[0]
                if module.__doc__
                else "N/A",
            }

            if isinstance(module, nn.Conv2d):
                info.update(
                    {
                        "in_channels": module.in_channels,
                        "out_channels": module.out_channels,
                        "kernel_size": module.kernel_size,
                        "stride": module.stride,
                        "padding": module.padding,
                        "groups": module.groups,
                        "dilation": module.dilation,
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
                        "affine": module.affine,
                    }
                )
            elif isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
                info.update(
                    {
                        "input_size": module.input_size,
                        "hidden_size": module.hidden_size,
                        "num_layers": module.num_layers,
                        "bidirectional": module.bidirectional,
                        "dropout": module.dropout if hasattr(module, "dropout") else 0,
                        "bias": module.bias,
                    }
                )
            elif isinstance(module, nn.Dropout):
                info.update(
                    {
                        "p": module.p,
                        "inplace": module.inplace,
                    }
                )
            elif isinstance(
                module,
                (
                    nn.MaxPool2d,
                    nn.AvgPool2d,
                    nn.AdaptiveAvgPool2d,
                    nn.AdaptiveMaxPool2d,
                ),
            ):
                info_pool = {
                    "kernel_size": getattr(module, "kernel_size", "N/A"),
                    "stride": getattr(module, "stride", "N/A"),
                    "padding": getattr(module, "padding", "N/A"),
                }
                if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
                    info_pool["output_size"] = module.output_size
                info.update(info_pool)
            elif isinstance(module, nn.Embedding):
                info.update(
                    {
                        "num_embeddings": module.num_embeddings,
                        "embedding_dim": module.embedding_dim,
                        "padding_idx": module.padding_idx,
                    }
                )
            elif isinstance(module, nn.LayerNorm):
                info.update(
                    {
                        "normalized_shape": module.normalized_shape,
                        "eps": module.eps,
                        "elementwise_affine": module.elementwise_affine,
                    }
                )
            elif isinstance(module, nn.MultiheadAttention):
                info.update(
                    {
                        "embed_dim": module.embed_dim,
                        "num_heads": module.num_heads,
                        "dropout": module.dropout,
                        "bias": hasattr(module, "bias_k") and module.bias_k is not None,
                        "add_bias_kv": hasattr(module, "add_bias_kv")
                        and module.add_bias_kv,
                        "add_zero_attn": hasattr(module, "add_zero_attn")
                        and module.add_zero_attn,
                        "kdim": getattr(module, "kdim", None),
                        "vdim": getattr(module, "vdim", None),
                    }
                )
            elif isinstance(
                module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)
            ):
                info.update(
                    {
                        "d_model": module.self_attn.embed_dim
                        if hasattr(module, "self_attn")
                        else getattr(module, "d_model", "N/A"),
                        "nhead": module.self_attn.num_heads
                        if hasattr(module, "self_attn")
                        else getattr(module, "nhead", "N/A"),
                        "dim_feedforward": module.linear1.out_features
                        if hasattr(module, "linear1")
                        else getattr(module, "dim_feedforward", "N/A"),
                        "dropout": module.dropout.p
                        if hasattr(module, "dropout")
                        else getattr(module, "dropout", "N/A"),
                        "activation": type(module.activation).__name__
                        if hasattr(module, "activation")
                        else getattr(module, "activation", "N/A"),
                    }
                )
                if isinstance(module, nn.TransformerDecoderLayer):
                    info["cross_attention"] = True

            module_info[name] = info

        return module_info

    @staticmethod
    def create_interactive_visualization(
        model: nn.Module,
        output_path: Optional[str] = None,
        graph_attrs: Optional[Dict[str, str]] = None,
        node_attrs: Optional[Dict[str, str]] = None,
        edge_attrs: Optional[Dict[str, str]] = None,
        open_browser: bool = True,
        max_label_length: int = 30,
    ) -> str:
        """
        Creates an interactive visualization of the model structure.
        """
        if Digraph is None:
            raise ImportError(
                "The graphviz Python package is required. Install with 'pip install graphviz'."
            )

        default_graph_attrs = {
            "rankdir": "TB",
            "bgcolor": "transparent",
            "splines": "ortho",
            "fontname": "Arial, Helvetica, sans-serif",
            "fontsize": "14",
            "nodesep": "0.6",
            "ranksep": "0.8",
            "concentrate": "true",
            "overlap": "false",
        }
        if graph_attrs:
            default_graph_attrs.update(graph_attrs)

        default_node_attrs = {
            "style": "filled,rounded",
            "shape": "box",
            "fillcolor": "#E5F5FD",
            "color": "#4285F4",
            "fontname": "Arial, Helvetica, sans-serif",
            "fontsize": "11",
            "height": "0.4",
            "margin": "0.1,0.05",
        }
        if node_attrs:
            default_node_attrs.update(node_attrs)

        default_edge_attrs = {"color": "#757575", "arrowsize": "0.7"}
        if edge_attrs:
            default_edge_attrs.update(edge_attrs)

        dot = Digraph(
            "model_visualization", format="svg", graph_attr=default_graph_attrs
        )
        dot.attr("node", **default_node_attrs)
        dot.attr("edge", **default_edge_attrs)

        raw_named_modules = list(model.named_modules())
        tree, module_types = ModelVisualizer._build_module_tree(raw_named_modules)
        module_info = ModelVisualizer._collect_module_info(model)

        root_name_for_graph = "model"
        root_display_type = module_info["root"]["type"]
        root_label = f"{root_name_for_graph} ({root_display_type})"
        if len(root_label) > max_label_length:
            root_label = root_label[: max_label_length - 3] + "..."

        root_id = "node_root_model"
        dot.node(
            root_id,
            root_label,
            tooltip=f"Root: {root_display_type}\nParameters: {module_info['root']['parameters']:,}",
            id=root_id,
            data_name="root",
            fillcolor="#D1E7F7",
            shape="Mrecord",
        )

        node_ids = {"root": root_id}

        def add_nodes_and_edges(current_subtree, parent_full_path, parent_node_id):
            for name_part, children_subtree in sorted(current_subtree.items()):
                current_full_path = (
                    f"{parent_full_path}.{name_part}"
                    if parent_full_path != "root"
                    else name_part
                )
                node_id = (
                    f"node_{current_full_path.replace('.', '_').replace('-', '_')}"
                )
                node_ids[current_full_path] = node_id
                module_type_name = module_types.get(current_full_path, "Unknown")

                label = f"{name_part} ({module_type_name})"
                if len(label) > max_label_length:
                    label = label[: max_label_length - 3] + "..."
                tooltip_parts = [
                    f"Name: {current_full_path}",
                    f"Type: {module_type_name}",
                ]
                current_module_details = module_info.get(current_full_path)
                if current_module_details:
                    tooltip_parts.append(
                        f"Parameters: {current_module_details['parameters']:,}"
                    )
                    tooltip_parts.append(
                        f"Trainable: {'Yes' if current_module_details['trainable'] else 'No'}"
                    )
                node_fillcolor = default_node_attrs.get("fillcolor", "#E5F5FD")
                if not children_subtree:
                    node_fillcolor = "#C2E0F4"

                dot.node(
                    node_id,
                    label,
                    tooltip="\n".join(tooltip_parts),
                    fillcolor=node_fillcolor,
                    id=node_id,
                    data_name=current_full_path,
                )
                edge_id = f"edge_{parent_node_id}_{node_id}"
                dot.edge(
                    parent_node_id,
                    node_id,
                    id=edge_id,
                    data_source=parent_node_id,
                    data_target=node_id,
                )
                if children_subtree:
                    add_nodes_and_edges(children_subtree, current_full_path, node_id)

        add_nodes_and_edges(tree, "root", root_id)

        svg_content_bytes = dot.pipe(format="svg")
        svg_content = svg_content_bytes.decode("utf-8")

        for node_path_key, node_html_id in node_ids.items():
            data_name_attr_str = f'data-name="{node_path_key}"'
            g_block_pattern = (
                rf'(<g[^>]*id="{re.escape(node_html_id)}"[^>]*>)([\s\S]*?)(</g>)'
            )

            def process_g_block(match_obj):
                g_open_tag, g_content, g_close_tag = match_obj.groups()
                if data_name_attr_str not in g_open_tag:
                    g_open_tag = g_open_tag.rstrip(">") + f" {data_name_attr_str}>"

                def add_data_to_visual_child(child_match):
                    child_tag_open, child_tag_rest = child_match.groups()
                    if data_name_attr_str not in child_tag_open:
                        return (
                            child_tag_open + f" {data_name_attr_str}" + child_tag_rest
                        )
                    return child_match.group(0)

                g_content = re.sub(
                    r"(<(?:rect|polygon|ellipse|text|path|circle)\b[^>]*?)(/?>)",
                    add_data_to_visual_child,
                    g_content,
                )
                return g_open_tag + g_content + g_close_tag

            svg_content = re.sub(g_block_pattern, process_g_block, svg_content)

        js_module_info = json.dumps(module_info)

        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".html")
            os.close(fd)

        template_path = os.path.join(ModelVisualizer._ASSETS_PATH, "template.html")
        if not os.path.exists(template_path):
            logger.error(f"HTML template not found at {template_path}")
            raise FileNotFoundError(f"HTML template not found at {template_path}")
        with open(template_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        css_path = os.path.join(ModelVisualizer._ASSETS_PATH, "css", "styles.css")
        if not os.path.exists(css_path):
            logger.error(f"CSS file not found at {css_path}")
            raise FileNotFoundError(f"CSS file not found at {css_path}")
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()

        js_path = os.path.join(ModelVisualizer._ASSETS_PATH, "js", "script.js")
        if not os.path.exists(js_path):
            logger.error(f"JavaScript file not found at {js_path}")
            raise FileNotFoundError(f"JavaScript file not found at {js_path}")
        with open(js_path, "r", encoding="utf-8") as f:
            js_content = f.read()

        html_content = template_content.replace("{{SVG_CONTENT}}", svg_content)
        html_content = html_content.replace("{{MODULE_INFO}}", js_module_info)
        html_content = html_content.replace("{{CSS_CONTENT}}", css_content)
        html_content = html_content.replace("{{JS_CONTENT}}", js_content)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        if open_browser:
            try:
                url = "file://" + os.path.abspath(output_path)
                webbrowser.open(url)
                logger.info(f"Visualization opened in browser: {url}")
            except Exception as e:
                logger.warning(f"Could not automatically open browser: {e}")

        logger.info(f"Interactive model visualization saved to: {output_path}")
        return output_path

    @staticmethod
    def compare_models(
        model1: nn.Module,
        model2: nn.Module,
        output_path: Optional[str] = None,
        open_browser: bool = True,
    ) -> str:
        """
        Creates a comparison visualization of two PyTorch models, highlighting the differences in structure.

        Args:
            model1: First PyTorch model to compare
            model2: Second PyTorch model to compare
            output_path: Optional path to save the HTML output file. If None, a temporary file will be created.
            open_browser: Whether to open the visualization in a browser automatically.

        Returns:
            Path to the saved HTML file
        """
        if Digraph is None:
            raise ImportError(
                "The graphviz Python package is required. Install with 'pip install graphviz'."
            )

        # Collect info for both models
        info1 = ModelVisualizer._collect_module_info(model1)
        info2 = ModelVisualizer._collect_module_info(model2)

        # Build trees for both models
        tree1, types1 = ModelVisualizer._build_module_tree(model1.named_modules())
        tree2, types2 = ModelVisualizer._build_module_tree(model2.named_modules())

        # Create visualization
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".html")
            os.close(fd)

        # Create comparison data
        comparison_data = {
            "model1": {
                "name": "Model 1",
                "type": type(model1).__name__,
                "params": info1["root"]["parameters"],
                "info": info1,
                "tree": tree1,
                "types": types1,
            },
            "model2": {
                "name": "Model 2",
                "type": type(model2).__name__,
                "params": info2["root"]["parameters"],
                "info": info2,
                "tree": tree2,
                "types": types2,
            },
            "diff": {
                "only_in_model1": [],
                "only_in_model2": [],
                "param_differences": {},
                "type_differences": {},
            },
        }

        # Find unique modules in each model
        model1_paths = set(info1.keys())
        model2_paths = set(info2.keys())

        comparison_data["diff"]["only_in_model1"] = list(
            model1_paths - model2_paths - {"root"}
        )
        comparison_data["diff"]["only_in_model2"] = list(
            model2_paths - model1_paths - {"root"}
        )

        # Find type differences and parameter differences in common modules
        common_paths = model1_paths.intersection(model2_paths)
        for path in common_paths:
            if path == "root":
                continue

            # Check type differences
            if info1[path]["type"] != info2[path]["type"]:
                comparison_data["diff"]["type_differences"][path] = {
                    "model1_type": info1[path]["type"],
                    "model2_type": info2[path]["type"],
                }

            # Check parameter differences
            if info1[path]["parameters"] != info2[path]["parameters"]:
                comparison_data["diff"]["param_differences"][path] = {
                    "model1_params": info1[path]["parameters"],
                    "model2_params": info2[path]["parameters"],
                    "diff": info1[path]["parameters"] - info2[path]["parameters"],
                }

        # Generate comparison HTML
        template_path = os.path.join(ModelVisualizer._ASSETS_PATH, "template.html")
        if not os.path.exists(template_path):
            logger.error(f"HTML template not found at {template_path}")
            raise FileNotFoundError(f"HTML template not found at {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        # Customize the template for comparison view
        html_content = template_content.replace(
            "<title>Interactive Model Visualization</title>",
            "<title>Model Comparison Visualization</title>",
        )
        html_content = html_content.replace(
            "<h1>Interactive Model Visualization</h1>",
            "<h1>Model Comparison Visualization</h1>",
        )

        # Add comparison data
        comparison_json = json.dumps(comparison_data)
        html_content = html_content.replace(
            "const moduleInfo = {{MODULE_INFO}};",
            f"const comparisonData = {comparison_json};\nconst moduleInfo = null;",
        )

        # Add comparison-specific CSS and JS
        css_path = os.path.join(ModelVisualizer._ASSETS_PATH, "css", "styles.css")
        if not os.path.exists(css_path):
            logger.error(f"CSS file not found at {css_path}")
            raise FileNotFoundError(f"CSS file not found at {css_path}")

        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()

        # Inject custom comparison styling
        comparison_css = """
        .comparison-container {
            display: flex;
            flex-direction: row;
            gap: 20px;
        }
        .model-col {
            flex: 1;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 10px;
        }
        .diff-highlight {
            background-color: #ffe0e0;
        }
        .model1-only {
            background-color: #e0ffe0;
        }
        .model2-only {
            background-color: #e0e0ff;
        }
        .comparison-header {
            font-weight: bold;
            background-color: #f0f0f0;
            padding: 5px;
            margin-bottom: 10px;
            border-radius: 4px;
        }
        .diff-summary {
            background-color: #f9f9f9;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 3px solid #f44336;
        }
        body.dark-theme .diff-highlight {
            background-color: #5a3030;
        }
        body.dark-theme .model1-only {
            background-color: #304a30;
        }
        body.dark-theme .model2-only {
            background-color: #30305a;
        }
        """
        html_content = html_content.replace(
            "{{CSS_CONTENT}}", css_content + comparison_css
        )

        # Modify the visualization container for comparison view
        comparison_container = """
        <div class="comparison-container">
            <div class="model-col">
                <div class="comparison-header">Model 1: <span id="model1-name"></span></div>
                <div id="model1-structure"></div>
            </div>
            <div class="model-col">
                <div class="comparison-header">Model 2: <span id="model2-name"></span></div>
                <div id="model2-structure"></div>
            </div>
        </div>
        <div class="diff-summary" id="diff-summary"></div>
        """
        html_content = html_content.replace(
            '<div class="svg-pan-zoom-container" id="svgPanZoomContainer">\n             {{SVG_CONTENT}}\n        </div>',
            comparison_container,
        )

        # Add comparison JS (simplified for brevity)
        comparison_js = """
        // Simple comparison rendering logic
        document.addEventListener('DOMContentLoaded', function() {
            if (!comparisonData) {
                document.getElementById('infoPanel').innerHTML = '<p>Error: No comparison data available</p>';
                return;
            }
            
            // Set model names
            document.getElementById('model1-name').textContent = comparisonData.model1.type + ' (' + comparisonData.model1.params.toLocaleString() + ' params)';
            document.getElementById('model2-name').textContent = comparisonData.model2.type + ' (' + comparisonData.model2.params.toLocaleString() + ' params)';
            
            // Display summary
            const diffSummary = document.getElementById('diff-summary');
            diffSummary.innerHTML = `
                <h3>Differences Summary</h3>
                <p>Modules only in Model 1: ${comparisonData.diff.only_in_model1.length}</p>
                <p>Modules only in Model 2: ${comparisonData.diff.only_in_model2.length}</p>
                <p>Modules with different types: ${Object.keys(comparisonData.diff.type_differences).length}</p>
                <p>Modules with different parameters: ${Object.keys(comparisonData.diff.param_differences).length}</p>
            `;
            
            // Render simple tree views
            function renderTree(container, tree, prefix = '') {
                const ul = document.createElement('ul');
                
                for (const [name, subtree] of Object.entries(tree)) {
                    const li = document.createElement('li');
                    const fullPath = prefix ? prefix + '.' + name : name;
                    
                    // Apply highlighting
                    if (comparisonData.diff.only_in_model1.includes(fullPath)) {
                        li.classList.add('model1-only');
                    } else if (comparisonData.diff.only_in_model2.includes(fullPath)) {
                        li.classList.add('model2-only');
                    } else if (comparisonData.diff.type_differences[fullPath] || comparisonData.diff.param_differences[fullPath]) {
                        li.classList.add('diff-highlight');
                    }
                    
                    li.textContent = name;
                    if (Object.keys(subtree).length > 0) {
                        li.style.fontWeight = 'bold';
                        const childUl = renderTree(container, subtree, fullPath);
                        li.appendChild(childUl);
                    }
                    
                    ul.appendChild(li);
                }
                
                return ul;
            }
            
            document.getElementById('model1-structure').appendChild(
                renderTree(document.getElementById('model1-structure'), comparisonData.model1.tree)
            );
            
            document.getElementById('model2-structure').appendChild(
                renderTree(document.getElementById('model2-structure'), comparisonData.model2.tree)
            );
            
            // When clicking on model elements, show details in the info panel
            document.querySelectorAll('.model-col li').forEach(li => {
                li.addEventListener('click', function(e) {
                    e.stopPropagation();
                    // Logic to display module details would go here
                });
            });
        });
        """

        html_content = html_content.replace("{{JS_CONTENT}}", comparison_js)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        if open_browser:
            try:
                url = "file://" + os.path.abspath(output_path)
                webbrowser.open(url)
                logger.info(f"Model comparison visualization opened in browser: {url}")
            except Exception as e:
                logger.warning(f"Could not automatically open browser: {e}")

        logger.info(f"Model comparison visualization saved to: {output_path}")
        return output_path

    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        input_shape: tuple,
        output_path: Optional[str] = None,
        dynamic_axes: Optional[Dict] = None,
        opset_version: int = 11,
        visualize: bool = True,
    ) -> str:
        """
        Exports a PyTorch model to ONNX format and optionally visualizes it.

        Args:
            model: PyTorch model to export
            input_shape: Input shape for the model (excluding batch size)
            output_path: Path to save the ONNX file. If None, a temporary file will be created.
            dynamic_axes: Dynamic axes specification for ONNX export
            opset_version: ONNX opset version to use
            visualize: Whether to generate a visualization of the ONNX model

        Returns:
            Path to the saved ONNX file
        """
        try:
            import numpy as np
            import torch.onnx
        except ImportError:
            raise ImportError(
                "PyTorch with ONNX support is required for this function."
            )

        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".onnx")
            os.close(fd)

        # Create a dummy input tensor
        batch_size = 1
        full_shape = (batch_size,) + input_shape
        dummy_input = torch.randn(full_shape, requires_grad=False)

        # Set model to evaluation mode
        model.eval()

        # Default dynamic axes if not provided
        if dynamic_axes is None and visualize:
            # Assume batch dimension (0) is dynamic by default
            dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

        # Export to ONNX
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )
            logger.info(f"Model exported to ONNX format: {output_path}")
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {str(e)}")
            raise

        # Visualize the ONNX model if requested
        if visualize:
            try:
                visualization_path = ModelVisualizer._visualize_onnx(output_path)
                logger.info(
                    f"ONNX model visualization created at: {visualization_path}"
                )
            except ImportError as ie:
                logger.warning(f"Could not visualize ONNX model: {str(ie)}")
            except Exception as e:
                logger.warning(f"Error visualizing ONNX model: {str(e)}")

        return output_path

    @staticmethod
    def _visualize_onnx(onnx_path: str) -> str:
        """
        Creates a visualization of an ONNX model.

        Args:
            onnx_path: Path to the ONNX model file

        Returns:
            Path to the visualization file
        """
        try:
            import onnx
            from onnx.tools.net_drawer import GetOpNodeProducer, GetPydotGraph
        except ImportError:
            raise ImportError(
                "ONNX and pydot packages are required. Install with 'pip install onnx pydot'"
            )

        # Load the ONNX model
        model = onnx.load(onnx_path)

        # Create a graph visualization
        pydot_graph = GetPydotGraph(
            model.graph,
            name=model.graph.name,
            rankdir="TB",
            node_producer=GetOpNodeProducer(
                "docstring",
                color="yellow",
                fillcolor="#FFEEEE",
                style="filled",
                shape="box",
            ),
        )

        # Save the visualization
        base_path = os.path.splitext(onnx_path)[0]
        output_path = f"{base_path}_visualization.svg"
        pydot_graph.write_svg(output_path)

        return output_path

    @staticmethod
    def generate_model_summary(
        model: nn.Module,
        input_shape: Optional[tuple] = None,
        depth: int = -1,
        file_path: Optional[str] = None,
    ) -> str:
        """
        Generates a comprehensive text summary of a PyTorch model's architecture.

        Args:
            model: The PyTorch model to summarize
            input_shape: Optional input shape (excluding batch dimension) to calculate output shapes
            depth: Maximum depth to display (-1 for unlimited)
            file_path: Optional file path to save the summary to

        Returns:
            A string containing the model summary
        """
        try:
            from tabulate import tabulate
        except ImportError:
            logger.warning(
                "Tabulate package not found. Install with 'pip install tabulate' for better formatting."
            )
            tabulate = None

        # Function to count parameters
        def count_parameters(m: nn.Module) -> Tuple[int, int]:
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            total = sum(p.numel() for p in m.parameters())
            return trainable, total

        # Collect module info
        summary_data = []
        total_trainable, total_params = count_parameters(model)

        for name, module in model.named_modules():
            if not name:  # Skip the model itself
                continue

            # Skip if exceeding depth
            if depth > 0 and name.count(".") >= depth:
                continue

            trainable, total = count_parameters(module)
            if total == 0:
                continue  # Skip modules without parameters

            # Get module type
            module_type = type(module).__name__

            # Format parameter count
            param_str = f"{trainable:,} / {total:,}"

            # Get module shape info if available
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

            # Indentation for hierarchy display
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

        # Build the summary string
        summary_lines = []
        summary_lines.append(f"Model Summary: {type(model).__name__}")
        summary_lines.append(f"{'=' * 80}")
        summary_lines.append(
            f"Total Parameters: {total_params:,} ({total_trainable:,} trainable)"
        )
        summary_lines.append(
            f"Model Size: {total_params * 4 / (1024 * 1024):.2f} MB (assuming float32)"
        )
        summary_lines.append(f"{'=' * 80}")

        # Create the table
        if tabulate:
            headers = [
                "Layer",
                "Type",
                "Parameters (Trainable/Total)",
                "Shape",
                "% of Params",
            ]
            summary_lines.append(
                tabulate(summary_data, headers=headers, tablefmt="grid")
            )
        else:
            # Simple formatting if tabulate is not available
            headers = ["Layer", "Type", "Parameters", "Shape", "% of Params"]
            max_widths = [
                max(len(str(row[i])) for row in summary_data + [headers])
                for i in range(len(headers))
            ]

            # Add header
            header_row = " | ".join(
                h.ljust(max_widths[i]) for i, h in enumerate(headers)
            )
            summary_lines.append(header_row)
            summary_lines.append("-" * len(header_row))

            # Add data rows
            for row in summary_data:
                summary_lines.append(
                    " | ".join(
                        str(col).ljust(max_widths[i]) for i, col in enumerate(row)
                    )
                )

        # Combine into a single string
        summary_text = "\n".join(summary_lines)

        # Save to file if requested
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
            logger.info(f"Model summary saved to: {file_path}")

        return summary_text

    @staticmethod
    def count_ops(model: nn.Module, input_shape: tuple) -> Dict[str, Any]:
        """
        Estimates the number of operations (FLOPs) for a model.

        Args:
            model: The PyTorch model to analyze
            input_shape: Input shape (excluding batch dimension)

        Returns:
            A dictionary containing operation counts by layer
        """
        try:
            import torch.autograd.profiler as profiler
        except ImportError:
            logger.error("PyTorch profiler not available")
            raise ImportError("PyTorch profiler is required for this function")

        # Create a dummy input tensor
        batch_size = 1
        full_shape = (batch_size,) + input_shape
        dummy_input = torch.randn(full_shape, requires_grad=False)

        # Set model to evaluation mode
        model.eval()

        # Dictionary to store results
        op_counts = {
            "total_flops": 0,
            "total_params": sum(p.numel() for p in model.parameters()),
            "layers": {},
        }

        # Register hooks to count operations
        handles = []

        def count_conv2d(module, input_tensor, output_tensor):
            # Get input dimensions
            batch_size, in_channels, in_h, in_w = input_tensor[0].shape
            # Get output dimensions
            batch_size, out_channels, out_h, out_w = output_tensor.shape
            # Kernel dimensions
            kernel_h, kernel_w = module.kernel_size
            # Groups
            groups = module.groups

            # Calculate FLOPs per output element
            flops_per_element = (
                kernel_h * kernel_w * in_channels * out_channels // groups
            )

            # Total FLOPs
            total_flops = flops_per_element * out_h * out_w * batch_size

            # Store in our dictionary
            module_name = [
                name for name, mod in model.named_modules() if mod is module
            ][0]
            op_counts["layers"][module_name] = {
                "type": "Conv2d",
                "flops": total_flops,
                "params": sum(p.numel() for p in module.parameters()),
                "shape": f"{in_channels}x{in_h}x{in_w} → {out_channels}x{out_h}x{out_w}",
                "kernel": f"{kernel_h}x{kernel_w}",
            }
            op_counts["total_flops"] += total_flops

        def count_linear(module, input_tensor, output_tensor):
            # Get input dimensions
            if isinstance(input_tensor[0], torch.Tensor):
                batch_size, in_features = input_tensor[0].shape
            else:
                batch_size, in_features = 0, 0

            # Calculate FLOPs (multiply-adds)
            flops = batch_size * in_features * module.out_features

            # Store in our dictionary
            module_name = [
                name for name, mod in model.named_modules() if mod is module
            ][0]
            op_counts["layers"][module_name] = {
                "type": "Linear",
                "flops": flops,
                "params": sum(p.numel() for p in module.parameters()),
                "shape": f"{in_features} → {module.out_features}",
            }
            op_counts["total_flops"] += flops

        # Register hooks for different module types
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(count_conv2d))
            elif isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(count_linear))

        # Run the model to trigger the hooks
        with torch.no_grad():
            model(dummy_input)

        # Remove the hooks
        for handle in handles:
            handle.remove()

        return op_counts

    @staticmethod
    def export_to_torchscript(
        model: nn.Module,
        input_shape: tuple,
        output_path: Optional[str] = None,
        method: str = "trace",
    ) -> str:
        """
        Exports a PyTorch model to TorchScript format.

        Args:
            model: The PyTorch model to export
            input_shape: Input shape (excluding batch dimension)
            output_path: Path to save the TorchScript model
            method: TorchScript conversion method ('trace' or 'script')

        Returns:
            Path to the saved TorchScript model
        """
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".pt")
            os.close(fd)

        # Create a dummy input tensor
        batch_size = 1
        full_shape = (batch_size,) + input_shape
        dummy_input = torch.randn(full_shape, requires_grad=False)

        # Set model to evaluation mode
        model.eval()

        # Export using the specified method
        try:
            if method == "trace":
                logger.info("Tracing model to TorchScript...")
                traced_model = torch.jit.trace(model, dummy_input)
                traced_model.save(output_path)
            elif method == "script":
                logger.info("Scripting model to TorchScript...")
                scripted_model = torch.jit.script(model)
                scripted_model.save(output_path)
            else:
                raise ValueError(
                    f"Invalid method: {method}. Must be 'trace' or 'script'."
                )

            logger.info(f"Model exported to TorchScript: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to export model to TorchScript: {str(e)}")
            raise

    @staticmethod
    def check_hardware_compatibility(
        model: nn.Module, input_shape: tuple, target_devices: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Checks model compatibility with different hardware targets and evaluates efficiency.

        Args:
            model: The PyTorch model to analyze
            input_shape: Input shape (excluding batch dimension)
            target_devices: List of target devices to check (default: ['cpu', 'cuda'])

        Returns:
            Dictionary with compatibility information
        """
        if target_devices is None:
            target_devices = ["cpu"]
            if torch.cuda.is_available():
                target_devices.append("cuda")

        # Create a dummy input tensor
        batch_size = 1
        full_shape = (batch_size,) + input_shape
        dummy_input = torch.randn(full_shape, requires_grad=False)

        # Set model to evaluation mode
        model.eval()

        # Result dictionary
        compatibility = {
            "model_info": {
                "type": type(model).__name__,
                "parameters": sum(p.numel() for p in model.parameters()),
                "size_mb": sum(p.numel() for p in model.parameters())
                * 4
                / (1024 * 1024),
            },
            "devices": {},
        }

        # Check op compatibility
        def check_ops_compatibility(model_ops, target_device):
            # This is a simplified check - in a real implementation, you would
            # have a database of ops supported by each device/backend
            all_ops = set()
            unsupported_ops = set()

            # Collect all unique operations in the model
            for name, module in model.named_modules():
                op_type = type(module).__name__
                all_ops.add(op_type)

                # Example: Check for ops that might not be well-supported on certain devices
                if target_device == "cpu" and op_type in [
                    "ConvTranspose3d",
                    "PixelShuffle",
                ]:
                    unsupported_ops.add(op_type)
                elif (
                    target_device == "cuda"
                    and op_type in ["RNN", "LSTM"]
                    and not torch.cuda.is_available()
                ):
                    unsupported_ops.add(op_type)

            return {
                "total_op_types": len(all_ops),
                "all_op_types": sorted(list(all_ops)),
                "unsupported_op_types": sorted(list(unsupported_ops)),
                "compatibility_score": 1.0
                - (len(unsupported_ops) / max(1, len(all_ops))),
            }

        # Perform inference benchmark on each device
        for device_name in target_devices:
            try:
                # Check if device is available
                device_available = (device_name == "cpu") or (
                    device_name == "cuda" and torch.cuda.is_available()
                )

                if not device_available:
                    compatibility["devices"][device_name] = {
                        "available": False,
                        "error": f"{device_name} is not available in this environment",
                    }
                    continue

                # Move model and input to device
                device = torch.device(device_name)
                model_on_device = model.to(device)
                input_on_device = dummy_input.to(device)

                # Warm-up run
                with torch.no_grad():
                    model_on_device(input_on_device)

                # Benchmark run
                if device_name == "cuda":
                    torch.cuda.synchronize()

                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                # Multiple runs for more accurate timing
                num_runs = 10

                if device_name == "cuda":
                    start_time.record()
                    for _ in range(num_runs):
                        with torch.no_grad():
                            model_on_device(input_on_device)
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time) / num_runs  # ms
                else:
                    # CPU timing
                    import time

                    start = time.time()
                    for _ in range(num_runs):
                        with torch.no_grad():
                            model_on_device(input_on_device)
                    elapsed_time = (time.time() - start) * 1000 / num_runs  # ms

                # Check ops compatibility
                ops_compatibility = check_ops_compatibility(model, device_name)

                # Memory usage (only available for CUDA)
                memory_usage = None
                if device_name == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    with torch.no_grad():
                        model_on_device(input_on_device)
                    memory_usage = torch.cuda.max_memory_allocated() / (
                        1024 * 1024
                    )  # MB

                compatibility["devices"][device_name] = {
                    "available": True,
                    "inference_time_ms": elapsed_time,
                    "memory_usage_mb": memory_usage,
                    "ops_compatibility": ops_compatibility,
                }

            except Exception as e:
                compatibility["devices"][device_name] = {
                    "available": False,
                    "error": str(e),
                }
                logger.warning(
                    f"Error checking compatibility with {device_name}: {str(e)}"
                )

        return compatibility

    @staticmethod
    def generate_complexity_report(
        model: nn.Module, input_shapes: Dict[str, tuple]
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive complexity report for a model with different input sizes.

        Args:
            model: The PyTorch model to analyze
            input_shapes: Dictionary mapping input size names to input shapes (excluding batch dimension)

        Returns:
            Dictionary with complexity metrics for different input sizes
        """
        report: Dict[str, Any] = {
            "model_name": type(model).__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "model_size_mb": sum(p.numel() for p in model.parameters())
            * 4
            / (1024 * 1024),
            "input_variations": {},  # type: Dict[str, Dict[str, Any]]
        }

        for name, shape in input_shapes.items():
            try:
                # Calculate FLOPs
                op_counts = ModelVisualizer.count_ops(model, shape)

                # Create a dummy input tensor
                batch_size = 1
                full_shape = (batch_size,) + shape
                dummy_input = torch.randn(full_shape, requires_grad=False)

                # Measure inference time
                model.eval()
                device = next(model.parameters()).device
                dummy_input = dummy_input.to(device)

                # Warm-up
                with torch.no_grad():
                    model(dummy_input)

                # Timing
                import time

                num_runs = 5
                start = time.time()
                with torch.no_grad():
                    for _ in range(num_runs):
                        model(dummy_input)
                avg_time = (time.time() - start) / num_runs

                report["input_variations"][name] = {
                    "input_shape": shape,
                    "flops": op_counts["total_flops"],
                    "inference_time": avg_time,
                    "flops_per_second": op_counts["total_flops"] / avg_time
                    if avg_time > 0
                    else 0,
                }

            except Exception as e:
                report["input_variations"][name] = {
                    "input_shape": shape,
                    "error": str(e),
                }
                logger.warning(
                    f"Error analyzing complexity for input size {name}: {str(e)}"
                )

        return report
