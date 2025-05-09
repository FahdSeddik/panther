import torch.nn as nn
import os
import re
import json
import tempfile
import webbrowser
import logging
from typing import Dict, Any, Optional, Tuple

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
    _ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualization_assets')

    @staticmethod
    def _build_module_tree(named_modules: iter) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Builds a nested dict representing the module hierarchy and collects module types.
        """
        tree = {}
        module_types = {}
        
        for full_name, module in named_modules:
            module_types[full_name] = type(module).__name__
            parts = full_name.split('.') if full_name else []
            current_level = tree
            for part in parts:
                current_level = current_level.setdefault(part, {})
                
        return tree, module_types
    
    @staticmethod
    def _print_tree(subtree: Dict[str, Any], module_types: Dict[str, str], 
                    full_path: str = '', prefix: str = '', is_last: bool = True):
        """
        Recursively prints the nested dict as an ASCII tree with module types.
        """
        branch = '└─ ' if is_last else '├─ '
        for idx, (name, child) in enumerate(sorted(subtree.items())):
            is_child_last = (idx == len(subtree) - 1)
            current_path = f"{full_path}.{name}" if full_path else name
            module_type = f" ({module_types.get(current_path, 'UnknownType')})" if current_path in module_types else ""
            print(prefix + branch + name + module_type + ('/' if child else ''))
            if child:
                extension = '    ' if is_child_last else '│   '
                ModelVisualizer._print_tree(child, module_types, current_path, prefix + extension, is_child_last)
    
    @staticmethod
    def print_module_tree(model: nn.Module, root_name: str = 'model'):
        """
        Prints the modules of a PyTorch model in a tree structure with their types.
        """
        tree, module_types = ModelVisualizer._build_module_tree(model.named_modules())
        module_types[''] = type(model).__name__
        print(f"{root_name} ({module_types.get('', 'UnknownType')})/")
        ModelVisualizer._print_tree(tree, module_types, full_path=root_name)
    
    @staticmethod
    def _collect_module_info(model: nn.Module) -> Dict[str, Dict[str, Any]]:
        """
        Collects detailed information about each module in the model.
        """
        module_info = {}

        try:
            root_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            root_is_trainable = any(p.requires_grad for p in model.parameters())
        except AttributeError:
            root_param_count = 0
            root_is_trainable = False
            logger.warning("Could not retrieve parameter info for the root model.")

        module_info['root'] = {
            'type': type(model).__name__,
            'parameters': root_param_count,
            'trainable': root_is_trainable,
            'class': str(type(model)),
            'docstring': model.__doc__.strip().split('\n')[0] if model.__doc__ else "N/A"
        }
        
        for name, module in model.named_modules():
            if not name:
                continue
                
            try:
                param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                is_trainable = any(p.requires_grad for p in module.parameters())
            except AttributeError:
                param_count = 0
                is_trainable = False
                logger.warning(f"Module {name} ({type(module).__name__}) does not have 'parameters' attribute or it's not iterable.")
            except RuntimeError:
                param_count = 0
                is_trainable = False
                logger.warning(f"Could not count parameters for module {name} ({type(module).__name__}).")

            info = {
                'type': type(module).__name__,
                'parameters': param_count,
                'trainable': is_trainable,
                'class': str(type(module)),
                'docstring': module.__doc__.strip().split('\n')[0] if module.__doc__ else "N/A"
            }
            
            if isinstance(module, nn.Conv2d):
                info.update({
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding,
                    'groups': module.groups,
                    'dilation': module.dilation,
                })
            elif isinstance(module, nn.Linear):
                info.update({
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'bias': module.bias is not None,
                })
            elif isinstance(module, nn.BatchNorm2d):
                info.update({
                    'num_features': module.num_features,
                    'eps': module.eps,
                    'momentum': module.momentum,
                    'affine': module.affine,
                })
            elif isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
                info.update({
                    'input_size': module.input_size,
                    'hidden_size': module.hidden_size,
                    'num_layers': module.num_layers,
                    'bidirectional': module.bidirectional,
                    'dropout': module.dropout if hasattr(module, 'dropout') else 0,
                    'bias': module.bias
                })
            elif isinstance(module, nn.Dropout):
                info.update({
                    'p': module.p,
                    'inplace': module.inplace,
                })
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
                info_pool = {
                    'kernel_size': getattr(module, 'kernel_size', 'N/A'),
                    'stride': getattr(module, 'stride', 'N/A'),
                    'padding': getattr(module, 'padding', 'N/A'),
                }
                if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
                    info_pool['output_size'] = module.output_size
                info.update(info_pool)
            elif isinstance(module, nn.Embedding):
                info.update({
                    'num_embeddings': module.num_embeddings,
                    'embedding_dim': module.embedding_dim,
                    'padding_idx': module.padding_idx,
                })
            elif isinstance(module, nn.LayerNorm):
                info.update({
                    'normalized_shape': module.normalized_shape,
                    'eps': module.eps,
                    'elementwise_affine': module.elementwise_affine,
                })
            elif isinstance(module, nn.MultiheadAttention):
                info.update({
                    'embed_dim': module.embed_dim,
                    'num_heads': module.num_heads,
                    'dropout': module.dropout,
                    'bias': hasattr(module, 'bias_k') and module.bias_k is not None,
                    'add_bias_kv': hasattr(module, 'add_bias_kv') and module.add_bias_kv,
                    'add_zero_attn': hasattr(module, 'add_zero_attn') and module.add_zero_attn,
                    'kdim': getattr(module, 'kdim', None),
                    'vdim': getattr(module, 'vdim', None),
                })
            elif isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                info.update({
                    'd_model': module.self_attn.embed_dim if hasattr(module, 'self_attn') else getattr(module, 'd_model', 'N/A'),
                    'nhead': module.self_attn.num_heads if hasattr(module, 'self_attn') else getattr(module, 'nhead', 'N/A'),
                    'dim_feedforward': module.linear1.out_features if hasattr(module, 'linear1') else getattr(module, 'dim_feedforward', 'N/A'),
                    'dropout': module.dropout.p if hasattr(module, 'dropout') else getattr(module, 'dropout', 'N/A'),
                    'activation': type(module.activation).__name__ if hasattr(module, 'activation') else getattr(module, 'activation', 'N/A')
                })
                if isinstance(module, nn.TransformerDecoderLayer):
                     info['cross_attention'] = True
            
            module_info[name] = info
            
        return module_info
    
    @staticmethod
    def create_interactive_visualization(model: nn.Module, output_path: Optional[str] = None, 
                                        graph_attrs: Optional[Dict[str, str]] = None,
                                        node_attrs: Optional[Dict[str, str]] = None,
                                        edge_attrs: Optional[Dict[str, str]] = None,
                                        open_browser: bool = True,
                                        max_label_length: int = 30) -> str:
        """
        Creates an interactive visualization of the model structure.
        """
        if Digraph is None:
            raise ImportError("The graphviz Python package is required. Install with 'pip install graphviz'.")

        default_graph_attrs = {
            'rankdir': 'TB', 'bgcolor': 'transparent', 'splines': 'ortho',
            'fontname': 'Arial, Helvetica, sans-serif', 'fontsize': '14',
            'nodesep': '0.6', 'ranksep': '0.8', 'concentrate': 'true',
            'overlap': 'false',
        }
        if graph_attrs:
            default_graph_attrs.update(graph_attrs)
            
        default_node_attrs = {
            'style': 'filled,rounded', 'shape': 'box', 'fillcolor': '#E5F5FD', 
            'color': '#4285F4', 'fontname': 'Arial, Helvetica, sans-serif', 
            'fontsize': '11', 'height': '0.4', 'margin': '0.1,0.05'
        }
        if node_attrs:
            default_node_attrs.update(node_attrs)

        default_edge_attrs = {
            'color': '#757575', 'arrowsize': '0.7'
        }
        if edge_attrs:
            default_edge_attrs.update(edge_attrs)
        
        dot = Digraph(
            'model_visualization', 
            format='svg',
            graph_attr=default_graph_attrs
        )
        dot.attr('node', **default_node_attrs)
        dot.attr('edge', **default_edge_attrs)
        
        raw_named_modules = list(model.named_modules())
        tree, module_types = ModelVisualizer._build_module_tree(raw_named_modules)
        module_info = ModelVisualizer._collect_module_info(model)
        
        root_name_for_graph = 'model'
        root_display_type = module_info['root']['type']
        root_label = f"{root_name_for_graph} ({root_display_type})"
        if len(root_label) > max_label_length:
            root_label = root_label[:max_label_length-3] + "..."
        
        root_id = "node_root_model"
        dot.node(root_id, root_label, tooltip=f'Root: {root_display_type}\nParameters: {module_info["root"]["parameters"]:,}',
                 id=root_id, data_name='root', fillcolor='#D1E7F7', shape='Mrecord')
        
        node_ids = {'root': root_id}
        
        def add_nodes_and_edges(current_subtree, parent_full_path, parent_node_id):
            for name_part, children_subtree in sorted(current_subtree.items()):
                current_full_path = f"{parent_full_path}.{name_part}" if parent_full_path != 'root' else name_part
                node_id = f"node_{current_full_path.replace('.', '_').replace('-', '_')}"
                node_ids[current_full_path] = node_id
                module_type_name = module_types.get(current_full_path, "Unknown")
                
                label = f"{name_part} ({module_type_name})"
                if len(label) > max_label_length:
                    label = label[:max_label_length-3] + "..."
                tooltip_parts = [f"Name: {current_full_path}", f"Type: {module_type_name}"]
                current_module_details = module_info.get(current_full_path)
                if current_module_details:
                    tooltip_parts.append(f"Parameters: {current_module_details['parameters']:,}")
                    tooltip_parts.append(f"Trainable: {'Yes' if current_module_details['trainable'] else 'No'}")
                node_fillcolor = default_node_attrs.get('fillcolor', '#E5F5FD')
                if not children_subtree:
                    node_fillcolor = "#C2E0F4" 
                
                dot.node(node_id, label, tooltip='\n'.join(tooltip_parts), fillcolor=node_fillcolor,
                         id=node_id, data_name=current_full_path)
                edge_id = f"edge_{parent_node_id}_{node_id}"
                dot.edge(parent_node_id, node_id, id=edge_id, data_source=parent_node_id, data_target=node_id)
                if children_subtree:
                    add_nodes_and_edges(children_subtree, current_full_path, node_id)
        
        add_nodes_and_edges(tree, 'root', root_id)
        
        svg_content_bytes = dot.pipe(format='svg')
        svg_content = svg_content_bytes.decode('utf-8')
        
        for node_path_key, node_html_id in node_ids.items():
            data_name_attr_str = f'data-name="{node_path_key}"'
            g_block_pattern = rf'(<g[^>]*id="{re.escape(node_html_id)}"[^>]*>)([\s\S]*?)(</g>)'
            
            def process_g_block(match_obj):
                g_open_tag, g_content, g_close_tag = match_obj.groups()
                if data_name_attr_str not in g_open_tag:
                    g_open_tag = g_open_tag.rstrip('>') + f' {data_name_attr_str}>'
                def add_data_to_visual_child(child_match):
                    child_tag_open, child_tag_rest = child_match.groups()
                    if data_name_attr_str not in child_tag_open:
                        return child_tag_open + f' {data_name_attr_str}' + child_tag_rest
                    return child_match.group(0)
                
                g_content = re.sub(r'(<(?:rect|polygon|ellipse|text|path|circle)\b[^>]*?)(/?>)', 
                                   add_data_to_visual_child, 
                                   g_content)
                return g_open_tag + g_content + g_close_tag
            
            svg_content = re.sub(g_block_pattern, process_g_block, svg_content)

        js_module_info = json.dumps(module_info)
        
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix='.html')
            os.close(fd)
        
        template_path = os.path.join(ModelVisualizer._ASSETS_PATH, 'template.html')
        if not os.path.exists(template_path):
            logger.error(f"HTML template not found at {template_path}")
            raise FileNotFoundError(f"HTML template not found at {template_path}")
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
            
        css_path = os.path.join(ModelVisualizer._ASSETS_PATH, 'css', 'styles.css')
        if not os.path.exists(css_path):
            logger.error(f"CSS file not found at {css_path}")
            raise FileNotFoundError(f"CSS file not found at {css_path}")
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
            
        js_path = os.path.join(ModelVisualizer._ASSETS_PATH, 'js', 'script.js')
        if not os.path.exists(js_path):
            logger.error(f"JavaScript file not found at {js_path}")
            raise FileNotFoundError(f"JavaScript file not found at {js_path}")
        with open(js_path, 'r', encoding='utf-8') as f:
            js_content = f.read()
            
        html_content = template_content.replace('{{SVG_CONTENT}}', svg_content)
        html_content = html_content.replace('{{MODULE_INFO}}', js_module_info)
        html_content = html_content.replace('{{CSS_CONTENT}}', css_content)
        html_content = html_content.replace('{{JS_CONTENT}}', js_content)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        if open_browser:
            try:
                url = 'file://' + os.path.abspath(output_path)
                webbrowser.open(url)
                logger.info(f"Visualization opened in browser: {url}")
            except Exception as e:
                logger.warning(f"Could not automatically open browser: {e}")
            
        logger.info(f"Interactive model visualization saved to: {output_path}")
        return output_path