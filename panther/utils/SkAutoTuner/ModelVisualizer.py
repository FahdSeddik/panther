import torch.nn as nn
import os
import graphviz
import tempfile
import webbrowser
import json
import uuid
from typing import Dict, Any, Optional, Tuple
import pkg_resources

class ModelVisualizer:
    """
    A utility class for visualizing PyTorch model structures.
    """
    
    # Get the path to the visualization assets
    _ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualization_assets')
    
    @staticmethod
    def _build_module_tree(named_modules):
        """
        Builds a nested dict representing the module hierarchy.
        named_modules: iterable of (full_name, module) from model.named_modules()
        """
        tree = {}
        module_types = {}
        
        for full_name, module in named_modules:
            # Store module type
            module_types[full_name] = type(module).__name__
            
            # Build tree structure
            parts = full_name.split('.') if full_name else []
            subtree = tree
            for part in parts:
                subtree = subtree.setdefault(part, {})
                
        return tree, module_types

    @staticmethod
    def _print_tree(subtree, module_types, full_path='', prefix='', is_last=True):
        """
        Recursively prints the nested dict as an ASCII tree with module types.
        """
        # Choose branch characters
        branch = '└─ ' if is_last else '├─ '
        for idx, (name, child) in enumerate(sorted(subtree.items())):
            is_child_last = (idx == len(subtree) - 1)
            
            # Calculate the full path for this module
            current_path = f"{full_path}.{name}" if full_path else name
            
            # Get the module type (if available)
            module_type = f" ({module_types.get(current_path, '')})" if current_path in module_types else ""
            
            print(prefix + branch + name + module_type + ('/' if child else ''))
            
            # Prepare the prefix for children
            if child:
                extension = '    ' if is_child_last else '│   '
                ModelVisualizer._print_tree(child, module_types, current_path, prefix + extension, True)

    @staticmethod
    def print_module_tree(model: nn.Module, root_name: str = 'model'):
        """
        Prints the modules of a PyTorch model in a tree structure with their types.
        
        Example:
            ModelVisualizer.print_module_tree(my_model)
        """
        # Build tree from module names and get module types
        tree, module_types = ModelVisualizer._build_module_tree(model.named_modules())
        
        # Add root model type
        module_types[''] = type(model).__name__
        
        # Print the root
        print(f"{root_name} ({module_types.get('', '')})/")
        
        # Print its children
        ModelVisualizer._print_tree(tree, module_types)

    @staticmethod
    def _collect_module_info(model: nn.Module) -> Dict[str, Dict[str, Any]]:
        """
        Collects detailed information about each module in the model.
        
        Returns:
            A dictionary where keys are full module names and values are
            dictionaries containing module information.
        """
        module_info = {}
        
        for name, module in model.named_modules():
            if name == '':  # Skip the root module
                continue
                
            # Collect basic info
            try:
                param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                is_trainable = any(p.requires_grad for p in module.parameters())
            except Exception:
                param_count = 0
                is_trainable = False
                
            info = {
                'type': type(module).__name__,
                'parameters': param_count,
                'trainable': is_trainable,
                'class': str(type(module)),
            }
            
            # Add specific module type information
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
            elif isinstance(module, nn.RNN) or isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
                info.update({
                    'input_size': module.input_size,
                    'hidden_size': module.hidden_size,
                    'num_layers': module.num_layers,
                    'bidirectional': module.bidirectional,
                    'dropout': module.dropout,
                })
            elif isinstance(module, nn.Dropout):
                info.update({
                    'p': module.p,
                    'inplace': module.inplace,
                })
            elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d):
                info.update({
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding,
                })
            
            module_info[name] = info
            
        return module_info
    
    @staticmethod
    def create_interactive_visualization(model: nn.Module, output_path: Optional[str] = None, 
                                        graph_attrs: Optional[Dict[str, str]] = None,
                                        node_attrs: Optional[Dict[str, str]] = None,
                                        open_browser: bool = True) -> str:
        """
        Creates an interactive visualization of the model structure.
        
        When clicked, nodes will copy the full module name to clipboard and display
        detailed information about the module.
        
        Args:
            model (nn.Module): The PyTorch model to visualize
            output_path (str, optional): Path to save the HTML file. If None, a temporary file is used.
            graph_attrs (Dict[str, str], optional): Attributes for the graph
            node_attrs (Dict[str, str], optional): Default attributes for nodes
            open_browser (bool): Whether to open the visualization in a browser automatically
            
        Returns:
            str: Path to the generated HTML file
        """
        # Set default attributes if not provided
        if graph_attrs is None:
            graph_attrs = {
                'rankdir': 'TB',
                'bgcolor': 'transparent',
                'splines': 'ortho',
                'fontname': 'Arial',
                'fontsize': '14',
                'nodesep': '0.5',
                'ranksep': '0.7',
                'concentrate': 'true',
            }
            
        if node_attrs is None:
            node_attrs = {
                'style': 'filled',
                'shape': 'box',
                'fillcolor': '#E5F5FD',
                'color': '#4285F4',
                'fontname': 'Arial',
                'fontsize': '12',
                'height': '0.4',
            }
        
        # Create a directed graph
        dot = graphviz.Digraph(
            'model_visualization', 
            format='svg',
            graph_attr=graph_attrs
        )
        dot.attr('node', **node_attrs)
        
        # Get module information
        tree, module_types = ModelVisualizer._build_module_tree(model.named_modules())
        module_info = ModelVisualizer._collect_module_info(model)
        
        # Add root model info to module_info
        root_type = type(model).__name__
        module_info['root'] = {
            'type': root_type,
            'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'trainable': any(p.requires_grad for p in model.parameters()),
            'class': str(type(model)),
        }
        
        # Add the root node
        root_id = "node_root"
        dot.node(root_id, f'{root_type}', tooltip=f'Root: {root_type}', 
                id='node_root', data_name='root')
        
        # Mapping from module path to node ID
        node_ids = {'root': root_id}
        
        # Process the modules and build the graph
        def add_nodes(subtree, parent_path=''):
            for name, child in sorted(subtree.items()):
                current_path = f"{parent_path}.{name}" if parent_path else name
                
                # Create node ID that's valid for HTML
                node_id = f"node_{current_path.replace('.', '_')}"
                node_ids[current_path] = node_id
                
                # Get module type and add node
                module_type = module_types.get(current_path, "")
                label = f"{name} ({module_type})" if module_type else name
                
                # Create tooltip with basic info
                tooltip = f"Name: {current_path}\nType: {module_type}"
                if current_path in module_info:
                    info = module_info[current_path]
                    tooltip += f"\nParameters: {info['parameters']:,}"
                
                # Add the node with data attributes for JavaScript interactivity
                fillcolor = "#E5F5FD" if child else "#C2E0F4"  # Different color for leaf nodes
                dot.node(node_id, label, tooltip=tooltip, fillcolor=fillcolor,
                        id=node_id, data_name=current_path)
                
                # Connect to parent
                parent_id = node_ids[parent_path] if parent_path else root_id
                edge_id = f"edge_{parent_id}_{node_id}"
                dot.edge(parent_id, node_id, id=edge_id, data_source=parent_id, data_target=node_id)
                
                # Process children
                if child:
                    add_nodes(child, current_path)
        
        # Add all nodes starting from root
        add_nodes(tree)
        
        # Generate the SVG
        svg_content = dot.pipe().decode('utf-8')
        
        # Ensure nodes have proper data-name attributes by modifying the SVG
        # This step is crucial to make the interactivity work
        import re
        for node_path, node_id in node_ids.items():
            # Create a pattern to find the node group in the SVG
            node_group_pattern = rf'(<g[^>]*id="{node_id}"[^>]*)(>)'
            data_name_attr = f'data-name="{node_path}"'
            # Add data-name to the <g> if missing
            svg_content = re.sub(
                node_group_pattern,
                lambda m: m.group(1) + (f' {data_name_attr}' if data_name_attr not in m.group(1) else '') + m.group(2),
                svg_content
            )
            # Also add data-name to all direct children of this <g> (rect, text, etc.)
            # Find the <g ...>...</g> block
            g_block_pattern = rf'(<g[^>]*id="{node_id}"[^>]*>)([\s\S]*?)(</g>)'
            def add_data_name_to_children(match):
                g_open, g_content, g_close = match.groups()
                # Add data-name to all child elements (rect, text, polygon, etc.)
                g_content = re.sub(r'(<(rect|polygon|ellipse|text|title|path|circle)\b[^>]*)(/?>)',
                    lambda m2: m2.group(1) + (f' {data_name_attr}' if data_name_attr not in m2.group(1) else '') + m2.group(3),
                    g_content)
                return g_open + g_content + g_close
            svg_content = re.sub(g_block_pattern, add_data_name_to_children, svg_content)
        
        # Prepare the module info for JavaScript
        js_module_info = json.dumps(module_info)
        
        # Determine the output file path
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix='.html')
            os.close(fd)
        
        # Read HTML template
        template_path = os.path.join(ModelVisualizer._ASSETS_PATH, 'template.html')
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
            
        # Read CSS
        css_path = os.path.join(ModelVisualizer._ASSETS_PATH, 'css', 'styles.css')
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
            
        # Read JavaScript
        js_path = os.path.join(ModelVisualizer._ASSETS_PATH, 'js', 'script.js')
        with open(js_path, 'r', encoding='utf-8') as f:
            js_content = f.read()
        
        # Replace placeholders in template
        html_content = template_content.replace('{{SVG_CONTENT}}', svg_content)
        html_content = html_content.replace('{{MODULE_INFO}}', js_module_info)
        html_content = html_content.replace('{{CSS_CONTENT}}', css_content)
        html_content = html_content.replace('{{JS_CONTENT}}', js_content)
        
        # Write the HTML to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Open in browser if requested
        if open_browser:
            webbrowser.open('file://' + os.path.abspath(output_path))
            
        return output_path