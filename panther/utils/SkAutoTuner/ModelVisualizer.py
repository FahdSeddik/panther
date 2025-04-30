import torch.nn as nn

class ModelVisualizer:
    """
    A utility class for visualizing PyTorch model structures.
    """
    
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