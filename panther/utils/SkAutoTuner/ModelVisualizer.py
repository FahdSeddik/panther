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
        for full_name, _ in named_modules:
            parts = full_name.split('.') if full_name else []
            subtree = tree
            for part in parts:
                subtree = subtree.setdefault(part, {})
        return tree

    @staticmethod
    def _print_tree(subtree, prefix='', is_last=True):
        """
        Recursively prints the nested dict as an ASCII tree.
        """
        # Choose branch characters
        branch = '└─ ' if is_last else '├─ '
        for idx, (name, child) in enumerate(sorted(subtree.items())):
            is_child_last = (idx == len(subtree) - 1)
            print(prefix + branch + name + ('/' if child else ''))
            # Prepare the prefix for children
            if child:
                extension = '    ' if is_child_last else '│   '
                ModelVisualizer._print_tree(child, prefix + extension, True)

    @staticmethod
    def print_module_tree(model: nn.Module, root_name: str = 'model'):
        """
        Prints the modules of a PyTorch model in a tree structure.
        
        Example:
            ModelVisualizer.print_module_tree(my_model)
        """
        # Build tree from module names
        tree = ModelVisualizer._build_module_tree(model.named_modules())
        # Print the root
        print(f"{root_name}/")
        # Print its children
        ModelVisualizer._print_tree(tree)