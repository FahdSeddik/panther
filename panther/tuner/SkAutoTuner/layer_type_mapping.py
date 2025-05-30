import torch.nn as nn

# Use string references instead of direct imports to avoid circular dependencies
LAYER_TYPE_MAPPING = {
    nn.Linear: {
        "class_path": "panther.nn.linear.SKLinear",
        "params": ["num_terms", "low_rank"],
    },
    nn.Conv2d: {
        "class_path": "panther.nn.conv2d.SKConv2d",
        "params": ["num_terms", "low_rank"],
    },
    nn.MultiheadAttention: {
        "class_path": "panther.nn.attention.RandMultiHeadAttention",
        "params": ["num_random_features", "kernel_fn"],
    },
}


def get_sketched_class(layer_type):
    """Dynamically import and return the sketched class for a layer type."""
    if layer_type not in LAYER_TYPE_MAPPING:
        raise ValueError(f"Layer type {layer_type.__name__} is not supported")

    class_path = LAYER_TYPE_MAPPING[layer_type]["class_path"]
    module_path, class_name = class_path.rsplit(".", 1)

    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)
