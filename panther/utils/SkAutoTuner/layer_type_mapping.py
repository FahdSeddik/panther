import torch.nn as nn
from panther.nn.linear import SKLinear
from panther.nn.conv2d import SKConv2d
from panther.nn.attention import RandMultiHeadAttention

# Layer type mapping to their sketched versions and parameters
LAYER_TYPE_MAPPING = {
    nn.Linear: {
        "class": SKLinear,
        "params": ["num_terms", "low_rank"],
    },
    nn.Conv2d: {
        "class": SKConv2d,
        "params": ["num_terms", "low_rank"],
    },
    nn.MultiheadAttention: {
        "class": RandMultiHeadAttention,
        "params": ["num_random_features", "kernel_fn"],
    }
}