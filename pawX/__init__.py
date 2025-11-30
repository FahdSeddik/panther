from .verify import ensure_load

ensure_load()

from .pawX import (  # noqa: E402; BARRIER NO FORMAT; BARRIER; BARRIER NO FORMAT; BARRIER NO FORMAT
    Axis,
    DistributionFamily,
    causal_denominator_backward,
    causal_denominator_forward,
    causal_numerator_backward,
    causal_numerator_forward,
    count_skop,
    cqrrpt,
    create_projection_matrix,
    dense_sketch_operator,
    gaussian_skop,
    randomized_svd,
    rmha_forward,
    scaled_sign_sketch,
    sinSRPE,
    sjlt_skop,
    sketch_tensor,
    sketched_conv2d_backward,
    sketched_conv2d_forward,
    sketched_linear_backward,
    sketched_linear_forward,
    sparse_sketch_operator,
    srht,
)

# Conditionally import CUDA-only functions
try:
    from .pawX import test_tensor_accessor

    _HAS_CUDA_FEATURES = True
except ImportError:
    _HAS_CUDA_FEATURES = False
    test_tensor_accessor = None  # type: ignore

__all__ = [
    "scaled_sign_sketch",
    "sketched_linear_backward",
    "sketched_linear_forward",
    "cqrrpt",
    "randomized_svd",
    "DistributionFamily",
    "dense_sketch_operator",
    "sketch_tensor",
    "rmha_forward",
    "create_projection_matrix",
    "sketched_conv2d_backward",
    "sketched_conv2d_forward",
    "causal_numerator_forward",
    "causal_numerator_backward",
    "causal_denominator_forward",
    "causal_denominator_backward",
    "sparse_sketch_operator",
    "Axis",
    "gaussian_skop",
    "count_skop",
    "sjlt_skop",
    "srht",
    "sinSRPE",
]

# Only add CUDA-specific functions to __all__ if they're available
if _HAS_CUDA_FEATURES:
    __all__.append("test_tensor_accessor")
