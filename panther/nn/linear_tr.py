import math
from typing import Any, List, Tuple

import torch
import torch.nn as nn
import triton
from torch.autograd import Function
from torch.library import triton_op, wrap_triton
from torch.nn import init

from panther.sketch import scaled_sign_sketch as gen_U

from .linear_kernels import (
    calc_grad_S1s_kernel,
    calc_grad_S2s_kernel,
    first_pass_gU1s_g_S2s_kernel,
    first_pass_kernel,
    first_pass_U2s_hin_kernel,
    second_pass_gUS11_22_kernel,
    second_pass_kernel,
)


@triton_op("panther::forward_op", mutates_args={})
def forward_op(
    hin: torch.Tensor,
    S1s: torch.Tensor,
    S2s: torch.Tensor,
    U1s: torch.Tensor,
    U2s: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Forward pass of the sketched linear layer implemented with Triton.

    This implements the forward computation: out = 1/2 * (hin @ S1s @ U1s + hin @ U2s @ S2s) + bias

    Args:
        hin: Input tensor of shape (batch_size, in_features)
        S1s: Learnable parameters of shape (num_terms, in_features, low_rank)
        S2s: Learnable parameters of shape (num_terms, low_rank, out_features)
        U1s: Fixed random matrices of shape (num_terms, low_rank, out_features)
        U2s: Fixed random matrices of shape (num_terms, in_features, low_rank)
        bias: Bias term of shape (out_features)

    Returns:
        Output tensor of shape (batch_size, out_features)
    """
    device = "cuda"

    # First pass: Compute intermediate results for both paths
    # Computes: in1 = hin @ S1s and in2 = hin @ U2s
    ############
    L = S2s.shape[0]  # Number of terms
    BSIZE, d2 = hin.shape  # Batch size, input dimension
    L, _, K = S1s.shape  # Number of terms, input dimension, low rank

    # Allocate memory for intermediate results
    in1 = torch.empty((L, BSIZE, K), dtype=torch.float32, device=device)
    in2 = torch.empty((L, BSIZE, K), dtype=torch.float32, device=device)

    # Calculate tensor strides for efficient memory access
    stride_hin_bsize, stride_hin_d2 = hin.stride(0), hin.stride(1)
    stride_su_l, stride_su_d2, stride_su_k = S1s.stride(0), S1s.stride(1), S1s.stride(2)
    stride_out_l, stride_out_bsize, stride_out_k = (
        in1.stride(0),
        in1.stride(1),
        in1.stride(2),
    )

    # Define the computational grid for the Triton kernel
    grid = lambda META: (
        L,
        triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"])
        * triton.cdiv(K, META["BLOCK_SIZE_K"]),
    )

    # Launch the first pass kernel to compute intermediate tensors
    wrap_triton(first_pass_kernel)[grid](
        hin,
        S1s,
        U2s,
        in1,
        in2,
        BSIZE,
        K,
        d2,
        L,
        stride_hin_bsize,
        stride_hin_d2,
        stride_su_l,
        stride_su_d2,
        stride_su_k,
        stride_out_l,
        stride_out_bsize,
        stride_out_k,
    )

    # Second pass: Compute final output
    # Computes: out = (in1 @ U1s + in2 @ S2s) / 2 + bias
    #############
    bias_unsqueezed = bias.unsqueeze(0)  # Add dimension for broadcasting
    L, BSIZE, K = in1.shape
    _, _, d1 = U1s.shape  # Output dimension

    # Allocate memory for the output
    out = torch.empty((BSIZE, d1), dtype=torch.float32, device=device)

    # Calculate tensor strides for efficient memory access
    stride_in12_l, stride_in12_bsize, stride_in12_k = (
        in1.stride(0),
        in1.stride(1),
        in1.stride(2),
    )
    stride_us_l, stride_us_k, stride_us_d1 = U1s.stride(0), U1s.stride(1), U1s.stride(2)
    stride_bias_bsize, stride_bias_d1 = (
        bias_unsqueezed.stride(0),
        bias_unsqueezed.stride(1),
    )
    stride_out_bsize, stride_out_d1 = out.stride(0), out.stride(1)

    # Define the computational grid for the Triton kernel
    grid = lambda META: (
        triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"])
        * triton.cdiv(d1, META["BLOCK_SIZE_D1"]),
    )

    # Launch the second pass kernel to compute the final output
    wrap_triton(second_pass_kernel)[grid](
        in1,
        in2,
        U1s,
        S2s,
        bias_unsqueezed,
        out,
        BSIZE,
        d1,
        K,
        L,
        stride_in12_l,
        stride_in12_bsize,
        stride_in12_k,
        stride_us_l,
        stride_us_k,
        stride_us_d1,
        stride_bias_bsize,
        stride_bias_d1,
        stride_out_bsize,
        stride_out_d1,
    )

    return out


@forward_op.register_kernel("cpu")
def _(input, S1s, S2s, U1s, U2s, bias):
    """CPU implementation of the forward pass.

    This is used as a fallback when CUDA is not available.
    """
    num_terms = S2s.shape[0]
    # Efficiently perform the sum over all l terms
    input = input.unsqueeze(0).expand(num_terms, input.shape[0], input.shape[1])
    return (
        ((input.bmm(S1s)).bmm(U1s)).mean(0) / 2  # First term: input @ S1s @ U1s
        + ((input.bmm(U2s)).bmm(S2s)).mean(0) / 2  # Second term: input @ U2s @ S2s
        + bias  # Add bias
    )


@triton_op("panther::backward_op", mutates_args={})
def backward_op(
    hin: torch.Tensor,
    S1s: torch.Tensor,
    S2s: torch.Tensor,
    U1s: torch.Tensor,
    U2s: torch.Tensor,
    g: torch.Tensor,
) -> List[torch.Tensor]:
    """Backward pass of the sketched linear layer implemented with Triton.

    Computes gradients with respect to input, S1s, S2s, and bias.

    Args:
        hin: Input tensor from the forward pass
        S1s: Learnable parameters
        S2s: Learnable parameters
        U1s: Fixed random matrices
        U2s: Fixed random matrices
        g: Gradient from the next layer

    Returns:
        List of tensors: [grad_input, grad_S1s, grad_S2s, grad_bias]
    """
    device = "cuda"
    num_terms = S2s.shape[0]

    # Transpose tensors to align dimensions for matrix multiplication
    hin = hin.transpose(0, 1)
    U1s = U1s.transpose(1, 2)
    S1s = S1s.transpose(1, 2)
    U2s = U2s.transpose(1, 2)
    S2s = S2s.transpose(1, 2)

    # First pass: Compute intermediate gradients
    # Computes g_U1s = g @ U1s and g_S2s = g @ S2s
    #######################
    BSIZE, d1 = g.shape  # Batch size, output dimension
    L, _, K = U1s.shape  # Number of terms, output dimension, low rank

    # Allocate memory for intermediate results
    g_U1s = torch.empty((L, BSIZE, K), dtype=torch.float32, device="cuda")
    g_S2s = torch.empty((L, BSIZE, K), dtype=torch.float32, device="cuda")

    # Calculate tensor strides for efficient memory access
    stride_g_bsize, stride_g_d1 = g.stride(0), g.stride(1)
    stride_su_l, stride_su_d1, stride_su_k = U1s.stride(0), U1s.stride(1), U1s.stride(2)
    stride_out_l, stride_out_bsize, stride_out_k = (
        g_U1s.stride(0),
        g_U1s.stride(1),
        g_U1s.stride(2),
    )

    # Define the computational grid for the Triton kernel
    grid = lambda META: (
        L,
        triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"])
        * triton.cdiv(K, META["BLOCK_SIZE_K"]),
    )

    # Launch the kernel to compute intermediate gradients
    wrap_triton(first_pass_gU1s_g_S2s_kernel)[grid](
        g,
        U1s,
        S2s,
        g_U1s,
        g_S2s,
        BSIZE,
        K,
        d1,
        L,
        stride_g_bsize,
        stride_g_d1,
        stride_su_l,
        stride_su_d1,
        stride_su_k,
        stride_out_l,
        stride_out_bsize,
        stride_out_k,
    )

    # Second pass: Compute gradient with respect to input
    # Computes grad = g_U1s @ S1s + g_S2s @ U2s
    #######################
    L, BSIZE, K = g_U1s.shape
    _, _, d2 = S1s.shape  # Input dimension

    # Allocate memory for the input gradient
    grad = torch.empty((BSIZE, d2), dtype=torch.float32, device="cuda")

    # Calculate tensor strides for efficient memory access
    stride_g_U1s2_l, stride_g_U1s2_bsize, stride_g_U1s2_k = (
        g_U1s.stride(0),
        g_U1s.stride(1),
        g_U1s.stride(2),
    )
    stride_us_l, stride_us_k, stride_us_d2 = S1s.stride(0), S1s.stride(1), S1s.stride(2)
    stride_out_bsize, stride_out_d2 = grad.stride(0), grad.stride(1)

    # Define the computational grid for the Triton kernel
    grid = lambda META: (
        triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"])
        * triton.cdiv(d2, META["BLOCK_SIZE_d2"]),
    )

    # Launch the kernel to compute input gradients
    wrap_triton(second_pass_gUS11_22_kernel)[grid](
        g_U1s,
        g_S2s,
        S1s,
        U2s,
        grad,
        BSIZE,
        d2,
        K,
        L,
        stride_g_U1s2_l,
        stride_g_U1s2_bsize,
        stride_g_U1s2_k,
        stride_us_l,
        stride_us_k,
        stride_us_d2,
        stride_out_bsize,
        stride_out_d2,
    )

    # Compute gradient with respect to S1s
    # Computes grad_S1s = hin @ g_U1s
    ################
    d2, BSIZE = hin.shape  # Input dimension, batch size
    L, _, k = g_U1s.shape  # Number of terms, batch size, low rank

    # Allocate memory for S1s gradient
    grad_S1s = torch.empty((L, d2, k), dtype=torch.float32, device=device)

    # Calculate tensor strides for efficient memory access
    stride_hin_bsize, stride_hin_BSIZE = hin.stride(0), hin.stride(1)
    stride_su_l, stride_su_BSIZE, stride_su_k = (
        g_U1s.stride(0),
        g_U1s.stride(1),
        g_U1s.stride(2),
    )
    stride_out_l, stride_out_bsize, stride_out_k = (
        grad_S1s.stride(0),
        grad_S1s.stride(1),
        grad_S1s.stride(2),
    )

    # Define the computational grid for the Triton kernel
    grid = lambda META: (
        L,
        triton.cdiv(d2, META["BLOCK_SIZE_d2"]) * triton.cdiv(k, META["BLOCK_SIZE_k"]),
    )

    # Launch the kernel to compute S1s gradients
    wrap_triton(calc_grad_S1s_kernel)[grid](
        hin,
        g_U1s,
        grad_S1s,
        d2,
        k,
        BSIZE,
        L,
        stride_hin_bsize,
        stride_hin_BSIZE,
        stride_su_l,
        stride_su_BSIZE,
        stride_su_k,
        stride_out_l,
        stride_out_bsize,
        stride_out_k,
    )

    # Compute intermediate for S2s gradient
    # Computes U2s_hin = U2s @ hin
    ####################
    L, K, d2 = U2s.shape  # Number of terms, low rank, input dimension
    _, BSIZE = hin.shape  # Input dimension, batch size

    # Allocate memory for intermediate result
    U2s_hin = torch.empty((L, K, BSIZE), dtype=torch.float32, device=device)

    # Calculate tensor strides for efficient memory access
    stride_hin_d2, stride_hin_BSIZE = hin.stride(0), hin.stride(1)
    stride_su_l, stride_su_K, stride_su_d2 = U2s.stride(0), U2s.stride(1), U2s.stride(2)
    stride_out_l, stride_out_K, stride_out_BSIZE = (
        U2s_hin.stride(0),
        U2s_hin.stride(1),
        U2s_hin.stride(2),
    )

    # Define the computational grid for the Triton kernel
    grid = lambda META: (
        L,
        triton.cdiv(K, META["BLOCK_SIZE_K"])
        * triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"]),
    )

    # Launch the kernel to compute intermediate result
    wrap_triton(first_pass_U2s_hin_kernel)[grid](
        hin,
        U2s,
        U2s_hin,
        K,
        d2,
        BSIZE,
        L,
        stride_hin_d2,
        stride_hin_BSIZE,
        stride_su_l,
        stride_su_K,
        stride_su_d2,
        stride_out_l,
        stride_out_K,
        stride_out_BSIZE,
    )

    # Compute gradient with respect to S2s
    # Computes grad_S2s = U2s_hin @ g
    ###############
    L, K, BSIZE = U2s_hin.shape  # Number of terms, low rank, batch size
    _, d1 = g.shape  # Batch size, output dimension

    # Allocate memory for S2s gradient
    grad_S2s = torch.empty((L, K, d1), dtype=torch.float32, device=device)

    # Calculate tensor strides for efficient memory access
    stride_g_BSIZE, stride_g_d1 = g.stride(0), g.stride(1)
    stride_su_l, stride_su_K, stride_su_BSIZE = (
        U2s_hin.stride(0),
        U2s_hin.stride(1),
        U2s_hin.stride(2),
    )
    stride_out_l, stride_out_K, stride_out_d1 = (
        grad_S2s.stride(0),
        grad_S2s.stride(1),
        grad_S2s.stride(2),
    )

    # Define the computational grid for the Triton kernel
    grid = lambda META: (
        L,
        triton.cdiv(K, META["BLOCK_SIZE_K"]) * triton.cdiv(d1, META["BLOCK_SIZE_d1"]),
    )

    # Launch the kernel to compute S2s gradients
    wrap_triton(calc_grad_S2s_kernel)[grid](
        g,
        U2s_hin,
        grad_S2s,
        K,
        BSIZE,
        d1,
        L,
        stride_g_BSIZE,
        stride_g_d1,
        stride_su_l,
        stride_su_K,
        stride_su_BSIZE,
        stride_out_l,
        stride_out_K,
        stride_out_d1,
    )

    # Return gradients for input, S1s, S2s, and bias
    return [
        grad,  # Gradient with respect to input
        grad_S1s,  # Gradient with respect to S1s
        grad_S2s,  # Gradient with respect to S2s
        g.sum(0) / (2 * num_terms),  # Gradient with respect to bias
    ]


@backward_op.register_kernel("cpu")
def _(input, S1s, S2s, U1s, U2s, grad_output):
    """CPU implementation of the backward pass.

    This is used as a fallback when CUDA is not available.
    """
    num_terms = S2s.shape[0]
    # Scale gradient by number of terms
    g = grad_output / (2 * num_terms)
    # Expand g to match the number of terms
    g = g.unsqueeze(0).expand(num_terms, g.shape[0], g.shape[1])
    # Prepare input for batch matrix multiplications
    input = (
        input.unsqueeze(0)
        .expand(num_terms, input.shape[0], input.shape[1])
        .transpose(1, 2)
    )
    # Transpose matrices for matrix multiplication
    U1s = U1s.transpose(1, 2)
    S1s = S1s.transpose(1, 2)
    U2s = U2s.transpose(1, 2)
    S2s = S2s.transpose(1, 2)

    # Compute intermediate result
    t1 = g.bmm(U1s)
    # Compute gradient with respect to input
    grad = t1.bmm(S1s).sum(0) + g.bmm(S2s).bmm(U2s).sum(0)
    # Compute gradient with respect to S2s
    grad_S2s = (U2s.bmm(input)).bmm(g)
    # Compute gradient with respect to S1s
    grad_S1s = input.bmm(g.bmm(U1s))

    # Use first slice of g for bias gradient
    g = g[0]
    return [
        grad,  # Gradient with respect to input
        grad_S1s,  # Gradient with respect to S1s
        grad_S2s,  # Gradient with respect to S2s
        # Sum g on batch dimension for bias gradient
        g.reshape(input.shape[2], -1).sum(0),
    ]


class SketchedLinearFunction_triton(Function):
    """Autograd function for the sketched linear layer.

    This class defines the forward and backward operations for the sketched linear layer
    using PyTorch's autograd mechanism with Triton-accelerated kernels.
    """

    @staticmethod
    def forward(
        input: torch.Tensor,
        S1s: torch.Tensor,
        S2s: torch.Tensor,
        U1s: torch.Tensor,
        U2s: torch.Tensor,
        bias: torch.Tensor,
    ):
        """Forward pass for the sketched linear layer.

        Args:
            input: Input tensor of shape (batch_size, in_features)
            S1s: Learnable parameters of shape (num_terms, in_features, low_rank)
            S2s: Learnable parameters of shape (num_terms, low_rank, out_features)
            U1s: Fixed random matrices of shape (num_terms, low_rank, out_features)
            U2s: Fixed random matrices of shape (num_terms, in_features, low_rank)
            bias: Bias term of shape (out_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        return forward_op(input, S1s, S2s, U1s, U2s, bias)

    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any):
        """Save tensors needed for the backward pass.

        Args:
            ctx: Context object to save tensors for backward
            inputs: Tuple of all inputs passed to forward
            output: Output of the forward pass
        """
        input, S1s, S2s, U1s, U2s, bias = inputs
        ctx.save_for_backward(input, S1s, S2s, U1s, U2s, bias)

    @staticmethod
    def backward(ctx: Any, *grad_output: Any) -> Any:
        """Backward pass for the sketched linear layer.

        The gradient computations follow these formulas:
        - dl/dS2_i = U1_i g h_in^T / 2 * l
        - dl/dS1_i = g h_in^T U2_i^T / 2 * l
        - dl/dh_in = 1/(2*l) * (sum_{i=1}^{l} (S1_i^T U1_i g) + sum_{i=1}^{l} (U2_i^T S2_i g))
        - dl/db = g

        Args:
            ctx: Context object with saved tensors
            grad_output: Gradient from the next layer

        Returns:
            Tuple of gradients for each input to forward
        """
        hin, S1s, S2s, U1s, U2s, _ = ctx.saved_tensors
        grad_input, grad_S1s, grad_S2s, grad_bias = backward_op(
            hin, S1s, S2s, U1s, U2s, grad_output[0]
        )
        # Return None for U1s and U2s since they are fixed random matrices
        return grad_input, grad_S1s, grad_S2s, None, None, grad_bias


class SKLinear_triton(nn.Module):
    """Sketched Linear Layer with Triton acceleration.

    This layer approximates a dense linear layer y = Wx + b using a low-rank approximation:
    y = 1/2 * (x @ S1s @ U1s + x @ U2s @ S2s) + b

    where S1s and S2s are learnable parameters, and U1s and U2s are fixed random matrices.
    This approach reduces the number of parameters from O(in_features * out_features) to
    O(num_terms * low_rank * (in_features + out_features)).

    Attributes:
        in_features: Size of each input sample
        out_features: Size of each output sample
        num_terms: Number of terms in the approximation (l)
        low_rank: Rank of the approximation (k)
        S1s: Learnable parameters of shape (num_terms, in_features, low_rank)
        S2s: Learnable parameters of shape (num_terms, low_rank, out_features)
        U1s: Fixed random matrices of shape (num_terms, low_rank, out_features)
        U2s: Fixed random matrices of shape (num_terms, in_features, low_rank)
        bias: Bias term of shape (out_features)
    """

    __constants__ = ["in_features", "out_features", "num_terms", "low_rank"]
    in_features: int
    out_features: int
    num_terms: int
    low_rank: int
    S1s: torch.Tensor
    S2s: torch.Tensor
    U1s: torch.Tensor
    U2s: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_terms: int,
        low_rank: int,
        W_init=None,
        bias: bool = True,
        dtype=None,
        device=None,
    ):
        """Initialize the sketched linear layer.

        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            num_terms: Number of terms in the approximation (l)
            low_rank: Rank of the approximation (k)
            W_init: Initial weight matrix to initialize S1s and S2s (optional)
            bias: Whether to include a bias term (default: True)
            dtype: Data type of the parameters (default: None)
            device: Device to place the parameters on (default: None)
        """
        factory_kwargs = {"dtype": dtype, "device": device}
        super(SKLinear_triton, self).__init__()

        # Uncomment to enable parameter count check
        # if (
        #     2 * num_terms * low_rank * (out_features + in_features)
        #     > out_features * in_features
        # ):
        #     raise ValueError(
        #         "The number of parameters in the sketching layer is larger "
        #         + "than the number of parameters in the fully connected layer."
        #     )

        self.num_terms = num_terms  # l: Number of terms in the approximation
        self.low_rank = low_rank  # k: Rank of the approximation
        self.out_features = out_features  # d1: Output dimension
        self.in_features = in_features  # d2: Input dimension

        # Register U1s and U2s as buffers since they are not learnable
        # U1s: Random projection matrices for output dimension
        self.register_buffer(
            "U1s",
            torch.stack(
                [
                    gen_U(low_rank, out_features, **factory_kwargs)
                    for _ in range(num_terms)
                ]
            ),
        )  # k(low rank)xd1(out) stacked along the zeros dimension (l) -> l x k x d1

        # U2s: Random projection matrices for input dimension
        self.register_buffer(
            "U2s",
            torch.stack(
                [
                    gen_U(in_features, low_rank, **factory_kwargs)
                    for _ in range(num_terms)
                ]
            ),
        )  # d2xk stacked along the zeros dimension (l) -> l x d2 x k

        # Initialize weight matrix W to compute S1s and S2s
        if W_init is None:
            # If no initial weights provided, use Kaiming uniform initialization
            W = torch.empty(in_features, out_features, **factory_kwargs)  # d2 * d1
            init.kaiming_uniform_(W, a=math.sqrt(5))
        else:
            # Use provided initial weights (transposed)
            W = W_init.T.detach().clone()

        # S1s: Learnable parameters for the first path
        self.S1s = nn.Parameter(
            torch.stack([torch.matmul(W, self.U1s[i].T) for i in range(num_terms)])
        )  # d2xk stacked along the zeros dimension (l) -> l x d2 x k

        # S2s: Learnable parameters for the second path
        self.S2s = nn.Parameter(
            torch.stack([torch.matmul(self.U2s[i].T, W) for i in range(num_terms)])
        )  # kxd1 stacked along the zeros dimension (l) -> l x k x d1

        # Initialize bias term if needed
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))  # d1
            # Initialize bias with uniform distribution
            fan_in, _ = init._calculate_fan_in_and_fan_out(W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, h_in):
        """Forward pass of the sketched linear layer.

        Args:
            h_in: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # TODO: Make sure all the tensors are contiguous
        return SketchedLinearFunction_triton.apply(
            h_in, self.S1s, self.S2s, self.U1s, self.U2s, self.bias
        )
