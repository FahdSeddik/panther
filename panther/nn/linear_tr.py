import math
from typing import Any, Tuple, List

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import init
import triton
from panther.random import scaled_sign_sketch as gen_U
from torch.library import triton_op, wrap_triton
from .linear_kernels import (
    first_pass_kernel,
    second_pass_kernel,
    first_pass_gU1s_g_S2s_kernel,
    second_pass_gUS11_22_kernel,
    calc_grad_S1s_kernel,
    first_pass_U2s_hin_kernel,
    calc_grad_S2s_kernel
)

@triton_op("mylib::forward_op", mutates_args={})
def forward_op(hin: torch.Tensor, S1s: torch.Tensor, S2s: torch.Tensor, U1s: torch.Tensor, U2s: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    device = 'cuda'
    
    # first pass
    ############
    L = S2s.shape[0]
    BSIZE, d2 = hin.shape
    L, _, K = S1s.shape
    
    in1 = torch.empty((L, BSIZE, K), dtype=torch.float16, device=device)
    in2 = torch.empty((L, BSIZE, K), dtype=torch.float16, device=device)
    
    stride_hin_bsize, stride_hin_d2 = hin.shape[1] , 1
    stride_su_l, stride_su_d2, stride_su_k = S1s.shape[1] * S1s.shape[2], S1s.shape[2], 1
    stride_out_l, stride_out_bsize, stride_out_k = in1.shape[1] * in1.shape[2], in1.shape[2], 1
    
    grid = lambda META: (L, triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]), )
    
    wrap_triton(first_pass_kernel)[grid](
        hin, S1s, U2s, in1, in2,
        BSIZE, K, d2, L,
        stride_hin_bsize, stride_hin_d2,
        stride_su_l, stride_su_d2, stride_su_k,
        stride_out_l, stride_out_bsize, stride_out_k,
    )
    
    # second pass
    ############# 
    L, BSIZE, K = in1.shape
    _, _, d1 = U1s.shape
    
    out = torch.empty((BSIZE, d1), dtype=torch.float16, device=device)

    stride_in12_l, stride_in12_bsize, stride_in12_k = in1.shape[1] * in1.shape[2], in1.shape[2], 1
    stride_us_l, stride_us_k, stride_us_d1 = U1s.shape[1] * U1s.shape[2], U1s.shape[2], 1
    stride_bias_bsize, stride_bias_d1 = bias.shape[1], 1
    stride_out_bsize, stride_out_d1 = out.shape[1], 1
    
    grid = lambda META: (triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"]) * triton.cdiv(d1, META["BLOCK_SIZE_D1"]), )
    
    wrap_triton(second_pass_kernel)[grid](
        in1, in2, U1s, S2s, bias, out,
        BSIZE, d1, K, L,
        stride_in12_l, stride_in12_bsize, stride_in12_k,
        stride_us_l, stride_us_k, stride_us_d1,
        stride_bias_bsize, stride_bias_d1,
        stride_out_bsize, stride_out_d1,
    )
    
    return out

@forward_op.register_kernel("cpu")
def _(input, S1s, S2s, U1s, U2s, bias):
    num_terms = S2s.shape[0]
    # Efficiently perform the sum over all l terms
    input = input.unsqueeze(0).expand(num_terms, input.shape[0], input.shape[1])
    return (
        ((input.bmm(S1s)).bmm(U1s)).mean(0) / 2
        + ((input.bmm(U2s)).bmm(S2s)).mean(0) / 2
        + bias
    )
    
@triton_op("mylib::backward_op", mutates_args={})
def backward_op(hin: torch.Tensor, S1s: torch.Tensor, S2s: torch.Tensor, U1s: torch.Tensor, U2s: torch.Tensor, g: torch.Tensor) -> List[torch.Tensor, 
                                                                                                                                         torch.Tensor, 
                                                                                                                                         torch.Tensor, 
                                                                                                                                         None, 
                                                                                                                                         None, 
                                                                                                                                         torch.Tensor]:
    device = 'cuda'
    num_terms = S2s.shape[0]
        
    hin = hin.transpose(0, 1)
    U1s = U1s.transpose(1, 2)
    S1s = S1s.transpose(1, 2)
    U2s = U2s.transpose(1, 2)
    S2s = S2s.transpose(1, 2)
    
    # first_pass_gU1s_g_S2s
    #######################
    BSIZE, d1 = g.shape
    L, _, K = U1s.shape
    
    g_U1s = torch.empty((L, BSIZE, K), dtype=torch.float16, device='cuda')
    g_S2s = torch.empty((L, BSIZE, K), dtype=torch.float16, device='cuda')

    stride_g_bsize, stride_g_d1 = g.shape[1], 1
    stride_su_l, stride_su_d1, stride_su_k = U1s.shape[1] * U1s.shape[2], U1s.shape[2], 1
    stride_out_l, stride_out_bsize, stride_out_k = g_U1s.shape[1] * g_U1s.shape[2], g_U1s.shape[2], 1
    
    grid = lambda META: (L, triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]), )
    
    wrap_triton(first_pass_gU1s_g_S2s_kernel)[grid](
        g, U1s, S2s, g_U1s, g_S2s,
        BSIZE, K, d1, L,
        stride_g_bsize, stride_g_d1,
        stride_su_l, stride_su_d1, stride_su_k,
        stride_out_l, stride_out_bsize, stride_out_k
    )

    # second_pass_gUS11_22
    #######################
    L, BSIZE, K = g_U1s.shape
    _, _, d2 = S1s.shape
    
    grad = torch.empty((BSIZE, d2), dtype=torch.float16, device='cuda')

    stride_g_U1s2_l, stride_g_U1s2_bsize, stride_g_U1s2_k = g_U1s.shape[1] * g_U1s.shape[2], g_U1s.shape[2], 1
    stride_us_l, stride_us_k, stride_us_d2 = S1s.shape[1] * S1s.shape[2], S1s.shape[2], 1
    stride_out_bsize, stride_out_d2 = grad.shape[1], 1
    
    grid = lambda META: (triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"]) * triton.cdiv(d2, META["BLOCK_SIZE_d2"]), )
    
    wrap_triton(second_pass_gUS11_22_kernel)[grid](
        g_U1s, g_S2s, S1s, U2s, grad,
        BSIZE, d2, K, L,
        stride_g_U1s2_l, stride_g_U1s2_bsize, stride_g_U1s2_k,
        stride_us_l, stride_us_k, stride_us_d2,
        stride_out_bsize, stride_out_d2,
    )
    
    # calc_grad_S1s
    ################
    d2, BSIZE = hin.shape
    L, _, k = g_U1s.shape
    
    grad_S1s = torch.empty((L, d2, k), dtype=torch.float16, device=device)

    stride_hin_bsize, stride_hin_BSIZE = hin.shape[1], 1
    stride_su_l, stride_su_BSIZE, stride_su_k = g_U1s.shape[1] * g_U1s.shape[2], g_U1s.shape[2], 1
    stride_out_l, stride_out_bsize, stride_out_k = grad_S1s.shape[1] * grad_S1s.shape[2], grad_S1s.shape[2], 1
    
    grid = lambda META: (L, triton.cdiv(d2, META["BLOCK_SIZE_d2"]) * triton.cdiv(k, META["BLOCK_SIZE_k"]), )
    
    wrap_triton(calc_grad_S1s_kernel)[grid](
        hin, g_U1s, grad_S1s,
        d2, k, BSIZE, L,
        stride_hin_bsize, stride_hin_BSIZE,
        stride_su_l, stride_su_BSIZE, stride_su_k,
        stride_out_l, stride_out_bsize, stride_out_k
    )
    
    # first_pass_U2s_hin
    ####################
    L, K, d2 = U2s.shape
    _, BSIZE = hin.shape
    
    U2s_hin = torch.empty((L, K, BSIZE), dtype=torch.float16, device=device)

    stride_hin_d2, stride_hin_BSIZE = hin.shape[1], 1
    stride_su_l, stride_su_K, stride_su_d2 = U2s.shape[1] * U2s.shape[2], U2s.shape[2], 1
    stride_out_l, stride_out_K, stride_out_BSIZE = U2s_hin.shape[1] * U2s_hin.shape[2], U2s_hin.shape[2], 1
    
    grid = lambda META: (L, triton.cdiv(K, META["BLOCK_SIZE_K"]) * triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"]), )
    
    wrap_triton(first_pass_U2s_hin_kernel)[grid](
        hin, U2s, U2s_hin,
        K, d2, BSIZE, L,
        stride_hin_d2, stride_hin_BSIZE,
        stride_su_l, stride_su_K, stride_su_d2,
        stride_out_l, stride_out_K, stride_out_BSIZE
    )
    
    # calc_grad_S2s
    ###############
    L, K, BSIZE = U2s_hin.shape
    _, d1 = g.shape
    
    grad_S2s = torch.empty((L, K, d1), dtype=torch.float16, device=device)

    stride_g_BSIZE, stride_g_d1 = g.shape[1], 1
    stride_su_l, stride_su_K, stride_su_BSIZE = U2s_hin.shape[1] * U2s_hin.shape[2], U2s_hin.shape[2], 1
    stride_out_l, stride_out_K, stride_out_d1 = grad_S2s.shape[1] * grad_S2s.shape[2], grad_S2s.shape[2], 1
    
    grid = lambda META: (L, triton.cdiv(K, META["BLOCK_SIZE_K"]) * triton.cdiv(d1, META["BLOCK_SIZE_d1"]), )
    
    wrap_triton(calc_grad_S2s_kernel)[grid](
        g, U2s_hin, grad_S2s,
        K, BSIZE, d1, L,
        stride_g_BSIZE, stride_g_d1,
        stride_su_l, stride_su_K, stride_su_BSIZE,
        stride_out_l, stride_out_K, stride_out_d1
    )

    return (
        grad,
        grad_S1s,
        grad_S2s,
        None,
        None,
        g.sum(0) / (2 * num_terms),
    )
    
@backward_op.register_kernel("cpu")
def _(input, S1s, S2s, U1s, U2s, grad_output):
    num_terms = S2s.shape[0]
    g = grad_output / (2 * num_terms)
    g = g.unsqueeze(0).expand(num_terms, g.shape[0], g.shape[1])
    input = (
        input.unsqueeze(0)
        .expand(num_terms, input.shape[0], input.shape[1])
        .transpose(1, 2)
    )
    U1s = U1s.transpose(1, 2)
    S1s = S1s.transpose(1, 2)
    U2s = U2s.transpose(1, 2)
    S2s = S2s.transpose(1, 2)
    t1 = g.bmm(U1s)
    grad = t1.bmm(S1s).sum(0) + g.bmm(S2s).bmm(U2s).sum(0)
    grad_S2s = (U2s.bmm(input)).bmm(g)
    grad_S1s = input.bmm(g.bmm(U1s))

    g = g[0]
    return (
        grad,
        grad_S1s,
        grad_S2s,
        None,
        None,
        # sum g on batch dimension input.shape[0]
        g.reshape(input.shape[2], -1).sum(0),
    )
    
class SketchedLinearFunction_triton(Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(
        input: torch.Tensor,
        S1s: torch.Tensor,
        S2s: torch.Tensor,
        U1s: torch.Tensor,
        U2s: torch.Tensor,
        bias: torch.Tensor,
    ):
        return torch.mylib.ops.forward_op(input, S1s, S2s, U1s, U2s, bias)

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any):
        input, S1s, S2s, U1s, U2s, bias = inputs
        ctx.save_for_backward(input, S1s, S2s, U1s, U2s, bias)

    @staticmethod
    def backward(ctx: Any, *grad_output: Any) -> Any:
        # dl/dS2_i = U1_i g h_in^T / 2 * l
        # dl/dS1_i = g h_in^T U2_i^T / 2 * l
        # dl/dh_in = 1/(2*l) * (sum_{i=1}^{l} (S1_i^T U1_i g) + sum_{i=1}^{l} (U2_i^T S2_i g))
        # dl/db = g
        hin, S1s, S2s, U1s, U2s, _ = ctx.saved_tensors
        return torch.mylib.ops.backward_op(hin, S1s, S2s, U1s, U2s, grad_output[0])


class SKLinear_triton(nn.Module):
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
        factory_kwargs = {"dtype": dtype, "device": device}
        super(SKLinear_triton, self).__init__()

        # if (
        #     2 * num_terms * low_rank * (out_features + in_features)
        #     > out_features * in_features
        # ):
        #     raise ValueError(
        #         "The number of parameters in the sketching layer is larger "
        #         + "than the number of parameters in the fully connected layer."
        #     )

        self.num_terms = num_terms # l
        self.low_rank = low_rank # k
        self.out_features = out_features
        self.in_features = in_features

        # Register U1s and U2s as buffers since they are not learnable
        self.register_buffer(
            "U1s",
            torch.stack(
                [
                    gen_U(low_rank, out_features, **factory_kwargs)
                    for _ in range(num_terms)
                ]
            ),
        )  # k(low rank)xd1(out) stacked along the zeros dimension (l) -> l x k x d1
        self.register_buffer(
            "U2s",
            torch.stack(
                [
                    gen_U(in_features, low_rank, **factory_kwargs)
                    for _ in range(num_terms)
                ]
            ),
        )  # d2xk stacked along the zeros dimension (l) -> l x d2 x k

        # W is used to only initialize S
        if W_init is None:
            W = torch.empty(in_features, out_features, **factory_kwargs) # d2 * d1
            init.kaiming_uniform_(W, a=math.sqrt(5))
        else:
            W = W_init.T.detach().clone()

        # S1s and S2s are precomputed but not updated in the backward pass
        self.S1s = nn.Parameter(
            torch.stack([torch.matmul(W, self.U1s[i].T) for i in range(num_terms)])
        )  # d2xk stacked along the zeros dimension (l) -> l x d2 x k
        self.S2s = nn.Parameter(
            torch.stack([torch.matmul(self.U2s[i].T, W) for i in range(num_terms)])
        )  # kxd1 stacked along the zeros dimension (l) -> l x k x d1

        # Bias term initialized with a small standard deviation
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs)) #1 * d1
            fan_in, _ = init._calculate_fan_in_and_fan_out(W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, h_in):
        # TODO: Make sure all the things are contiguos
        return SketchedLinearFunction_triton.apply(
            h_in, self.S1s, self.S2s, self.U1s, self.U2s, self.bias
        )