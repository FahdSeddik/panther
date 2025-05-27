"""
Backward computation kernels for linear operations.

This module contains Triton kernel implementations for the backward pass of linear
operations, including gradient computation for various components in the linear model.
The kernels are optimized for GPU acceleration using Triton.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_K": 256,
                "BLOCK_SIZE_d1": 64,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 64,
                "BLOCK_SIZE_K": 256,
                "BLOCK_SIZE_d1": 32,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_K": 128,
                "BLOCK_SIZE_d1": 32,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_K": 64,
                "BLOCK_SIZE_d1": 32,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 64,
                "BLOCK_SIZE_K": 128,
                "BLOCK_SIZE_d1": 32,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_K": 32,
                "BLOCK_SIZE_d1": 32,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 64,
                "BLOCK_SIZE_K": 32,
                "BLOCK_SIZE_d1": 32,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 32,
                "BLOCK_SIZE_K": 64,
                "BLOCK_SIZE_d1": 32,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_K": 256,
                "BLOCK_SIZE_d1": 128,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 256,
                "BLOCK_SIZE_K": 128,
                "BLOCK_SIZE_d1": 128,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 256,
                "BLOCK_SIZE_K": 64,
                "BLOCK_SIZE_d1": 128,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 64,
                "BLOCK_SIZE_K": 256,
                "BLOCK_SIZE_d1": 128,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_K": 128,
                "BLOCK_SIZE_d1": 128,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_K": 64,
                "BLOCK_SIZE_d1": 64,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 64,
                "BLOCK_SIZE_K": 128,
                "BLOCK_SIZE_d1": 64,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_d1": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
    ],
    key=["BSIZE", "K", "d1", "L"],
)
@triton.jit
def first_pass_gU1s_g_S2s_kernel(
    g_ptr,
    U1s_ptr,
    S2s_ptr,
    g_U1s_ptr,
    g_S2s_ptr,
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
    BLOCK_SIZE_BSIZE: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_d1: tl.constexpr,
    GROUP_SIZE_BSIZE: tl.constexpr,
):
    """
    Compute the first pass of gradient calculations for U1s and S2s.

    This kernel computes the dot products of gradient 'g' with U1s and S2s matrices
    to produce intermediate gradients g_U1s and g_S2s.

    Args:
        g_ptr: Pointer to gradient tensor
        U1s_ptr: Pointer to U1s tensor
        S2s_ptr: Pointer to S2s tensor
        g_U1s_ptr: Pointer to output g_U1s gradient tensor
        g_S2s_ptr: Pointer to output g_S2s gradient tensor
        BSIZE: Batch size dimension
        K: Feature dimension
        d1: Input dimension
        L: Number of layers
        stride_*: Stride values for tensor memory layout
        BLOCK_SIZE_*: Block sizes for tiling (constexpr)
        GROUP_SIZE_BSIZE: Group size for batch dimension (constexpr)
    """
    pid = tl.program_id(axis=1)
    batch_id = tl.program_id(axis=0)

    num_pid_bsize = tl.cdiv(BSIZE, BLOCK_SIZE_BSIZE)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_BSIZE * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_bsize = group_id * GROUP_SIZE_BSIZE
    group_size_bsize = min(num_pid_bsize - first_pid_bsize, GROUP_SIZE_BSIZE)
    pid_bsize = first_pid_bsize + ((pid % num_pid_in_group) % group_size_bsize)
    pid_k = (pid % num_pid_in_group) // group_size_bsize

    offs_bsize = pid_bsize * BLOCK_SIZE_BSIZE + tl.arange(0, BLOCK_SIZE_BSIZE)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_d1 = tl.arange(0, BLOCK_SIZE_d1)

    g_ptrs = g_ptr + (
        offs_bsize[:, None] * stride_g_bsize + offs_d1[None, :] * stride_g_d1
    )

    su_tmp = batch_id * stride_su_l + (
        offs_d1[:, None] * stride_su_d1 + offs_k[None, :] * stride_su_k
    )
    U1s_ptrs = U1s_ptr + su_tmp
    S2s_ptrs = S2s_ptr + su_tmp

    accumulator1 = tl.full(
        shape=(BLOCK_SIZE_BSIZE, BLOCK_SIZE_K), value=0.0, dtype=tl.float32
    )
    accumulator2 = tl.full(
        shape=(BLOCK_SIZE_BSIZE, BLOCK_SIZE_K), value=0.0, dtype=tl.float32
    )

    for d1_i in range(0, tl.cdiv(d1, BLOCK_SIZE_d1)):
        g = tl.load(
            g_ptrs, mask=(offs_d1[None, :] < d1 - d1_i * BLOCK_SIZE_d1), other=0.0
        )

        su_mask = offs_d1[:, None] < d1 - d1_i * BLOCK_SIZE_d1
        U1s = tl.load(U1s_ptrs, mask=su_mask, other=0.0)
        S2s = tl.load(S2s_ptrs, mask=su_mask, other=0.0)

        accumulator1 += tl.dot(g, U1s, input_precision="ieee")
        accumulator2 += tl.dot(g, S2s, input_precision="ieee")

        g_ptrs += BLOCK_SIZE_d1 * stride_g_d1
        U1s_ptrs += BLOCK_SIZE_d1 * stride_su_d1
        S2s_ptrs += BLOCK_SIZE_d1 * stride_su_d1

    out_tmp = (
        batch_id * stride_out_l
        + stride_out_bsize * offs_bsize[:, None]
        + stride_out_k * offs_k[None, :]
    )
    g_U1s_ptrs = g_U1s_ptr + out_tmp
    g_S2s_ptrs = g_S2s_ptr + out_tmp

    out_mask = (offs_bsize[:, None] < BSIZE) & (offs_k[None, :] < K)

    tl.store(g_U1s_ptrs, accumulator1, mask=out_mask)
    tl.store(g_S2s_ptrs, accumulator2, mask=out_mask)


def first_pass_gU1s_g_S2s(g, U1s, S2s):
    """
    Wrapper function for the first pass gradient computation kernel.

    Computes the dot products of gradient 'g' with U1s and S2s to produce
    intermediate gradients needed for backpropagation.

    Args:
        g: Gradient tensor of shape (BSIZE, d1)
        U1s: First set of weights of shape (L, d1, K)
        S2s: Second set of weights of shape (L, d1, K)

    Returns:
        tuple: (g_U1s, g_S2s) intermediate gradient tensors of shape (L, BSIZE, K)
    """
    # assert g.shape[1] == U1s.shape[1], "Incompatible dimensions"
    # assert g.shape[1] == S2s.shape[1], "Incompatible dimensions"
    # assert g.is_contiguous(), "Matrix A must be contiguous"
    # assert U1s.is_contiguous(), "Matrix A must be contiguous"
    # assert S2s.is_contiguous(), "Matrix A must be contiguous"
    # assert U1s.stride() == S2s.stride(), "Matrix A must be contiguous"

    BSIZE, d1 = g.shape
    L, _, K = U1s.shape

    g_U1s = torch.empty((L, BSIZE, K), dtype=torch.float32, device="cuda")
    g_S2s = torch.empty((L, BSIZE, K), dtype=torch.float32, device="cuda")

    # stride_g_bsize, stride_g_d1 = g.stride()
    # stride_su_l, stride_su_d1, stride_su_k = U1s.stride()
    # stride_out_l, stride_out_bsize, stride_out_k = g_U1s.stride()
    stride_g_bsize, stride_g_d1 = g.shape[1], 1
    stride_su_l, stride_su_d1, stride_su_k = (
        U1s.shape[1] * U1s.shape[2],
        U1s.shape[2],
        1,
    )
    stride_out_l, stride_out_bsize, stride_out_k = (
        g_U1s.shape[1] * g_U1s.shape[2],
        g_U1s.shape[2],
        1,
    )

    # assert g_U1s.stride() == g_S2s.stride(), "Matrix A must be contiguous"

    grid = lambda META: (
        L,
        triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"])
        * triton.cdiv(K, META["BLOCK_SIZE_K"]),
    )

    first_pass_gU1s_g_S2s_kernel[grid](
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

    return g_U1s, g_S2s


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_d2": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 64,
                "BLOCK_SIZE_d2": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_d2": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_d2": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 64,
                "BLOCK_SIZE_d2": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_d2": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 64,
                "BLOCK_SIZE_d2": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 32,
                "BLOCK_SIZE_d2": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_d2": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 256,
                "BLOCK_SIZE_d2": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 256,
                "BLOCK_SIZE_d2": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 64,
                "BLOCK_SIZE_d2": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_d2": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_d2": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 64,
                "BLOCK_SIZE_d2": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_d2": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_BSIZE": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
    ],
    key=["BSIZE", "d2", "K", "L"],
)
@triton.jit
def second_pass_gUS11_22_kernel(
    g_U1s_ptr,
    g_S2s_ptr,
    S1s_ptr,
    U2s_ptr,
    out_ptr,
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
    BLOCK_SIZE_BSIZE: tl.constexpr,
    BLOCK_SIZE_d2: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_BSIZE: tl.constexpr,
):
    """
    Compute the second pass of gradient calculations for the linear model.

    This kernel combines the intermediate gradients from the first pass with
    S1s and U2s weights to compute the final gradient.

    Args:
        g_U1s_ptr: Pointer to g_U1s intermediate gradient tensor
        g_S2s_ptr: Pointer to g_S2s intermediate gradient tensor
        S1s_ptr: Pointer to S1s weights tensor
        U2s_ptr: Pointer to U2s weights tensor
        out_ptr: Pointer to output gradient tensor
        BSIZE: Batch size dimension
        d2: Output dimension
        K: Feature dimension
        L: Number of layers
        stride_*: Stride values for tensor memory layout
        BLOCK_SIZE_*: Block sizes for tiling (constexpr)
        GROUP_SIZE_BSIZE: Group size for batch dimension (constexpr)
    """
    pid = tl.program_id(axis=0)

    num_pid_bsize = tl.cdiv(BSIZE, BLOCK_SIZE_BSIZE)
    num_pid_d2 = tl.cdiv(d2, BLOCK_SIZE_d2)
    num_pid_in_group = GROUP_SIZE_BSIZE * num_pid_d2
    group_id = pid // num_pid_in_group
    first_pid_bsize = group_id * GROUP_SIZE_BSIZE
    GROUP_SIZE_BSIZE = min(num_pid_bsize - first_pid_bsize, GROUP_SIZE_BSIZE)
    pid_bsize = first_pid_bsize + ((pid % num_pid_in_group) % GROUP_SIZE_BSIZE)
    pid_d2 = (pid % num_pid_in_group) // GROUP_SIZE_BSIZE

    offs_bsize = pid_bsize * BLOCK_SIZE_BSIZE + tl.arange(0, BLOCK_SIZE_BSIZE)
    offs_d2 = pid_d2 * BLOCK_SIZE_d2 + tl.arange(0, BLOCK_SIZE_d2)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    in_tmp = (
        offs_bsize[:, None] * stride_g_U1s2_bsize + offs_k[None, :] * stride_g_U1s2_k
    )
    us_tmp = offs_k[:, None] * stride_us_k + offs_d2[None, :] * stride_us_d2

    accumulator = tl.full(
        shape=(BLOCK_SIZE_BSIZE, BLOCK_SIZE_d2), value=0.0, dtype=tl.float32
    )

    for l in range(0, L):
        l_tmp_stride = l * stride_g_U1s2_l

        g_U1s_ptrs = l_tmp_stride + g_U1s_ptr + in_tmp
        g_S2s_ptrs = l_tmp_stride + g_S2s_ptr + in_tmp

        S1s_ptrs = l_tmp_stride + S1s_ptr + us_tmp
        U2s_ptrs = l_tmp_stride + U2s_ptr + us_tmp

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            in_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
            g_U1s = tl.load(g_U1s_ptrs, mask=in_mask, other=0.0)
            g_S2s = tl.load(g_S2s_ptrs, mask=in_mask, other=0.0)

            us_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            S1s = tl.load(S1s_ptrs, mask=us_mask, other=0.0)
            U2s = tl.load(U2s_ptrs, mask=us_mask, other=0.0)

            accumulator += tl.dot(g_U1s, S1s, input_precision="ieee")
            accumulator += tl.dot(g_S2s, U2s, input_precision="ieee")

            in_inc = BLOCK_SIZE_K * stride_g_U1s2_k
            g_U1s_ptrs += in_inc
            g_S2s_ptrs += in_inc

            us_inc = BLOCK_SIZE_K * stride_us_k
            S1s_ptrs += us_inc
            U2s_ptrs += us_inc

    accumulator *= 1.0 / (2.0 * L)

    out_ptrs = (
        out_ptr
        + stride_out_bsize * offs_bsize[:, None]
        + stride_out_d2 * offs_d2[None, :]
    )
    out_mask = (offs_bsize[:, None] < BSIZE) & (offs_d2[None, :] < d2)

    tl.store(out_ptrs, accumulator, mask=out_mask)


def second_pass_gUS11_22(g_U1s, g_S2s, S1s, U2s):
    """
    Wrapper function for the second pass gradient computation kernel.

    Combines the intermediate gradients with weights to compute the final gradient.
    The function implements part of the backward pass in the linear model.

    Args:
        g_U1s: First intermediate gradient tensor of shape (L, BSIZE, K)
        g_S2s: Second intermediate gradient tensor of shape (L, BSIZE, K)
        S1s: First set of weights of shape (L, K, d2)
        U2s: Second set of weights of shape (L, K, d2)

    Returns:
        torch.Tensor: Final gradient tensor of shape (BSIZE, d2)
    """
    L, BSIZE, K = g_U1s.shape
    _, _, d2 = S1s.shape

    out = torch.empty((BSIZE, d2), dtype=torch.float32, device="cuda")

    stride_g_U1s2_l, stride_g_U1s2_bsize, stride_g_U1s2_k = (
        g_U1s.shape[1] * g_U1s.shape[2],
        g_U1s.shape[2],
        1,
    )
    stride_us_l, stride_us_k, stride_us_d2 = (
        S1s.shape[1] * S1s.shape[2],
        S1s.shape[2],
        1,
    )
    stride_out_bsize, stride_out_d2 = out.shape[1], 1

    grid = lambda META: (
        triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"])
        * triton.cdiv(d2, META["BLOCK_SIZE_d2"]),
    )

    second_pass_gUS11_22_kernel[grid](
        g_U1s,
        g_S2s,
        S1s,
        U2s,
        out,
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

    return out  # grad


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_d2": 128,
                "BLOCK_SIZE_k": 256,
                "BLOCK_SIZE_BSIZE": 64,
                "GROUP_SIZE_d2": 8,
            },
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_d2": 64,
                "BLOCK_SIZE_k": 256,
                "BLOCK_SIZE_BSIZE": 32,
                "GROUP_SIZE_d2": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_d2": 128,
                "BLOCK_SIZE_k": 128,
                "BLOCK_SIZE_BSIZE": 32,
                "GROUP_SIZE_d2": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_d2": 128,
                "BLOCK_SIZE_k": 64,
                "BLOCK_SIZE_BSIZE": 32,
                "GROUP_SIZE_d2": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_d2": 64,
                "BLOCK_SIZE_k": 128,
                "BLOCK_SIZE_BSIZE": 32,
                "GROUP_SIZE_d2": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_d2": 128,
                "BLOCK_SIZE_k": 32,
                "BLOCK_SIZE_BSIZE": 32,
                "GROUP_SIZE_d2": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_d2": 64,
                "BLOCK_SIZE_k": 32,
                "BLOCK_SIZE_BSIZE": 32,
                "GROUP_SIZE_d2": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_d2": 32,
                "BLOCK_SIZE_k": 64,
                "BLOCK_SIZE_BSIZE": 32,
                "GROUP_SIZE_d2": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_d2": 128,
                "BLOCK_SIZE_k": 256,
                "BLOCK_SIZE_BSIZE": 128,
                "GROUP_SIZE_d2": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_d2": 256,
                "BLOCK_SIZE_k": 128,
                "BLOCK_SIZE_BSIZE": 128,
                "GROUP_SIZE_d2": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_d2": 256,
                "BLOCK_SIZE_k": 64,
                "BLOCK_SIZE_BSIZE": 128,
                "GROUP_SIZE_d2": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_d2": 64,
                "BLOCK_SIZE_k": 256,
                "BLOCK_SIZE_BSIZE": 128,
                "GROUP_SIZE_d2": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_d2": 128,
                "BLOCK_SIZE_k": 128,
                "BLOCK_SIZE_BSIZE": 128,
                "GROUP_SIZE_d2": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_d2": 128,
                "BLOCK_SIZE_k": 64,
                "BLOCK_SIZE_BSIZE": 64,
                "GROUP_SIZE_d2": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_d2": 64,
                "BLOCK_SIZE_k": 128,
                "BLOCK_SIZE_BSIZE": 64,
                "GROUP_SIZE_d2": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_d2": 128,
                "BLOCK_SIZE_k": 32,
                "BLOCK_SIZE_BSIZE": 64,
                "GROUP_SIZE_d2": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
    ],
    key=["d2", "k", "BSIZE", "L"],
)
@triton.jit
def calc_grad_S1s_kernel(
    hin_ptr,
    g_U1s_ptr,
    grad_g_S1s_ptr,
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
    BLOCK_SIZE_d2: tl.constexpr,
    BLOCK_SIZE_k: tl.constexpr,
    BLOCK_SIZE_BSIZE: tl.constexpr,
    GROUP_SIZE_d2: tl.constexpr,
):
    """
    Compute gradients for S1s weights.

    This kernel calculates the gradient for S1s by computing the dot product
    of hidden activations (hin) with the intermediate gradient g_U1s.

    Args:
        hin_ptr: Pointer to hidden activation tensor
        g_U1s_ptr: Pointer to intermediate gradient tensor
        grad_g_S1s_ptr: Pointer to output gradient tensor for S1s
        d2: Output dimension
        k: Feature dimension
        BSIZE: Batch size dimension
        L: Number of layers
        stride_*: Stride values for tensor memory layout
        BLOCK_SIZE_*: Block sizes for tiling (constexpr)
        GROUP_SIZE_d2: Group size for output dimension (constexpr)
    """
    # Get program IDs for batch and parallel dimensions
    pid = tl.program_id(axis=1)
    batch_id = tl.program_id(axis=0)

    # Calculate the number of blocks and handle tiling
    num_pid_bsize = tl.cdiv(d2, BLOCK_SIZE_d2)
    num_pid_k = tl.cdiv(k, BLOCK_SIZE_k)
    num_pid_in_group = GROUP_SIZE_d2 * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_bsize = group_id * GROUP_SIZE_d2
    group_size_bsize = min(num_pid_bsize - first_pid_bsize, GROUP_SIZE_d2)
    pid_bsize = first_pid_bsize + ((pid % num_pid_in_group) % group_size_bsize)
    pid_k = (pid % num_pid_in_group) // group_size_bsize

    # Calculate offsets for memory access
    offs_bsize = pid_bsize * BLOCK_SIZE_d2 + tl.arange(0, BLOCK_SIZE_d2)
    offs_k = pid_k * BLOCK_SIZE_k + tl.arange(0, BLOCK_SIZE_k)
    offs_BSIZE = tl.arange(0, BLOCK_SIZE_BSIZE)

    # Ensure offsets are properly aligned
    offs_bsize = tl.max_contiguous(
        tl.multiple_of(offs_bsize, BLOCK_SIZE_d2), BLOCK_SIZE_d2
    )
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_SIZE_k), BLOCK_SIZE_k)
    offs_BSIZE = tl.max_contiguous(
        tl.multiple_of(offs_BSIZE, BLOCK_SIZE_BSIZE), BLOCK_SIZE_BSIZE
    )

    # Set up pointers for accessing input tensors
    hin_ptrs = hin_ptr + (
        offs_bsize[:, None] * stride_hin_bsize + offs_BSIZE[None, :] * stride_hin_BSIZE
    )
    su_tmp = batch_id * stride_su_l + (
        offs_BSIZE[:, None] * stride_su_BSIZE + offs_k[None, :] * stride_su_k
    )
    g_U1s_ptrs = g_U1s_ptr + su_tmp

    # Initialize accumulator for results
    accumulator1 = tl.full(
        shape=(BLOCK_SIZE_d2, BLOCK_SIZE_k), value=0.0, dtype=tl.float32
    )

    # Process blocks in the batch dimension
    for BSIZE_i in range(0, tl.cdiv(BSIZE, BLOCK_SIZE_BSIZE)):
        # Load hidden activations with appropriate mask
        hin_mask = (offs_bsize[:, None] < d2) & (
            offs_BSIZE[None, :] < BSIZE - BSIZE_i * BLOCK_SIZE_BSIZE
        )
        hin = tl.load(hin_ptrs, mask=hin_mask, other=0.0)

        # Load intermediate gradients with appropriate mask
        su_mask = (offs_BSIZE[:, None] < BSIZE - BSIZE_i * BLOCK_SIZE_BSIZE) & (
            offs_k[None, :] < k
        )
        g_U1s = tl.load(g_U1s_ptrs, mask=su_mask, other=0.0)

        # Compute dot product and accumulate
        accumulator1 += tl.dot(hin, g_U1s, input_precision="ieee")

        # Update pointers for next iteration
        hin_ptrs += BLOCK_SIZE_BSIZE * stride_hin_BSIZE
        g_U1s_ptrs += BLOCK_SIZE_BSIZE * stride_su_BSIZE

    # Calculate output pointers
    out_tmp = (
        batch_id * stride_out_l
        + stride_out_bsize * offs_bsize[:, None]
        + stride_out_k * offs_k[None, :]
    )
    grad_g_S1s_ptrs = grad_g_S1s_ptr + out_tmp

    # Create mask for output and store results
    out_mask = (offs_bsize[:, None] < d2) & (offs_k[None, :] < k)
    tl.store(grad_g_S1s_ptrs, accumulator1, mask=out_mask)


def calc_grad_S1s(hin, g_U1s):
    """
    Wrapper function for calculating gradients for S1s weights.

    This function computes the gradient for S1s by calculating the dot product
    of hidden activations with the intermediate gradient g_U1s.

    Args:
        hin: Hidden activation tensor of shape (d2, BSIZE)
        g_U1s: Intermediate gradient tensor of shape (L, BSIZE, k)

    Returns:
        torch.Tensor: Gradient tensor for S1s of shape (L, d2, k)
    """
    device = "cuda"

    # Extract dimensions from input tensors
    d2, BSIZE = hin.shape
    L, _, k = g_U1s.shape

    # Initialize output tensor
    grad_g_S1s = torch.empty((L, d2, k), dtype=torch.float32, device=device)

    # Calculate strides for memory layout
    stride_hin_bsize, stride_hin_BSIZE = hin.shape[1], 1
    stride_su_l, stride_su_BSIZE, stride_su_k = (
        g_U1s.shape[1] * g_U1s.shape[2],
        g_U1s.shape[2],
        1,
    )
    stride_out_l, stride_out_bsize, stride_out_k = (
        grad_g_S1s.shape[1] * grad_g_S1s.shape[2],
        grad_g_S1s.shape[2],
        1,
    )

    # Define the grid for kernel launch
    grid = lambda META: (
        L,
        triton.cdiv(d2, META["BLOCK_SIZE_d2"]) * triton.cdiv(k, META["BLOCK_SIZE_k"]),
    )

    # Launch the kernel
    calc_grad_S1s_kernel[grid](
        hin,
        g_U1s,
        grad_g_S1s,
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

    return grad_g_S1s


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_K": 128,
                "BLOCK_SIZE_d1": 256,
                "BLOCK_SIZE_BSIZE": 64,
                "GROUP_SIZE_K": 8,
            },
            num_stages=1,
            num_warps=4,
        ),
        # ... Additional configurations for performance tuning
        triton.Config(
            {
                "BLOCK_SIZE_K": 128,
                "BLOCK_SIZE_d1": 32,
                "BLOCK_SIZE_BSIZE": 64,
                "GROUP_SIZE_K": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
    ],
    key=["K", "BSIZE", "d1", "L"],
)
@triton.jit
def calc_grad_S2s_kernel(
    g_ptr,
    U2s_hin_ptr,
    grad_S2s_ptr,
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
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_d1: tl.constexpr,
    BLOCK_SIZE_BSIZE: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
):
    """
    Compute gradients for S2s weights.

    This kernel calculates the gradient for S2s by computing the dot product
    of the intermediate tensor U2s_hin with the gradient g.

    Args:
        g_ptr: Pointer to gradient tensor
        U2s_hin_ptr: Pointer to intermediate tensor (U2s dot hin)
        grad_S2s_ptr: Pointer to output gradient tensor for S2s
        K: Feature dimension
        BSIZE: Batch size dimension
        d1: Input dimension
        L: Number of layers
        stride_*: Stride values for tensor memory layout
        BLOCK_SIZE_*: Block sizes for tiling (constexpr)
        GROUP_SIZE_K: Group size for feature dimension (constexpr)
    """
    # Get program IDs for batch and parallel dimensions
    pid = tl.program_id(axis=1)
    batch_id = tl.program_id(axis=0)

    # Calculate the number of blocks and handle tiling
    num_pid_K = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_d1 = tl.cdiv(d1, BLOCK_SIZE_d1)
    num_pid_in_group = GROUP_SIZE_K * num_pid_d1
    group_id = pid // num_pid_in_group
    first_pid_K = group_id * GROUP_SIZE_K
    group_size_d1 = min(num_pid_K - first_pid_K, GROUP_SIZE_K)
    pid_K = first_pid_K + ((pid % num_pid_in_group) % group_size_d1)
    pid_d1 = (pid % num_pid_in_group) // group_size_d1

    # Calculate offsets for memory access
    offs_K = pid_K * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_d1 = pid_d1 * BLOCK_SIZE_d1 + tl.arange(0, BLOCK_SIZE_d1)
    offs_BSIZE = tl.arange(0, BLOCK_SIZE_BSIZE)

    # Ensure offsets are properly aligned
    offs_K = tl.max_contiguous(tl.multiple_of(offs_K, BLOCK_SIZE_K), BLOCK_SIZE_K)
    offs_d1 = tl.max_contiguous(tl.multiple_of(offs_d1, BLOCK_SIZE_d1), BLOCK_SIZE_d1)
    offs_BSIZE = tl.max_contiguous(
        tl.multiple_of(offs_BSIZE, BLOCK_SIZE_BSIZE), BLOCK_SIZE_BSIZE
    )

    # Set up pointers for accessing input tensors
    g_ptrs = g_ptr + (
        offs_BSIZE[:, None] * stride_g_BSIZE + offs_d1[None, :] * stride_g_d1
    )

    su_tmp = batch_id * stride_su_l + (
        offs_K[:, None] * stride_su_K + offs_BSIZE[None, :] * stride_su_BSIZE
    )
    U2s_hin_ptrs = U2s_hin_ptr + su_tmp

    # Initialize accumulator for results
    accumulator1 = tl.full(
        shape=(BLOCK_SIZE_K, BLOCK_SIZE_d1), value=0.0, dtype=tl.float32
    )

    # Process blocks in the batch dimension
    for BSIZE_i in range(0, tl.cdiv(BSIZE, BLOCK_SIZE_BSIZE)):
        # Load gradient values with appropriate mask
        g_mask = (offs_BSIZE[:, None] < BSIZE - BSIZE_i * BLOCK_SIZE_BSIZE) & (
            offs_d1[None, :] < d1
        )
        g = tl.load(g_ptrs, mask=g_mask, other=0.0)

        # Load intermediate tensor values with appropriate mask
        su_mask = (offs_K[:, None] < K) & (
            offs_BSIZE[None, :] < BSIZE - BSIZE_i * BLOCK_SIZE_BSIZE
        )
        U2s_hin = tl.load(U2s_hin_ptrs, mask=su_mask, other=0.0)

        # Compute dot product and accumulate
        accumulator1 += tl.dot(U2s_hin, g, input_precision="ieee")

        # Update pointers for next iteration
        g_ptrs += BLOCK_SIZE_BSIZE * stride_g_BSIZE
        U2s_hin_ptrs += BLOCK_SIZE_BSIZE * stride_su_BSIZE

    # Calculate output pointers
    out_tmp = (
        batch_id * stride_out_l
        + stride_out_K * offs_K[:, None]
        + stride_out_d1 * offs_d1[None, :]
    )
    grad_S2s_ptrs = grad_S2s_ptr + out_tmp

    # Create mask for output and store results
    out_mask = (offs_K[:, None] < K) & (offs_d1[None, :] < d1)

    tl.store(grad_S2s_ptrs, accumulator1, mask=out_mask)


def calc_grad_S2s(U2s_hin, g):
    """
    Wrapper function for calculating gradients for S2s weights.

    This function computes the gradient for S2s by calculating the dot product
    of the intermediate tensor U2s_hin with the gradient g.

    Args:
        U2s_hin: Intermediate tensor of shape (L, K, BSIZE) from first_pass_U2s_hin
        g: Gradient tensor of shape (BSIZE, d1)

    Returns:
        torch.Tensor: Gradient tensor for S2s of shape (L, K, d1)
    """
    device = "cuda"

    # Extract dimensions from input tensors
    L, K, BSIZE = U2s_hin.shape
    _, d1 = g.shape

    # Initialize output tensor
    grad_S2s = torch.empty((L, K, d1), dtype=torch.float32, device=device)

    # Calculate strides for memory layout
    stride_g_BSIZE, stride_g_d1 = g.shape[1], 1
    stride_su_l, stride_su_K, stride_su_BSIZE = (
        U2s_hin.shape[1] * U2s_hin.shape[2],
        U2s_hin.shape[2],
        1,
    )
    stride_out_l, stride_out_K, stride_out_d1 = (
        grad_S2s.shape[1] * grad_S2s.shape[2],
        grad_S2s.shape[2],
        1,
    )

    # Define the grid for kernel launch
    grid = lambda META: (
        L,
        triton.cdiv(K, META["BLOCK_SIZE_K"]) * triton.cdiv(d1, META["BLOCK_SIZE_d1"]),
    )

    # Launch the kernel
    calc_grad_S2s_kernel[grid](
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

    return grad_S2s


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_K": 128,
                "BLOCK_SIZE_BSIZE": 256,
                "BLOCK_SIZE_d2": 64,
                "GROUP_SIZE_K": 8,
            },
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_K": 64,
                "BLOCK_SIZE_BSIZE": 256,
                "BLOCK_SIZE_d2": 32,
                "GROUP_SIZE_K": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_K": 128,
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_d2": 32,
                "GROUP_SIZE_K": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_K": 128,
                "BLOCK_SIZE_BSIZE": 64,
                "BLOCK_SIZE_d2": 32,
                "GROUP_SIZE_K": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_K": 64,
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_d2": 32,
                "GROUP_SIZE_K": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_K": 128,
                "BLOCK_SIZE_BSIZE": 32,
                "BLOCK_SIZE_d2": 32,
                "GROUP_SIZE_K": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_K": 64,
                "BLOCK_SIZE_BSIZE": 32,
                "BLOCK_SIZE_d2": 32,
                "GROUP_SIZE_K": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_K": 32,
                "BLOCK_SIZE_BSIZE": 64,
                "BLOCK_SIZE_d2": 32,
                "GROUP_SIZE_K": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_K": 128,
                "BLOCK_SIZE_BSIZE": 256,
                "BLOCK_SIZE_d2": 128,
                "GROUP_SIZE_K": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_K": 256,
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_d2": 128,
                "GROUP_SIZE_K": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_K": 256,
                "BLOCK_SIZE_BSIZE": 64,
                "BLOCK_SIZE_d2": 128,
                "GROUP_SIZE_K": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_K": 64,
                "BLOCK_SIZE_BSIZE": 256,
                "BLOCK_SIZE_d2": 128,
                "GROUP_SIZE_K": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_K": 128,
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_d2": 128,
                "GROUP_SIZE_K": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_K": 128,
                "BLOCK_SIZE_BSIZE": 64,
                "BLOCK_SIZE_d2": 64,
                "GROUP_SIZE_K": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_K": 64,
                "BLOCK_SIZE_BSIZE": 128,
                "BLOCK_SIZE_d2": 64,
                "GROUP_SIZE_K": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_K": 128,
                "BLOCK_SIZE_BSIZE": 32,
                "BLOCK_SIZE_d2": 64,
                "GROUP_SIZE_K": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
    ],
    key=["K", "d2", "BSIZE", "L"],
)
@triton.jit
def first_pass_U2s_hin_kernel(
    hin_ptr,
    U2s_ptr,
    U2s_h_in_ptr,
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
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_BSIZE: tl.constexpr,
    BLOCK_SIZE_d2: tl.constexpr,
    GROUP_SIZE_K: tl.constexpr,
):
    """
    Computes the product of U2s and hin in a tiled manner.

    This kernel performs the first pass computation for a larger operation,
    specifically calculating U2s_h_in = U2s @ hin.

    Args:
        hin_ptr: Pointer to the input tensor hin (d2, BSIZE).
        U2s_ptr: Pointer to the input tensor U2s (L, K, d2).
        U2s_h_in_ptr: Pointer to the output tensor U2s_h_in (L, K, BSIZE).
        K: Dimension K of the U2s tensor.
        d2: Dimension d2 of the U2s and hin tensors.
        BSIZE: Dimension BSIZE of the hin tensor.
        L: Dimension L (batch dimension) of the U2s and U2s_h_in tensors.
        stride_hin_d2: Stride of the hin tensor along dimension d2.
        stride_hin_BSIZE: Stride of the hin tensor along dimension BSIZE.
        stride_su_l: Stride of the U2s tensor along dimension L.
        stride_su_K: Stride of the U2s tensor along dimension K.
        stride_su_d2: Stride of the U2s tensor along dimension d2.
        stride_out_l: Stride of the U2s_h_in tensor along dimension L.
        stride_out_K: Stride of the U2s_h_in tensor along dimension K.
        stride_out_BSIZE: Stride of the U2s_h_in tensor along dimension BSIZE.
        BLOCK_SIZE_K: Tile size for dimension K.
        BLOCK_SIZE_BSIZE: Tile size for dimension BSIZE.
        BLOCK_SIZE_d2: Tile size for dimension d2.
        GROUP_SIZE_K: Group size for K dimension, used for managing thread blocks.
    """
    # Get program IDs for batch and other dimensions
    pid = tl.program_id(axis=1)
    batch_id = tl.program_id(axis=0)  # This corresponds to L

    # Calculate the number of program IDs (blocks) for K and BSIZE dimensions
    num_pid_K = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_BSIZE = tl.cdiv(BSIZE, BLOCK_SIZE_BSIZE)

    # Grouping logic for managing thread blocks, particularly for the K dimension
    num_pid_in_group = GROUP_SIZE_K * num_pid_BSIZE
    group_id = pid // num_pid_in_group
    first_pid_K = group_id * GROUP_SIZE_K

    # Determine the actual size of the current group for K dimension (can be smaller at the boundary)
    group_size_BSIZE = min(
        num_pid_K - first_pid_K, GROUP_SIZE_K
    )  # Note: Name seems misleading, it's related to K blocks

    # Calculate specific program ID for K and BSIZE within the group
    pid_K = first_pid_K + ((pid % num_pid_in_group) % group_size_BSIZE)
    pid_BSIZE = (pid % num_pid_in_group) // group_size_BSIZE

    # Define offsets for K, BSIZE, and d2 dimensions for the current block
    offs_K = pid_K * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_BSIZE = pid_BSIZE * BLOCK_SIZE_BSIZE + tl.arange(0, BLOCK_SIZE_BSIZE)
    offs_d2 = tl.arange(0, BLOCK_SIZE_d2)

    # Ensure offsets are contiguous and aligned to block sizes
    offs_K = tl.max_contiguous(tl.multiple_of(offs_K, BLOCK_SIZE_K), BLOCK_SIZE_K)
    offs_BSIZE = tl.max_contiguous(
        tl.multiple_of(offs_BSIZE, BLOCK_SIZE_BSIZE), BLOCK_SIZE_BSIZE
    )
    offs_d2 = tl.max_contiguous(tl.multiple_of(offs_d2, BLOCK_SIZE_d2), BLOCK_SIZE_d2)

    # Calculate pointers to the current block in hin
    hin_ptrs = hin_ptr + (
        offs_d2[:, None] * stride_hin_d2 + offs_BSIZE[None, :] * stride_hin_BSIZE
    )

    # Calculate pointers to the current block in U2s, including batch offset
    su_tmp = batch_id * stride_su_l + (
        offs_K[:, None] * stride_su_K + offs_d2[None, :] * stride_su_d2
    )
    U2s_ptrs = U2s_ptr + su_tmp

    # Initialize accumulator for the dot product
    accumulator1 = tl.full(
        shape=(BLOCK_SIZE_K, BLOCK_SIZE_BSIZE), value=0.0, dtype=tl.float32
    )

    # Loop over the d2 dimension in blocks
    for d2_i in range(0, tl.cdiv(d2, BLOCK_SIZE_d2)):
        # Create masks for loading hin to handle boundary conditions
        hin_mask = (offs_d2[:, None] < d2 - d2_i * BLOCK_SIZE_d2) & (
            offs_BSIZE[None, :] < BSIZE
        )
        hin = tl.load(hin_ptrs, mask=hin_mask, other=0.0)

        # Create masks for loading U2s to handle boundary conditions
        su_mask = (offs_K[:, None] < K) & (offs_d2[None, :] < d2 - d2_i * BLOCK_SIZE_d2)
        U2s = tl.load(U2s_ptrs, mask=su_mask, other=0.0)

        # Perform matrix multiplication (dot product) for the current blocks
        accumulator1 += tl.dot(U2s, hin, input_precision="ieee")

        # Advance pointers to the next block in d2 dimension
        hin_ptrs += BLOCK_SIZE_d2 * stride_hin_d2
        U2s_ptrs += BLOCK_SIZE_d2 * stride_su_d2

    # Calculate pointers to the output tensor U2s_h_in, including batch offset
    out_tmp = (
        batch_id * stride_out_l
        + stride_out_K * offs_K[:, None]
        + stride_out_BSIZE * offs_BSIZE[None, :]
    )
    U2s_h_in_ptrs = U2s_h_in_ptr + out_tmp

    # Create mask for storing the output to handle boundary conditions
    out_mask = (offs_K[:, None] < K) & (offs_BSIZE[None, :] < BSIZE)

    # Store the accumulated result
    tl.store(U2s_h_in_ptrs, accumulator1, mask=out_mask)
