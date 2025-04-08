import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_d1': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=1,
                      num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_d1': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_d1': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_d1': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_d1': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_d1': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_d1': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=5,
                       num_warps=2),
         triton.Config({'BLOCK_SIZE_BSIZE': 32, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_d1': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=5,
                       num_warps=2),
         triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_d1': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=3,
                       num_warps=8),
         triton.Config({'BLOCK_SIZE_BSIZE': 256, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_d1': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=3,
                       num_warps=8),
         triton.Config({'BLOCK_SIZE_BSIZE': 256, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_d1': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_d1': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_d1': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_d1': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_d1': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_d1': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4)
    ],
    key=['BSIZE', 'K', 'd1', 'L'],
)
@triton.jit
def first_pass_gU1s_g_S2s_kernel(
        g_ptr, U1s_ptr, S2s_ptr, g_U1s_ptr, g_S2s_ptr,
        BSIZE, K, d1, L,
        stride_g_bsize, stride_g_d1,
        stride_su_l, stride_su_d1, stride_su_k,
        stride_out_l, stride_out_bsize, stride_out_k,
        BLOCK_SIZE_BSIZE: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_d1: tl.constexpr,
        GROUP_SIZE_BSIZE: tl.constexpr
):
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
    offs_k = pid_k *  BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_d1 = tl.arange(0, BLOCK_SIZE_d1)

    g_ptrs = g_ptr + (offs_bsize[:, None] * stride_g_bsize + offs_d1[None, :] * stride_g_d1)

    su_tmp = batch_id * stride_su_l + (offs_d1[:, None] * stride_su_d1 + offs_k[None, :] * stride_su_k)
    U1s_ptrs = U1s_ptr + su_tmp
    S2s_ptrs = S2s_ptr + su_tmp

    accumulator1 = tl.full(shape=(BLOCK_SIZE_BSIZE, BLOCK_SIZE_K), value=0.0, dtype=tl.float32)
    accumulator2 = tl.full(shape=(BLOCK_SIZE_BSIZE, BLOCK_SIZE_K), value=0.0, dtype=tl.float32)
    
    
    for d1_i in range(0, tl.cdiv(d1, BLOCK_SIZE_d1)):
        g = tl.load(g_ptrs, mask=(offs_d1[None, :] < d1 - d1_i * BLOCK_SIZE_d1), other=0.0)
        
        su_mask = (offs_d1[:, None] < d1 - d1_i * BLOCK_SIZE_d1)
        U1s = tl.load(U1s_ptrs, mask=su_mask, other=0.0)
        S2s = tl.load(S2s_ptrs, mask=su_mask, other=0.0)
        
        accumulator1 += tl.dot(g, U1s)
        accumulator2 += tl.dot(g, S2s)
        
        g_ptrs += BLOCK_SIZE_d1 * stride_g_d1
        U1s_ptrs += BLOCK_SIZE_d1 * stride_su_d1
        S2s_ptrs += BLOCK_SIZE_d1 * stride_su_d1

    out_tmp = batch_id * stride_out_l + stride_out_bsize * offs_bsize[:, None] + stride_out_k * offs_k[None, :]
    g_U1s_ptrs = g_U1s_ptr + out_tmp
    g_S2s_ptrs = g_S2s_ptr + out_tmp
    
    out_mask = (offs_bsize[:, None] < BSIZE) & (offs_k[None, :] < K)
    
    tl.store(g_U1s_ptrs, accumulator1, mask=out_mask)
    tl.store(g_S2s_ptrs, accumulator2, mask=out_mask)

def first_pass_gU1s_g_S2s(g, U1s, S2s):
    # assert g.shape[1] == U1s.shape[1], "Incompatible dimensions"
    # assert g.shape[1] == S2s.shape[1], "Incompatible dimensions"
    # assert g.is_contiguous(), "Matrix A must be contiguous"
    # assert U1s.is_contiguous(), "Matrix A must be contiguous"
    # assert S2s.is_contiguous(), "Matrix A must be contiguous"
    # assert U1s.stride() == S2s.stride(), "Matrix A must be contiguous"
    
    BSIZE, d1 = g.shape
    L, _, K = U1s.shape
    
    g_U1s = torch.empty((L, BSIZE, K), dtype=torch.float16, device='cuda')
    g_S2s = torch.empty((L, BSIZE, K), dtype=torch.float16, device='cuda')

    # stride_g_bsize, stride_g_d1 = g.stride()
    # stride_su_l, stride_su_d1, stride_su_k = U1s.stride()
    # stride_out_l, stride_out_bsize, stride_out_k = g_U1s.stride()
    stride_g_bsize, stride_g_d1 = g.shape[1], 1
    stride_su_l, stride_su_d1, stride_su_k = U1s.shape[1] * U1s.shape[2], U1s.shape[2], 1
    stride_out_l, stride_out_bsize, stride_out_k = g_U1s.shape[1] * g_U1s.shape[2], g_U1s.shape[2], 1
    
    # assert g_U1s.stride() == g_S2s.stride(), "Matrix A must be contiguous"
    
    grid = lambda META: (L, triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]), )
    
    first_pass_gU1s_g_S2s_kernel[grid](
        g, U1s, S2s, g_U1s, g_S2s,
        BSIZE, K, d1, L,
        stride_g_bsize, stride_g_d1,
        stride_su_l, stride_su_d1, stride_su_k,
        stride_out_l, stride_out_bsize, stride_out_k
    )
    
    return g_U1s, g_S2s
  
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_d2': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=1,
                      num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_d2': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_d2': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_d2': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_d2': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_d2': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_d2': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=5,
                       num_warps=2),
         triton.Config({'BLOCK_SIZE_BSIZE': 32, 'BLOCK_SIZE_d2': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=5,
                       num_warps=2),
         triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_d2': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=3,
                       num_warps=8),
         triton.Config({'BLOCK_SIZE_BSIZE': 256, 'BLOCK_SIZE_d2': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=3,
                       num_warps=8),
         triton.Config({'BLOCK_SIZE_BSIZE': 256, 'BLOCK_SIZE_d2': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_d2': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_d2': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_d2': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_d2': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4),
         triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_d2': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                       num_warps=4)
    ],
    key=['BSIZE', 'd2', 'K', 'L'],
)
@triton.jit
def second_pass_gUS11_22_kernel(
        g_U1s_ptr, g_S2s_ptr, S1s_ptr, U2s_ptr, out_ptr,
        BSIZE, d2, K, L,
        stride_g_U1s2_l, stride_g_U1s2_bsize, stride_g_U1s2_k,
        stride_us_l, stride_us_k, stride_us_d2,
        stride_out_bsize, stride_out_d2,
        BLOCK_SIZE_BSIZE: tl.constexpr, BLOCK_SIZE_d2: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_BSIZE: tl.constexpr
):
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
    offs_d2 = pid_d2 *  BLOCK_SIZE_d2 + tl.arange(0, BLOCK_SIZE_d2)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    in_tmp = offs_bsize[:, None] * stride_g_U1s2_bsize + offs_k[None, :] * stride_g_U1s2_k
    us_tmp = offs_k[:, None] * stride_us_k + offs_d2[None, :] * stride_us_d2

    accumulator = tl.full(shape=(BLOCK_SIZE_BSIZE, BLOCK_SIZE_d2), value=0.0, dtype=tl.float32)
    
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
            
            accumulator += tl.dot(g_U1s, S1s)
            accumulator += tl.dot(g_S2s, U2s)

            in_inc = BLOCK_SIZE_K * stride_g_U1s2_k
            g_U1s_ptrs += in_inc
            g_S2s_ptrs += in_inc
            
            us_inc = BLOCK_SIZE_K * stride_us_k
            S1s_ptrs += us_inc
            U2s_ptrs += us_inc
    
    accumulator *= (1.0/ (2.0 * L))

    out_ptrs = out_ptr + stride_out_bsize * offs_bsize[:, None] + stride_out_d2 * offs_d2[None, :]
    out_mask = (offs_bsize[:, None] < BSIZE) & (offs_d2[None, :] < d2)
    
    tl.store(out_ptrs, accumulator, mask=out_mask)

def second_pass_gUS11_22(g_U1s, g_S2s, S1s, U2s):
    # assert g_U1s.shape[2] == S1s.shape[1], "Incompatible dimensions"
    # assert g_S2s.shape[2] == U2s.shape[1], "Incompatible dimensions"
    # assert g_U1s.is_contiguous(), "Matrix A must be contiguous"
    # assert g_S2s.is_contiguous(), "Matrix A must be contiguous"
    # assert S1s.is_contiguous(), "Matrix A must be contiguous"
    # assert U2s.is_contiguous(), "Matrix A must be contiguous"
    # assert S1s.stride() == U2s.stride(), "Matrix A must be contiguous"
    # assert g_U1s.stride() == g_S2s.stride(), "Matrix A must be contiguous"
    
    L, BSIZE, K = g_U1s.shape
    _, _, d2 = S1s.shape
    
    out = torch.empty((BSIZE, d2), dtype=torch.float16, device='cuda')

    # stride_g_U1s2_l, stride_g_U1s2_bsize, stride_g_U1s2_k = g_U1s.stride()
    # stride_us_l, stride_us_k, stride_us_d2 = S1s.stride()
    # stride_out_bsize, stride_out_d2 = out.stride()
    stride_g_U1s2_l, stride_g_U1s2_bsize, stride_g_U1s2_k = g_U1s.shape[1] * g_U1s.shape[2], g_U1s.shape[2], 1
    stride_us_l, stride_us_k, stride_us_d2 = S1s.shape[1] * S1s.shape[2], S1s.shape[2], 1
    stride_out_bsize, stride_out_d2 = out.shape[1], 1
    
    grid = lambda META: (triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"]) * triton.cdiv(d2, META["BLOCK_SIZE_d2"]), )
    
    second_pass_gUS11_22_kernel[grid](
        g_U1s, g_S2s, S1s, U2s, out,
        BSIZE, d2, K, L,
        stride_g_U1s2_l, stride_g_U1s2_bsize, stride_g_U1s2_k,
        stride_us_l, stride_us_k, stride_us_d2,
        stride_out_bsize, stride_out_d2,
    )
    
    return out # grad
  
@triton.autotune(
    configs=[
    triton.Config({'BLOCK_SIZE_d2': 128, 'BLOCK_SIZE_k': 256, 'BLOCK_SIZE_BSIZE': 64, 'GROUP_SIZE_d2': 8}, num_stages=1, num_warps=4),
    triton.Config({'BLOCK_SIZE_d2': 64, 'BLOCK_SIZE_k': 256, 'BLOCK_SIZE_BSIZE': 32, 'GROUP_SIZE_d2': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_d2': 128, 'BLOCK_SIZE_k': 128, 'BLOCK_SIZE_BSIZE': 32, 'GROUP_SIZE_d2': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_d2': 128, 'BLOCK_SIZE_k': 64, 'BLOCK_SIZE_BSIZE': 32, 'GROUP_SIZE_d2': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_d2': 64, 'BLOCK_SIZE_k': 128, 'BLOCK_SIZE_BSIZE': 32, 'GROUP_SIZE_d2': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_d2': 128, 'BLOCK_SIZE_k': 32, 'BLOCK_SIZE_BSIZE': 32, 'GROUP_SIZE_d2': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_d2': 64, 'BLOCK_SIZE_k': 32, 'BLOCK_SIZE_BSIZE': 32, 'GROUP_SIZE_d2': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_d2': 32, 'BLOCK_SIZE_k': 64, 'BLOCK_SIZE_BSIZE': 32, 'GROUP_SIZE_d2': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_d2': 128, 'BLOCK_SIZE_k': 256, 'BLOCK_SIZE_BSIZE': 128, 'GROUP_SIZE_d2': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_d2': 256, 'BLOCK_SIZE_k': 128, 'BLOCK_SIZE_BSIZE': 128, 'GROUP_SIZE_d2': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_d2': 256, 'BLOCK_SIZE_k': 64, 'BLOCK_SIZE_BSIZE': 128, 'GROUP_SIZE_d2': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_d2': 64, 'BLOCK_SIZE_k': 256, 'BLOCK_SIZE_BSIZE': 128, 'GROUP_SIZE_d2': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_d2': 128, 'BLOCK_SIZE_k': 128, 'BLOCK_SIZE_BSIZE': 128, 'GROUP_SIZE_d2': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_d2': 128, 'BLOCK_SIZE_k': 64, 'BLOCK_SIZE_BSIZE': 64, 'GROUP_SIZE_d2': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_d2': 64, 'BLOCK_SIZE_k': 128, 'BLOCK_SIZE_BSIZE': 64, 'GROUP_SIZE_d2': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_d2': 128, 'BLOCK_SIZE_k': 32, 'BLOCK_SIZE_BSIZE': 64, 'GROUP_SIZE_d2': 8}, num_stages=4, num_warps=4)
],
    key=['d2', 'k', 'BSIZE', 'L'],
)
@triton.jit
def calc_grad_S1s_kernel(
        hin_ptr, g_U1s_ptr, grad_g_S1s_ptr,
        d2, k, BSIZE, L,
        stride_hin_bsize, stride_hin_BSIZE,
        stride_su_l, stride_su_BSIZE, stride_su_k,
        stride_out_l, stride_out_bsize, stride_out_k,
        BLOCK_SIZE_d2: tl.constexpr, BLOCK_SIZE_k: tl.constexpr, BLOCK_SIZE_BSIZE: tl.constexpr,
        GROUP_SIZE_d2: tl.constexpr
):
    pid = tl.program_id(axis=1)
    batch_id = tl.program_id(axis=0)
    
    num_pid_bsize = tl.cdiv(d2, BLOCK_SIZE_d2)
    num_pid_k = tl.cdiv(k, BLOCK_SIZE_k)
    num_pid_in_group = GROUP_SIZE_d2 * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_bsize = group_id * GROUP_SIZE_d2
    group_size_bsize = min(num_pid_bsize - first_pid_bsize, GROUP_SIZE_d2)
    pid_bsize = first_pid_bsize + ((pid % num_pid_in_group) % group_size_bsize)
    pid_k = (pid % num_pid_in_group) // group_size_bsize

    offs_bsize = pid_bsize * BLOCK_SIZE_d2 + tl.arange(0, BLOCK_SIZE_d2)
    offs_k = pid_k *  BLOCK_SIZE_k + tl.arange(0, BLOCK_SIZE_k)
    offs_BSIZE = tl.arange(0, BLOCK_SIZE_BSIZE)

    offs_bsize = tl.max_contiguous(tl.multiple_of(offs_bsize, BLOCK_SIZE_d2), BLOCK_SIZE_d2)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_SIZE_k), BLOCK_SIZE_k)
    offs_BSIZE = tl.max_contiguous(tl.multiple_of(offs_BSIZE, BLOCK_SIZE_BSIZE), BLOCK_SIZE_BSIZE)
    
    hin_ptrs = hin_ptr + (offs_bsize[:, None] * stride_hin_bsize + offs_BSIZE[None, :] * stride_hin_BSIZE)

    su_tmp = batch_id * stride_su_l + (offs_BSIZE[:, None] * stride_su_BSIZE + offs_k[None, :] * stride_su_k)
    g_U1s_ptrs = g_U1s_ptr + su_tmp

    accumulator1 = tl.full(shape=(BLOCK_SIZE_d2, BLOCK_SIZE_k), value=0.0, dtype=tl.float32)
    accumulator2 = tl.full(shape=(BLOCK_SIZE_d2, BLOCK_SIZE_k), value=0.0, dtype=tl.float32)
    
    for BSIZE_i in range(0, tl.cdiv(BSIZE, BLOCK_SIZE_BSIZE)):
        hin_mask = (offs_bsize[:, None] < d2) & (offs_BSIZE[None, :] < BSIZE - BSIZE_i * BLOCK_SIZE_BSIZE)
        hin = tl.load(hin_ptrs, mask=hin_mask, other=0.0)
        
        su_mask = (offs_BSIZE[:, None] < BSIZE - BSIZE_i * BLOCK_SIZE_BSIZE) & (offs_k[None, :] < k)
        g_U1s = tl.load(g_U1s_ptrs, mask=su_mask, other=0.0)
        
        accumulator1 += tl.dot(hin, g_U1s)
        
        hin_ptrs += BLOCK_SIZE_BSIZE * stride_hin_BSIZE
        g_U1s_ptrs += BLOCK_SIZE_BSIZE * stride_su_BSIZE

    accumulator1 = accumulator1.to(tl.float16)
    accumulator2 = accumulator2.to(tl.float16)

    out_tmp = batch_id * stride_out_l + stride_out_bsize * offs_bsize[:, None] + stride_out_k * offs_k[None, :]
    grad_g_S1s_ptrs = grad_g_S1s_ptr + out_tmp
    
    out_mask = (offs_bsize[:, None] < d2) & (offs_k[None, :] < k)
    
    tl.store(grad_g_S1s_ptrs, accumulator1, mask=out_mask)

def calc_grad_S1s(hin, g_U1s):
    device = 'cuda'
    # assert hin.shape[1] == g_U1s.shape[1], "Incompatible dimensions"
    # assert hin.is_contiguous(), "Matrix A must be contiguous"
    # assert g_U1s.is_contiguous(), "Matrix A must be contiguous"
    
    d2, BSIZE = hin.shape
    L, _, k = g_U1s.shape
    
    grad_g_S1s = torch.empty((L, d2, k), dtype=torch.float16, device=device)

    # stride_hin_bsize, stride_hin_BSIZE = hin.stride()
    # stride_su_l, stride_su_BSIZE, stride_su_k = g_U1s.stride()
    # stride_out_l, stride_out_bsize, stride_out_k = grad_g_S1s.stride()
    stride_hin_bsize, stride_hin_BSIZE = hin.shape[1], 1
    stride_su_l, stride_su_BSIZE, stride_su_k = g_U1s.shape[1] * g_U1s.shape[2], g_U1s.shape[2], 1
    stride_out_l, stride_out_bsize, stride_out_k = grad_g_S1s.shape[1] * grad_g_S1s.shape[2], grad_g_S1s.shape[2], 1
    
    grid = lambda META: (L, triton.cdiv(d2, META["BLOCK_SIZE_d2"]) * triton.cdiv(k, META["BLOCK_SIZE_k"]), )
    
    calc_grad_S1s_kernel[grid](
        hin, g_U1s, grad_g_S1s,
        d2, k, BSIZE, L,
        stride_hin_bsize, stride_hin_BSIZE,
        stride_su_l, stride_su_BSIZE, stride_su_k,
        stride_out_l, stride_out_bsize, stride_out_k
    )
    
    return grad_g_S1s
  
@triton.autotune(
    configs=[
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_BSIZE': 256, 'BLOCK_SIZE_d2': 64, 'GROUP_SIZE_K': 8}, num_stages=1, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_BSIZE': 256, 'BLOCK_SIZE_d2': 32, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_d2': 32, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_d2': 32, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_d2': 32, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_BSIZE': 32, 'BLOCK_SIZE_d2': 32, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_BSIZE': 32, 'BLOCK_SIZE_d2': 32, 'GROUP_SIZE_K': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_d2': 32, 'GROUP_SIZE_K': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_BSIZE': 256, 'BLOCK_SIZE_d2': 128, 'GROUP_SIZE_K': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_d2': 128, 'GROUP_SIZE_K': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_d2': 128, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_BSIZE': 256, 'BLOCK_SIZE_d2': 128, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_d2': 128, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_d2': 64, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_d2': 64, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_BSIZE': 32, 'BLOCK_SIZE_d2': 64, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4)
],
    key=['K', 'd2', 'BSIZE', 'L'],
)
@triton.jit
def first_pass_U2s_hin_kernel(
        hin_ptr, U2s_ptr, U2s_h_in_ptr,
        K, d2, BSIZE, L,
        stride_hin_d2, stride_hin_BSIZE,
        stride_su_l, stride_su_K, stride_su_d2,
        stride_out_l, stride_out_K, stride_out_BSIZE,
        BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_BSIZE: tl.constexpr, BLOCK_SIZE_d2: tl.constexpr,
        GROUP_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=1)
    batch_id = tl.program_id(axis=0)
    
    num_pid_K = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_BSIZE = tl.cdiv(BSIZE, BLOCK_SIZE_BSIZE)
    num_pid_in_group = GROUP_SIZE_K * num_pid_BSIZE
    group_id = pid // num_pid_in_group
    first_pid_K = group_id * GROUP_SIZE_K
    group_size_BSIZE = min(num_pid_K - first_pid_K, GROUP_SIZE_K)
    pid_K = first_pid_K + ((pid % num_pid_in_group) % group_size_BSIZE)
    pid_BSIZE = (pid % num_pid_in_group) // group_size_BSIZE

    offs_K = pid_K * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_BSIZE = pid_BSIZE *  BLOCK_SIZE_BSIZE + tl.arange(0, BLOCK_SIZE_BSIZE)
    offs_d2 = tl.arange(0, BLOCK_SIZE_d2)

    offs_K = tl.max_contiguous(tl.multiple_of(offs_K, BLOCK_SIZE_K), BLOCK_SIZE_K)
    offs_BSIZE = tl.max_contiguous(tl.multiple_of(offs_BSIZE, BLOCK_SIZE_BSIZE), BLOCK_SIZE_BSIZE)
    offs_d2 = tl.max_contiguous(tl.multiple_of(offs_d2, BLOCK_SIZE_d2), BLOCK_SIZE_d2)
    
    hin_ptrs = hin_ptr + (offs_d2[:, None] * stride_hin_d2 + offs_BSIZE[None, :] * stride_hin_BSIZE)

    su_tmp = batch_id * stride_su_l + (offs_K[:, None] * stride_su_K + offs_d2[None, :] * stride_su_d2)
    U2s_ptrs = U2s_ptr + su_tmp

    accumulator1 = tl.full(shape=(BLOCK_SIZE_K, BLOCK_SIZE_BSIZE), value=0.0, dtype=tl.float32)
    
    for d2_i in range(0, tl.cdiv(d2, BLOCK_SIZE_d2)):
        hin_mask = (offs_d2[:, None] < d2 - d2_i * BLOCK_SIZE_d2) & (offs_BSIZE[None, :] < BSIZE)
        hin = tl.load(hin_ptrs, mask=hin_mask, other=0.0)
        
        su_mask = (offs_K[:, None] < K) & (offs_d2[None, :] < d2 - d2_i * BLOCK_SIZE_d2)
        U2s = tl.load(U2s_ptrs, mask=su_mask, other=0.0)
        
        accumulator1 += tl.dot(U2s, hin)
        
        hin_ptrs += BLOCK_SIZE_d2 * stride_hin_d2
        U2s_ptrs += BLOCK_SIZE_d2 * stride_su_d2

    accumulator1 = accumulator1.to(tl.float16)

    out_tmp = batch_id * stride_out_l + stride_out_K * offs_K[:, None] + stride_out_BSIZE * offs_BSIZE[None, :]
    U2s_h_in_ptrs = U2s_h_in_ptr + out_tmp
    
    out_mask = (offs_K[:, None] < K) & (offs_BSIZE[None, :] < BSIZE)
    
    tl.store(U2s_h_in_ptrs, accumulator1, mask=out_mask)

def first_pass_U2s_hin(U2s, hin):
    device = 'cuda'
    # assert U2s.shape[2] == hin.shape[0], "Incompatible dimensions"
    # assert hin.is_contiguous(), "Matrix A must be contiguous"
    # assert U2s.is_contiguous(), "Matrix A must be contiguous"
    
    L, K, d2 = U2s.shape
    _, BSIZE = hin.shape
    
    U2s_h_in = torch.empty((L, K, BSIZE), dtype=torch.float16, device=device)

    # stride_hin_d2, stride_hin_BSIZE = hin.stride()
    # stride_su_l, stride_su_K, stride_su_d2 = U2s.stride()
    # stride_out_l, stride_out_K, stride_out_BSIZE = U2s_h_in.stride()
    stride_hin_d2, stride_hin_BSIZE = hin.shape[1], 1
    stride_su_l, stride_su_K, stride_su_d2 = U2s.shape[1] * U2s.shape[2], U2s.shape[2], 1
    stride_out_l, stride_out_K, stride_out_BSIZE = U2s_h_in.shape[1] * U2s_h_in.shape[2], U2s_h_in.shape[2], 1

    BLOCK_SIZE_K, BLOCK_SIZE_BSIZE, BLOCK_SIZE_d2 = 128, 256, 64
    GROUP_SIZE_K = 8
    
    grid = lambda META: (L, triton.cdiv(K, META["BLOCK_SIZE_K"]) * triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"]), )
    
    first_pass_U2s_hin_kernel[grid](
        hin, U2s, U2s_h_in,
        K, d2, BSIZE, L,
        stride_hin_d2, stride_hin_BSIZE,
        stride_su_l, stride_su_K, stride_su_d2,
        stride_out_l, stride_out_K, stride_out_BSIZE
    )
    
    return U2s_h_in
  
@triton.autotune(
    configs=[
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_d1': 256, 'BLOCK_SIZE_BSIZE': 64, 'GROUP_SIZE_K': 8}, num_stages=1, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_d1': 256, 'BLOCK_SIZE_BSIZE': 32, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_d1': 128, 'BLOCK_SIZE_BSIZE': 32, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_d1': 64, 'BLOCK_SIZE_BSIZE': 32, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_d1': 128, 'BLOCK_SIZE_BSIZE': 32, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_d1': 32, 'BLOCK_SIZE_BSIZE': 32, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_d1': 32, 'BLOCK_SIZE_BSIZE': 32, 'GROUP_SIZE_K': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_d1': 64, 'BLOCK_SIZE_BSIZE': 32, 'GROUP_SIZE_K': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_d1': 256, 'BLOCK_SIZE_BSIZE': 128, 'GROUP_SIZE_K': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_d1': 128, 'BLOCK_SIZE_BSIZE': 128, 'GROUP_SIZE_K': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_d1': 64, 'BLOCK_SIZE_BSIZE': 128, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_d1': 256, 'BLOCK_SIZE_BSIZE': 128, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_d1': 128, 'BLOCK_SIZE_BSIZE': 128, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_d1': 64, 'BLOCK_SIZE_BSIZE': 64, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_d1': 128, 'BLOCK_SIZE_BSIZE': 64, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_d1': 32, 'BLOCK_SIZE_BSIZE': 64, 'GROUP_SIZE_K': 8}, num_stages=4, num_warps=4)
],
    key=['K', 'BSIZE', 'd1', 'L'],
)
@triton.jit
def calc_grad_S2s_kernel(
        g_ptr, U2s_hin_ptr, grad_S2s_ptr,
        K, BSIZE, d1, L,
        stride_g_BSIZE, stride_g_d1,
        stride_su_l, stride_su_K, stride_su_BSIZE,
        stride_out_l, stride_out_K, stride_out_d1,
        BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_d1: tl.constexpr, BLOCK_SIZE_BSIZE: tl.constexpr,
        GROUP_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=1)
    batch_id = tl.program_id(axis=0)
    
    num_pid_K = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_d1 = tl.cdiv(d1, BLOCK_SIZE_d1)
    num_pid_in_group = GROUP_SIZE_K * num_pid_d1
    group_id = pid // num_pid_in_group
    first_pid_K = group_id * GROUP_SIZE_K
    group_size_d1 = min(num_pid_K - first_pid_K, GROUP_SIZE_K)
    pid_K = first_pid_K + ((pid % num_pid_in_group) % group_size_d1)
    pid_d1 = (pid % num_pid_in_group) // group_size_d1

    offs_K = pid_K * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_d1 = pid_d1 *  BLOCK_SIZE_d1 + tl.arange(0, BLOCK_SIZE_d1)
    offs_BSIZE = tl.arange(0, BLOCK_SIZE_BSIZE)

    offs_K = tl.max_contiguous(tl.multiple_of(offs_K, BLOCK_SIZE_K), BLOCK_SIZE_K)
    offs_d1 = tl.max_contiguous(tl.multiple_of(offs_d1, BLOCK_SIZE_d1), BLOCK_SIZE_d1)
    offs_BSIZE = tl.max_contiguous(tl.multiple_of(offs_BSIZE, BLOCK_SIZE_BSIZE), BLOCK_SIZE_BSIZE)
    
    g_ptrs = g_ptr + (offs_BSIZE[:, None] * stride_g_BSIZE + offs_d1[None, :] * stride_g_d1)

    su_tmp = batch_id * stride_su_l + (offs_K[:, None] * stride_su_K + offs_BSIZE[None, :] * stride_su_BSIZE)
    U2s_hin_ptrs = U2s_hin_ptr + su_tmp

    accumulator1 = tl.full(shape=(BLOCK_SIZE_K, BLOCK_SIZE_d1), value=0.0, dtype=tl.float32)
    
    for BSIZE_i in range(0, tl.cdiv(BSIZE, BLOCK_SIZE_BSIZE)):
        g_mask = (offs_BSIZE[:, None] < BSIZE - BSIZE_i * BLOCK_SIZE_BSIZE) & (offs_d1[None, :] < d1)
        g = tl.load(g_ptrs, mask=g_mask, other=0.0)
        
        su_mask = (offs_K[:, None] < K) & (offs_BSIZE[None, :] < BSIZE - BSIZE_i * BLOCK_SIZE_BSIZE)
        U2s_hin = tl.load(U2s_hin_ptrs, mask=su_mask, other=0.0)
        
        accumulator1 += tl.dot(U2s_hin, g)
        
        g_ptrs += BLOCK_SIZE_BSIZE * stride_g_BSIZE
        U2s_hin_ptrs += BLOCK_SIZE_BSIZE * stride_su_BSIZE

    accumulator1 = accumulator1.to(tl.float16)

    out_tmp = batch_id * stride_out_l + stride_out_K * offs_K[:, None] + stride_out_d1 * offs_d1[None, :]
    grad_S2s_ptrs = grad_S2s_ptr + out_tmp
    
    out_mask = (offs_K[:, None] < K) & (offs_d1[None, :] < d1)
    
    tl.store(grad_S2s_ptrs, accumulator1, mask=out_mask)

def calc_grad_S2s(U2s_hin, g):
    device = 'cuda'
    # assert U2s_hin.shape[2] == g.shape[0], "Incompatible dimensions"
    # assert g.is_contiguous(), "Matrix A must be contiguous"
    # assert U2s_hin.is_contiguous(), "Matrix A must be contiguous"
    
    L, K, BSIZE = U2s_hin.shape
    _, d1 = g.shape
    
    grad_S2s = torch.empty((L, K, d1), dtype=torch.float16, device=device)

    # stride_g_BSIZE, stride_g_d1 = g.stride()
    # stride_su_l, stride_su_K, stride_su_BSIZE = U2s_hin.stride()
    # stride_out_l, stride_out_K, stride_out_d1 = grad_S2s.stride()
    stride_g_BSIZE, stride_g_d1 = g.shape[1], 1
    stride_su_l, stride_su_K, stride_su_BSIZE = U2s_hin.shape[1] * U2s_hin.shape[2], U2s_hin.shape[2], 1
    stride_out_l, stride_out_K, stride_out_d1 = grad_S2s.shape[1] * grad_S2s.shape[2], grad_S2s.shape[2], 1

    BLOCK_SIZE_K, BLOCK_SIZE_d1, BLOCK_SIZE_BSIZE = 128, 256, 64
    GROUP_SIZE_K = 8
    
    grid = lambda META: (L, triton.cdiv(K, META["BLOCK_SIZE_K"]) * triton.cdiv(d1, META["BLOCK_SIZE_d1"]), )
    
    calc_grad_S2s_kernel[grid](
        g, U2s_hin, grad_S2s,
        K, BSIZE, d1, L,
        stride_g_BSIZE, stride_g_d1,
        stride_su_l, stride_su_K, stride_su_BSIZE,
        stride_out_l, stride_out_K, stride_out_d1
    )
    
    return grad_S2s