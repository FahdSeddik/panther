import torch
import triton
import triton.language as tl

# def getConfigs(names, ranges, num_stages_range, num_warps_range):
#     configs = [
#         triton.Config({f'{names[0]}': x0, f'{names[1]}': x1, f'{names[2]}': x2, f'{names[3]}': x3}, num_stages=s, num_warps=w) \
#         for x0 in ranges[0]\
#         for x1 in ranges[1]\
#         for x2 in ranges[2]\
#         for x3 in ranges[3]\
#         for s in num_stages_range\
#         for w in num_warps_range\
#     ]  
#     return configs

# ranges = [[32, 64, 128, 256, 512, 1024], [32, 64, 128, 256, 512, 1024], [32, 64, 128, 256, 512, 1024], [2,4,8]]
# num_stages_range = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# num_warps_range = [2, 4, 8, 16, 32]

@triton.autotune(
    configs=[
    triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_D2': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=1, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_D2': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_D2': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_D2': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_D2': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_D2': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_D2': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_BSIZE': 32, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_D2': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_D2': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_BSIZE': 256, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_D2': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_BSIZE': 256, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_D2': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_D2': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_D2': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_D2': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_D2': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_D2': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4)
],
    # configs=getConfigs(['BLOCK_SIZE_BSIZE', 'BLOCK_SIZE_K', 'BLOCK_SIZE_D2', 'GROUP_SIZE_BSIZE'], ranges, num_stages_range, num_warps_range),
    key=['BSIZE', 'K', 'd2', 'L'],
)
@triton.jit
def first_pass_kernel(
        hin_ptr, S1s_ptr, U2s_ptr, out1_ptr, out2_ptr,
        BSIZE, K, d2, L,
        stride_hin_bsize, stride_hin_d2,
        stride_su_l, stride_su_d2, stride_su_k,
        stride_out_l, stride_out_bsize, stride_out_k,
        BLOCK_SIZE_BSIZE: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_D2: tl.constexpr,
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
    offs_d2 = tl.arange(0, BLOCK_SIZE_D2)

    offs_bsize = tl.max_contiguous(tl.multiple_of(offs_bsize, BLOCK_SIZE_BSIZE), BLOCK_SIZE_BSIZE)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_SIZE_K), BLOCK_SIZE_K)
    offs_d2 = tl.max_contiguous(tl.multiple_of(offs_d2, BLOCK_SIZE_D2), BLOCK_SIZE_D2)
    
    hin_ptrs = hin_ptr + (offs_bsize[:, None] * stride_hin_bsize + offs_d2[None, :] * stride_hin_d2)

    su_tmp = batch_id * stride_su_l + (offs_d2[:, None] * stride_su_d2 + offs_k[None, :] * stride_su_k)
    S1s_ptrs = S1s_ptr + su_tmp
    U2s_ptrs = U2s_ptr + su_tmp

    accumulator1 = tl.full(shape=(BLOCK_SIZE_BSIZE, BLOCK_SIZE_K), value=0.0, dtype=tl.float32)
    accumulator2 = tl.full(shape=(BLOCK_SIZE_BSIZE, BLOCK_SIZE_K), value=0.0, dtype=tl.float32)
    
    for d2_i in range(0, tl.cdiv(d2, BLOCK_SIZE_D2)):
        hin_mask = (offs_bsize[:, None] < BSIZE) & (offs_d2[None, :] < d2 - d2_i * BLOCK_SIZE_D2)
        hin = tl.load(hin_ptrs, mask=hin_mask, other=0.0)
        
        su_mask = (offs_d2[:, None] < d2 - d2_i * BLOCK_SIZE_D2) & (offs_k[None, :] < K)
        S1s = tl.load(S1s_ptrs, mask=su_mask, other=0.0)
        U2s = tl.load(U2s_ptrs, mask=su_mask, other=0.0)
        
        accumulator1 += tl.dot(hin, S1s, input_precision="ieee")
        accumulator2 += tl.dot(hin, U2s, input_precision="ieee")
        
        hin_ptrs += BLOCK_SIZE_D2 * stride_hin_d2
        S1s_ptrs += BLOCK_SIZE_D2 * stride_su_d2
        U2s_ptrs += BLOCK_SIZE_D2 * stride_su_d2

    out_tmp = batch_id * stride_out_l + stride_out_bsize * offs_bsize[:, None] + stride_out_k * offs_k[None, :]
    out1_ptrs = out1_ptr + out_tmp
    out2_ptrs = out2_ptr + out_tmp
    
    out_mask = (offs_bsize[:, None] < BSIZE) & (offs_k[None, :] < K)
    
    tl.store(out1_ptrs, accumulator1, mask=out_mask)
    tl.store(out2_ptrs, accumulator2, mask=out_mask)

def first_pass(hin, S1s, U2s):
    device = 'cuda'
    # assert hin.shape[1] == S1s.shape[1], "Incompatible dimensions"
    # assert hin.shape[1] == U2s.shape[1], "Incompatible dimensions"
    # assert hin.is_contiguous(), "Matrix A must be contiguous"
    # assert S1s.is_contiguous(), "Matrix A must be contiguous"
    # assert U2s.is_contiguous(), "Matrix A must be contiguous"
    # assert S1s.stride() == U2s.stride(), "Matrix A must be contiguous"
    
    BSIZE, d2 = hin.shape
    L, _, K = S1s.shape
    
    out1 = torch.empty((L, BSIZE, K), dtype=torch.float32, device=device)
    out2 = torch.empty((L, BSIZE, K), dtype=torch.float32, device=device)

    # stride_hin_bsize, stride_hin_d2 = hin.stride()
    # stride_su_l, stride_su_d2, stride_su_k = S1s.stride()
    # stride_out_l, stride_out_bsize, stride_out_k = out1.stride()
    stride_hin_bsize, stride_hin_d2 = hin.shape[1] , 1
    stride_su_l, stride_su_d2, stride_su_k = S1s.shape[1] * S1s.shape[2], S1s.shape[2], 1
    stride_out_l, stride_out_bsize, stride_out_k = out1.shape[1] * out1.shape[2], out1.shape[2], 1
    
    # assert out1.stride() == out2.stride(), "Matrix A must be contiguous"
    
    grid = lambda META: (L, triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]), )
    
    first_pass_kernel[grid](
        hin, S1s, U2s, out1, out2,
        BSIZE, K, d2, L,
        stride_hin_bsize, stride_hin_d2,
        stride_su_l, stride_su_d2, stride_su_k,
        stride_out_l, stride_out_bsize, stride_out_k,
    )
    
    return out1, out2
  
@triton.autotune(
    configs = [
    triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_D1': 64, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_BSIZE': 8}, num_stages=1, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_D1': 32, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_D1': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_D1': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_D1': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_D1': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_D1': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_BSIZE': 32, 'BLOCK_SIZE_D1': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_D1': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_BSIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_BSIZE': 256, 'BLOCK_SIZE_D1': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_BSIZE': 256, 'BLOCK_SIZE_D1': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_D1': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_D1': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_D1': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_D1': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_D1': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4, num_warps=4)
],
    key=['BSIZE', 'd1', 'K', 'L'],
)
@triton.jit
def second_pass_kernel(
        in1_ptr, in2_ptr, U1s_ptr, S2s_ptr, bias_ptr, out_ptr,
        BSIZE, d1, K, L,
        stride_in12_l, stride_in12_bsize, stride_in12_k,
        stride_us_l, stride_us_k, stride_us_d1,
        stride_bias_bsize, stride_bias_d1,
        stride_out_bsize, stride_out_d1,
        BLOCK_SIZE_BSIZE: tl.constexpr, BLOCK_SIZE_D1: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_BSIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    num_pid_bsize = tl.cdiv(BSIZE, BLOCK_SIZE_BSIZE)
    num_pid_d1 = tl.cdiv(d1, BLOCK_SIZE_D1)
    num_pid_in_group = GROUP_SIZE_BSIZE * num_pid_d1
    group_id = pid // num_pid_in_group
    first_pid_bsize = group_id * GROUP_SIZE_BSIZE
    GROUP_SIZE_BSIZE = min(num_pid_bsize - first_pid_bsize, GROUP_SIZE_BSIZE)
    pid_bsize = first_pid_bsize + ((pid % num_pid_in_group) % GROUP_SIZE_BSIZE)
    pid_d1 = (pid % num_pid_in_group) // GROUP_SIZE_BSIZE

    offs_bsize = pid_bsize * BLOCK_SIZE_BSIZE + tl.arange(0, BLOCK_SIZE_BSIZE)
    offs_d1 = pid_d1 *  BLOCK_SIZE_D1 + tl.arange(0, BLOCK_SIZE_D1)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    offs_bsize = tl.max_contiguous(tl.multiple_of(offs_bsize, BLOCK_SIZE_BSIZE), BLOCK_SIZE_BSIZE)
    offs_d1 = tl.max_contiguous(tl.multiple_of(offs_d1, BLOCK_SIZE_D1), BLOCK_SIZE_D1)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_SIZE_K), BLOCK_SIZE_K)

    in_tmp = offs_bsize[:, None] * stride_in12_bsize + offs_k[None, :] * stride_in12_k
    us_tmp = offs_k[:, None] * stride_us_k + offs_d1[None, :] * stride_us_d1

    accumulator = tl.full(shape=(BLOCK_SIZE_BSIZE, BLOCK_SIZE_D1), value=0.0, dtype=tl.float32)
    
    for l in range(0, L):
        l_tmp_stride = l * stride_in12_l
        
        in1_ptrs = l_tmp_stride + in1_ptr + in_tmp
        in2_ptrs = l_tmp_stride + in2_ptr + in_tmp

        U1s_ptrs = l_tmp_stride + U1s_ptr + us_tmp
        S2s_ptrs = l_tmp_stride + S2s_ptr + us_tmp
        
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            in_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
            in1 = tl.load(in1_ptrs, mask=in_mask, other=0.0)
            in2 = tl.load(in2_ptrs, mask=in_mask, other=0.0)
            
            us_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            U1s = tl.load(U1s_ptrs, mask=us_mask, other=0.0)
            S2s = tl.load(S2s_ptrs, mask=us_mask, other=0.0)
            
            accumulator += tl.dot(in1, U1s, input_precision="ieee")
            accumulator += tl.dot(in2, S2s, input_precision="ieee")

            in_inc = BLOCK_SIZE_K * stride_in12_k
            in1_ptrs += in_inc
            in2_ptrs += in_inc
            
            us_inc = BLOCK_SIZE_K * stride_us_k
            U1s_ptrs += us_inc
            S2s_ptrs += us_inc
    
    bias_ptrs = bias_ptr + offs_d1[None, :] * stride_bias_d1
    bias_mask = (offs_d1[None, :] < d1)
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0)

    accumulator *= (1.0/ (2.0 * L))
    accumulator += bias

    out_ptrs = out_ptr + stride_out_bsize * offs_bsize[:, None] + stride_out_d1 * offs_d1[None, :]
    out_mask = (offs_bsize[:, None] < BSIZE) & (offs_d1[None, :] < d1)
    
    tl.store(out_ptrs, accumulator, mask=out_mask)

def second_pass(in1, in2, U1s, S2s, bias):
    # assert in1.shape[2] == U1s.shape[1], "Incompatible dimensions"
    # assert in2.shape[2] == S2s.shape[1], "Incompatible dimensions"
    # assert in1.is_contiguous(), "Matrix A must be contiguous"
    # assert in2.is_contiguous(), "Matrix A must be contiguous"
    # assert U1s.is_contiguous(), "Matrix A must be contiguous"
    # assert S2s.is_contiguous(), "Matrix A must be contiguous"
    # assert bias.is_contiguous(), "Matrix A must be contiguous"
    # assert U1s.stride() == S2s.stride(), "Matrix A must be contiguous"
    # assert in1.stride() == in2.stride(), "Matrix A must be contiguous"
    
    L, BSIZE, K = in1.shape
    _, _, d1 = U1s.shape
    
    out = torch.empty((BSIZE, d1), dtype=torch.float32, device=device)

    # stride_in12_l, stride_in12_bsize, stride_in12_k = in1.stride()
    # stride_us_l, stride_us_k, stride_us_d1 = U1s.stride()
    # stride_bias_bsize, stride_bias_d1 = bias.stride()
    # stride_out_bsize, stride_out_d1 = out.stride()
    stride_in12_l, stride_in12_bsize, stride_in12_k = in1.shape[1] * in1.shape[2], in1.shape[2], 1
    stride_us_l, stride_us_k, stride_us_d1 = U1s.shape[1] * U1s.shape[2], U1s.shape[2], 1
    stride_bias_bsize, stride_bias_d1 = bias.shape[1], 1
    stride_out_bsize, stride_out_d1 = out.shape[1], 1
    
    grid = lambda META: (triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"]) * triton.cdiv(d1, META["BLOCK_SIZE_D1"]), )
    
    second_pass_kernel[grid](
        in1, in2, U1s, S2s, bias, out,
        BSIZE, d1, K, L,
        stride_in12_l, stride_in12_bsize, stride_in12_k,
        stride_us_l, stride_us_k, stride_us_d1,
        stride_bias_bsize, stride_bias_d1,
        stride_out_bsize, stride_out_d1,
    )
    
    return out