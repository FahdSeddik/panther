def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_D2': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_D2': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_D2': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_D2': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_D2': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_D2': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_D2': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_BSIZE': 32, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_D2': 32, 'GROUP_SIZE_BSIZE': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_D2': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_BSIZE': 256, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_D2': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_BSIZE': 256, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_D2': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_D2': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_D2': 128, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_D2': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BSIZE': 64, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_D2': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_BSIZE': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_D2': 64, 'GROUP_SIZE_BSIZE': 8}, num_stages=4,
                      num_warps=4)
    ]

@triton.autotune(
    configs=get_cuda_autotune_config(),
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

    hin_ptrs = hin_ptr + (offs_bsize[:, None] * stride_hin_bsize + offs_d2[None, :] * stride_hin_d2)

    su_tmp = batch_id * stride_su_l + (offs_d2[:, None] * stride_su_d2 + offs_k[None, :] * stride_su_k)
    S1s_ptrs = S1s_ptr + su_tmp
    U2s_ptrs = U2s_ptr + su_tmp

    accumulator1 = tl.full(shape=(BLOCK_SIZE_BSIZE, BLOCK_SIZE_K), value=0.0, dtype=tl.float16)
    accumulator2 = tl.full(shape=(BLOCK_SIZE_BSIZE, BLOCK_SIZE_K), value=0.0, dtype=tl.float16)
    
    for d2_i in range(0, tl.cdiv(d2, BLOCK_SIZE_D2)):
        hin_mask = (offs_bsize[:, None] < BSIZE) & (offs_d2[None, :] < d2 - d2_i * BLOCK_SIZE_D2)
        hin = tl.load(hin_ptrs, mask=hin_mask, other=0.0)
        
        su_mask = (offs_d2[:, None] < d2 - d2_i * BLOCK_SIZE_D2) & (offs_k[None, :] < K)
        S1s = tl.load(S1s_ptrs, mask=su_mask, other=0.0)
        U2s = tl.load(U2s_ptrs, mask=su_mask, other=0.0)
        
        accumulator1 += tl.dot(hin, S1s)
        accumulator2 += tl.dot(hin, U2s)
        
        hin_ptrs += BLOCK_SIZE_D2 * stride_hin_d2
        S1s_ptrs += BLOCK_SIZE_D2 * stride_su_d2
        U2s_ptrs += BLOCK_SIZE_D2 * stride_su_d2

    accumulator1 = accumulator1.to(tl.float16)
    accumulator2 = accumulator2.to(tl.float16)

    out_tmp = batch_id * stride_out_l + stride_out_bsize * offs_bsize[:, None] + stride_out_k * offs_k[None, :]
    out1_ptrs = out1_ptr + out_tmp
    out2_ptrs = out2_ptr + out_tmp
    
    out_mask = (offs_bsize[:, None] < BSIZE) & (offs_k[None, :] < K)
    
    tl.store(out1_ptrs, accumulator1, mask=out_mask)
    tl.store(out2_ptrs, accumulator2, mask=out_mask)

def first_pass(hin, S1s, U2s):
    device = 'cuda'
    assert hin.shape[1] == S1s.shape[1], "Incompatible dimensions"
    assert hin.shape[1] == U2s.shape[1], "Incompatible dimensions"
    assert hin.is_contiguous(), "Matrix A must be contiguous"
    assert S1s.is_contiguous(), "Matrix A must be contiguous"
    assert U2s.is_contiguous(), "Matrix A must be contiguous"
    assert S1s.stride() == U2s.stride(), "Matrix A must be contiguous"
    
    BSIZE, d2 = hin.shape
    L, _, K = S1s.shape
    
    out1 = torch.empty((L, BSIZE, K), dtype=torch.float16, device=device)
    out2 = torch.empty((L, BSIZE, K), dtype=torch.float16, device=device)

    stride_hin_bsize, stride_hin_d2 = hin.stride()
    stride_su_l, stride_su_d2, stride_su_k = S1s.stride()
    stride_out_l, stride_out_bsize, stride_out_k = out1.stride()
    
    assert out1.stride() == out2.stride(), "Matrix A must be contiguous"
    
    grid = lambda META: (L, triton.cdiv(BSIZE, META["BLOCK_SIZE_BSIZE"]) * triton.cdiv(K, META["BLOCK_SIZE_K"]), )
    
    first_pass_kernel[grid](
        hin, S1s, U2s, out1, out2,
        BSIZE, K, d2, L,
        stride_hin_bsize, stride_hin_d2,
        stride_su_l, stride_su_d2, stride_su_k,
        stride_out_l, stride_out_bsize, stride_out_k,
    )
    
    return out1, out2

device = 'cuda'

L = 4
BSIZE, d2 = 128, 16
K = 256

torch.manual_seed(0)
hin = torch.randn((BSIZE, d2), dtype=torch.float16, device=device)
S1s = torch.randn((L, d2, K), dtype=torch.float16, device=device)
U2s = torch.randn((L, d2, K), dtype=torch.float16, device=device)

out1, out2 = first_pass(hin, S1s, U2s)
torch_output1 = (hin.unsqueeze(0).expand(L, BSIZE, d2)).bmm(S1s)
torch_output2 = (hin.unsqueeze(0).expand(L, BSIZE, d2)).bmm(U2s)

rtol = 1e-2
if torch.allclose(out1, torch_output1, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match 1")
else:
    print("❌ Triton and Torch differ 1")

if torch.allclose(out2, torch_output2, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match 2")
else:
    print("❌ Triton and Torch differ 2")