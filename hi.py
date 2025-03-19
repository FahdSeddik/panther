import os
os.environ["TRITON_INTERPRET"] = "1"

import torch

import triton
import triton.language as tl

print(torch.__version__)
print(triton.__version__)
print(os.environ["TRITON_INTERPRET"])

DEVICE = "cuda"

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8)
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
        #               num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
        #               num_warps=2),
        # # Good config for fp8 inputs.
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
        #               num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
        #               num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
        #               num_warps=4)
    ]

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def firstMul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K, L : tl.constexpr,
        stride_am, stride_ak,
        stride_bl, stride_bk, stride_bn,
        stride_cl, stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    print("hello")

    for l in range(0, L):
      print("l:", l)
      print("pid_m:", pid_m)
      print("pid_n:", pid_n)
      print("offs_am:", offs_am)
      print("offs_bn:", offs_bn)
      print("offs_k:", offs_k)

      accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
      
      a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
      b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn) + (l * stride_bl)

      for k_i in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
          a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k_i * BLOCK_SIZE_K, other=0.0)
          b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_i * BLOCK_SIZE_K, other=0.0)

          accumulator = tl.dot(a, b, accumulator)

          a_ptrs += BLOCK_SIZE_K * stride_ak
          b_ptrs += BLOCK_SIZE_K * stride_bk

      offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
      offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
      offs_cl = tl.arange(0, L)

      c_ptrs = c_ptr + (stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]) + (l * stride_cl)
      c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

      # c_ptrs = c_ptr + stride_cl * offs_cl[:, None, None] + stride_cm * offs_cm[None, :, None] + stride_cn * offs_cn[None, None, :]
      # c_mask = (offs_cl[:, None, None] > -1) & (offs_cm[None, :, None] < M) & (offs_cn[None, None, :] < N)
      tl.store(c_ptrs, accumulator, mask=c_mask)
      
def firstMul(a, b):
  assert a.shape[1] == b.shape[1], "Incompatible dimensions"
  assert a.is_contiguous(), "Matrix A must be contiguous"
  assert b.is_contiguous(), "Matrix A must be contiguous"

  (M, K), (L, _, N) = a.shape, b.shape  

  c = torch.empty((L, M, N), device='cuda', dtype=torch.float32)

  stride_am, stride_ak = a.stride()
  stride_bl, stride_bk, stride_bn = b.stride()
  stride_cl, stride_cm, stride_cn = c.stride()

  grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]))

  firstMul_kernel[grid](
        a, b, c,
        M, N, K, L,
        stride_am, stride_ak,
        stride_bl, stride_bk, stride_bn,
        stride_cl, stride_cm, stride_cn
  )

  return c

if __name__ == "__main__":
  m = 128 * 4
  n = 192 * 4
  k = 64 * 4
  L = 4

  a = torch.randn(m, k, device='cuda', dtype=torch.float32)
  b = torch.randn(L, k, n, device='cuda', dtype=torch.float32)

  c_torch = (a.unsqueeze(0).expand(L, a.shape[0], a.shape[1])).bmm(b)
  c_triton = firstMul(a, b)

  assert torch.allclose(c_torch, c_triton)
  print("success")

  # print(c_torch)
  # print(c_torch.shape)

  # offs_a = torch.tensor([0, 1, 2])
  # offs_b = torch.tensor([0, 1])
  # offs_c = torch.tensor([0, 1, 2])

  # print(torch.arange(0, 3))

  # offsets = offs_b[:, None] * c_triton.stride(1) + offs_c[None, :] * c_triton.stride(2)
  # flat = c_torch.flatten()
  # print("flattened:", flat)
  # print(flat[offsets + 1 * c_triton.stride(0)])