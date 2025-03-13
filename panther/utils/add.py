import torch
from torch.utils._triton import has_triton

if not has_triton():
    print("Skipping because triton is not supported on this device.")
else:
    from torch.library import triton_op, wrap_triton

@triton_op("mylib::mysin", mutates_args={})
def mysin(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    wrap_triton(sin_kernel)[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)
    return out

@triton.jit
def sin_kernel(
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = tl.sin(x)
    tl.store(out_ptr + offsets, output, mask=mask)

def sin_triton(x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    sin_kernel[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)
    return out
  
  
x = torch.randn(3, device="cuda")
y = mysin(x)
z = torch.ops.mylib.mysin.default(x)

assert torch.allclose(y, x.sin())
assert torch.allclose(z, x.sin())

y = torch.compile(mysin)(x)
assert torch.allclose(y, x.sin())