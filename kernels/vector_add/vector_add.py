import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask)
    y = tl.load(y_ptr + offsets, mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask)

def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape, "x and y must have the same shape"
    assert x.device == DEVICE
    
    output = torch.empty_like(x)
    n_elements = x.shape[0]
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
