import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def seeded_dropout_kernel(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(axis=0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask)

    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p

    # write back
    output = tl.where(x_keep, x / (1 - p), 0.0)

    tl.store(output_ptr + offsets, output, mask)

def seeded_dropout(x: torch.Tensor, p: float, seed: int):
    assert 0 <= p <= 1, "dropout probability must be between 0 and 1"
    assert x.is_contiguous(), "input must be contiguous"

    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    output = torch.empty_like(x)

    seeded_dropout_kernel[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output
