import torch

from .vector_add import DEVICE, vector_add


def test_vector_add():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    output_torch = x + y
    output_triton = vector_add(x,y)

    if torch.allclose(output_torch, output_triton):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
