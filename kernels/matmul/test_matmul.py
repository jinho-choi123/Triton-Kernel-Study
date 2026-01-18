import pytest
import torch

from .matmul import DEVICE, matmul


@pytest.mark.parametrize("M, K, N", [(512, 512, 512), (1024, 1024, 1024), (512, 128, 128), (128, 512, 128)])
def test_matmul_fp16(M, K, N):
    torch.manual_seed(0)
    a = torch.rand((M, K), device=DEVICE, dtype=torch.float16) - 0.5
    b = torch.rand((K, N), device=DEVICE, dtype=torch.float16) - 0.5
    output_torch = torch.matmul(a, b)
    output_triton = matmul(a, b)
    if torch.allclose(output_torch, output_triton):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

