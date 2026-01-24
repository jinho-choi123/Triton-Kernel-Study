import pytest
import torch
from .softmax import DEVICE, softmax, torch_softmax


@pytest.mark.parametrize(
    "M, N", [(4, 4), (4, 64), (64, 4), (64, 64), (128, 128), (1024, 1024)]
)
def test_torch_softmax(M, N):
    torch.manual_seed(0)

    a = torch.rand((M, N), device=DEVICE, dtype=torch.float32)

    output_torch = torch_softmax(a)
    output_golden = torch.softmax(a, dim=-1)

    assert torch.allclose(output_torch, output_golden), (
        "result of torch_softmax and torch.softmax doesn't match...\n"
        f"torch_softmax: {output_torch}\n"
        f"torch.softmax: {output_golden}"
    )


@pytest.mark.parametrize(
    "M, N", [(4, 4), (4, 64), (64, 4), (64, 64), (128, 128), (1024, 1024)]
)
def test_triton_softmax(M, N):
    torch.manual_seed(0)

    a = torch.rand((M, N), device=DEVICE, dtype=torch.float32)

    output_triton = softmax(a)
    output_golden = torch.softmax(a, dim=-1)

    assert torch.allclose(output_triton, output_golden), (
        "result of torch_softmax and torch.softmax doesn't match...\n"
        f"output_triton: {output_triton}\n"
        f"torch.softmax: {output_golden}"
    )
