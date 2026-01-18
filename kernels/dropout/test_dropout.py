import tabulate
import torch

from .dropout import DEVICE, seeded_dropout


def test_seeded_dropout():
    x = torch.randn(size=(10, ), device=DEVICE)
    # Compare this to the baseline - dropout mask is never instantiated!
    output = seeded_dropout(x, p=0.5, seed=123)
    output2 = seeded_dropout(x, p=0.5, seed=123)
    output3 = seeded_dropout(x, p=0.5, seed=512)
    output4 = seeded_dropout(x, p=0.5, seed=512)

    assert torch.allclose(output, output2)
    assert torch.allclose(output3, output4)
    

    print(
    tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output.tolist(),
        ["output (seed = 123)"] + output2.tolist(),
        ["output (seed = 512)"] + output3.tolist(),
        ["output (seed = 512)"] + output4.tolist(),
    ]))


