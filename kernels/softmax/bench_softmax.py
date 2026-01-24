import triton
import torch
from .softmax import DEVICE, softmax, torch_softmax


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 100)
        ],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            "triton",
            "torch",
            "naive_softmax",
        ],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={"M": 4096},  # values for function arguments not in `x_names` and `y_name`
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == "naive_softmax":
        ms = triton.testing.do_bench(lambda: torch_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)
