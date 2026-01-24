import torch
import triton
import triton.language as tl
from loguru import logger
from triton.runtime import driver
import math

DEVICE = triton.runtime.driver.active.get_active_torch_device()

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()


def torch_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the row-wise softmax with pytorch.
    """
    x_row_max = torch.max(x, dim=-1).values

    # subtract the max value from x
    z = x - x_row_max[:, None]

    # calculate the exponent
    numerator = torch.exp(z)

    # Calculate the denominator
    denominator = torch.sum(numerator, dim=-1, keepdim=True)

    # divide the numerator by the denominator
    return numerator / denominator


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_strides,
    output_strides,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    # get the row index that is going to be processed by this program.
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    # Iterate over the rows
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # calculate the row starting pointer
        # Assumption: the input is 2D matrix
        input_row_start_ptr = input_ptr + row_idx * input_strides[0]

        # Calculate the column offset.
        # It is important to multiply with input_strides[1] because we can not assure it is 1.
        input_col_offsets = tl.arange(0, BLOCK_SIZE) * input_strides[1]

        input_row_ptrs = input_row_start_ptr + input_col_offsets

        input_mask = input_col_offsets < (n_cols * input_strides[1])

        # Load the data to SRAM
        row = tl.load(input_row_ptrs, mask=input_mask, other=-float("inf"))

        # Subtract maximum for preventing overflow
        row_max = tl.max(row, axis=0)

        row_minus_max = row - row_max
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        # Store the result to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_strides[0]
        output_col_offsets = tl.arange(0, BLOCK_SIZE) * output_strides[1]
        output_row_ptrs = output_row_start_ptr + output_col_offsets

        output_mask = output_col_offsets < (n_cols * output_strides[1])

        tl.store(output_row_ptrs, softmax_output, mask=output_mask)


def softmax(x: torch.Tensor):
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8

    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # allocate output
    y = torch.empty_like(x)

    # pre-compile the kernel to get register usage and compute thread occupancy.
    kernel = softmax_kernel.warmup(
        x,
        y,
        x.stride(),
        y.stride(),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_stages,
        num_warps=num_warps,
        grid=(1,),
    )

    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared

    # compute thread occupancy
    register_occupancy = (n_regs * WARP_SIZE * num_warps) / NUM_REGS

    smem_occupancy = size_smem / SIZE_SMEM

    occupancy = min(register_occupancy, smem_occupancy)

    num_programs = NUM_SM * math.floor(1 / occupancy)

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs
    kernel[num_programs, 1, 1](
        x,
        y,
        x.stride(),
        y.stride(),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        num_stages,
    )

    return y
