from functools import cache
from pathlib import Path
from types import ModuleType

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load

HADAMARD_BLOCK = 1024
GROUP = 128
LANES = 16
WORDS_PER_GROUP = GROUP // LANES
ROW_BLOCK = 32
GEMV_MAX_ROWS = 64


@cache
def _cuda() -> ModuleType:
    return load(
        name="prism_ternary_cuda",
        sources=[str(Path(__file__).with_name("kernels.cu"))],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def block_layout(weight: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out, words = weight.shape
    groups = scales.shape[1]
    if out % ROW_BLOCK:
        raise ValueError(f"output width {out} is not a multiple of {ROW_BLOCK}")
    blocked_w = weight.view(out // ROW_BLOCK, ROW_BLOCK, groups, WORDS_PER_GROUP).permute(0, 2, 1, 3)
    blocked_s = scales.view(out // ROW_BLOCK, ROW_BLOCK, groups).permute(0, 2, 1)
    return blocked_w.contiguous().view(out, words), blocked_s.contiguous().view(out, groups)


@triton.jit
def _dequant_kernel(w_ptr, s_ptr, out_ptr, groups, stride_on, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    g = tl.program_id(1)
    rows = tl.arange(0, BLOCK_N)
    words = tl.arange(0, 8)
    lanes = tl.arange(0, 16)
    w_base = w_ptr + (pid_n * groups + g) * (BLOCK_N * 8)
    w = tl.load(w_base + rows[:, None] * 8 + words[None, :])
    sc = tl.load(s_ptr + (pid_n * groups + g) * BLOCK_N + rows).to(tl.float32)
    codes = ((w[:, :, None] >> (lanes * 2)[None, None, :]) & 3).to(tl.float32) - 1.0
    vals = codes * sc[:, None, None]
    offs_n = pid_n * BLOCK_N + rows
    out_ptrs = out_ptr + offs_n[:, None, None] * stride_on + g * 128 + (words * 16)[None, :, None] + lanes[None, None, :]
    tl.store(out_ptrs, vals.to(out_ptr.dtype.element_ty))


def dequantize(weight: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    out, words = weight.shape
    width = words * LANES
    groups = width // GROUP
    full = torch.empty((out, width), dtype=dtype, device=weight.device)
    _dequant_kernel[(out // ROW_BLOCK, groups)](weight, scales, full, groups, full.stride(0), BLOCK_N=ROW_BLOCK, num_warps=4)
    return full


@torch.library.custom_op("prism_ternary::linear", mutates_args=())
def ternary_linear(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, signs: torch.Tensor
) -> torch.Tensor:
    rotated = _cuda().signed_hadamard(x.reshape(-1, x.shape[-1]), signs)
    y = (
        _cuda().ternary_gemv(rotated, weight, scales)
        if rotated.shape[0] <= GEMV_MAX_ROWS
        else rotated @ dequantize(weight, scales, rotated.dtype).T
    )
    return y.reshape(*x.shape[:-1], weight.shape[0])


@ternary_linear.register_fake
def _ternary_linear_fake(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, signs: torch.Tensor
) -> torch.Tensor:
    return x.new_empty(*x.shape[:-1], weight.shape[0])
