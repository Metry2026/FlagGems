# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Scoped C500 gate/up NT GEMM prototype; enabled only by explicit adapters.

The fixed starting tile is an existing MetaX configuration. There is no config
search, split-K, weight packing, activation fusion, or persistent tensor cache.
"""
import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.device_info import get_sm_count


@triton.jit
def gate_up_nt_persistent_kernel(
    A,
    W,
    C,
    M,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    N: tl.constexpr = 12288
    K: tl.constexpr = 2048
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = grid_m * grid_n
    group_width = GROUP_M * grid_n

    # A bounded CTA set reuses its program to consume disjoint output tiles.
    for tile in range(tl.program_id(0), total_tiles, tl.num_programs(0)):
        group = tile // group_width
        first_m = group * GROUP_M
        group_m = tl.minimum(grid_m - first_m, GROUP_M)
        local_tile = tile - group * group_width
        pid_m = first_m + local_tile % group_m
        pid_n = local_tile // group_m
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Keep ascending K chunks and FP32 accumulation; each CTA owns its C tile.
        for chunk in range(0, tl.cdiv(K, BLOCK_K)):
            ks = chunk * BLOCK_K + offsets_k
            a = tl.load(
                A + rows[:, None] * K + ks[None, :],
                mask=(rows[:, None] < M) & (ks[None, :] < K),
                other=0.0,
            )
            w = tl.load(
                W + cols[None, :] * K + ks[:, None],
                mask=(cols[None, :] < N) & (ks[:, None] < K),
                other=0.0,
            )
            accumulator = tl.dot(a, w, accumulator, out_dtype=tl.float32, allow_tf32=False)
        tl.store(
            C + rows[:, None] * N + cols[None, :],
            accumulator.to(C.dtype.element_ty),
            mask=(rows[:, None] < M) & (cols[None, :] < N),
        )


def eligible_pair(a, weight):
    """Guard is a supported scope, not a measured performance crossover."""
    return (
        runtime.device.vendor_name == "metax"
        and a.layout == torch.strided
        and weight.layout == torch.strided
        and a.ndim == 2
        and weight.ndim == 2
        and a.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and a.device == weight.device
        and a.device.type == "cuda"
        and a.shape[0] >= 128
        and a.shape[1] == 2048
        and weight.shape == (12288, 2048)
        and a.shape[0] * 12288 < 2**31
        and a.is_contiguous()
        and weight.is_contiguous()
    )


def eligible_output(a, weight, output):
    """Conservatively reject any shared input/output storage without GPU reads."""
    if output is None or not isinstance(output, torch.Tensor):
        return False
    if not (
        output.layout == torch.strided
        and output.ndim == 2
        and output.dtype == a.dtype
        and output.device == a.device
        and output.shape == (a.shape[0], 12288)
        and output.stride() == (12288, 1)
    ):
        return False
    try:
        # _overlaps is a storage-metadata predicate, including different-offset
        # views of the same storage. Unknown alias contracts use old fallback.
        return not torch._C._overlaps(output, a) and not torch._C._overlaps(output, weight)
    except (AttributeError, RuntimeError, TypeError):
        return False


def launch_gate_up_nt(a, weight, *, persistent=True, out=None):
    """Explicit kernel fixture entry; production adapters always use persistent.

    persistent=False is the same-tile per-program diagnostic, not a second
    production strategy. Native validation must confirm get_sm_count() matches
    the actual C500 properties (the existing helper has a generic fallback).
    """
    assert eligible_pair(a, weight)
    m = a.shape[0]
    tiles = triton.cdiv(m, 64) * triton.cdiv(12288, 64)
    with torch_device_fn.device(a.device):
        sms = get_sm_count()
        assert sms > 0
        programs = min(sms, tiles) if persistent else tiles
        if out is None:
            output = torch.empty((m, 12288), dtype=torch.bfloat16, device=a.device)
        else:
            assert eligible_output(a, weight, out)
            output = out
        gate_up_nt_persistent_kernel[(programs,)](
            a,
            weight,
            output,
            m,
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=64,
            GROUP_M=8,
            num_warps=4,
            num_stages=2,
            pipeline="basic",
        )
    return output


def try_gate_up_linear(a, weight, bias=None):
    if bias is not None or not eligible_pair(a, weight):
        return None
    return launch_gate_up_nt(a, weight)


def try_gate_up_mm(a, b):
    if b.layout != torch.strided or b.ndim != 2:
        return None
    if b.shape != (2048, 12288) or b.stride() != (1, 2048):
        return None
    weight = b.transpose(0, 1)  # view only: no packing or contiguous copy
    if not eligible_pair(a, weight):
        return None
    return launch_gate_up_nt(a, weight)


def try_gate_up_mm_out(a, b, *, out):
    if b.layout != torch.strided or b.ndim != 2:
        return None
    if b.shape != (2048, 12288) or b.stride() != (1, 2048):
        return None
    weight = b.transpose(0, 1)
    if not eligible_pair(a, weight) or not eligible_output(a, weight, out):
        return None
    return launch_gate_up_nt(a, weight, out=out)
