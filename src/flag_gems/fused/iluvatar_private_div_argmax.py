# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Task069 private-only quotient reduction; P/Q read-only, original argmax order."""
import torch
import triton
import triton.language as tl
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_min

@triton.jit
def private_div_argmax_kernel(
    inp,
    out_index,
    q,
    M,
    N,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
):
    pid_m = ext.program_id(0)

    dtype = inp.type.element_ty
    min_value = get_dtype_min(dtype)

    if ONE_TILE_PER_CTA:
        n_offset = tl.arange(0, TILE_N)
        offset = pid_m * N + n_offset
        mask = n_offset < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
        q_vals = tl.load(q + offset, mask=mask, other=1.0)
        inp_vals = inp_vals / q_vals
        inp_vals = tl.where(mask, inp_vals, min_value)
        local_max, local_argmax = tl.max(
            inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
        )
        out_index_ptrs = out_index + pid_m
        tl.store(out_index_ptrs, local_argmax)
    else:
        max_values = min_value
        argmax_values = 0

        loop_time = N // TILE_N
        remainder = N % TILE_N
        for start_n in range(0, loop_time):
            n_offset = start_n * TILE_N + tl.arange(0, TILE_N)
            offset = pid_m * N + n_offset
            inp_ptrs = inp + offset
            inp_vals = tl.load(inp_ptrs)
            q_vals = tl.load(q + offset)
            inp_vals = inp_vals / q_vals
            local_max, local_argmax = tl.max(
                inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
            )
            update = local_max > max_values
            max_values = tl.where(update, local_max, max_values)
            argmax_values = tl.where(
                update, start_n * TILE_N + local_argmax, argmax_values
            )

        if remainder:
            n_offset = loop_time * TILE_N + tl.arange(0, TILE_N)
            offset = pid_m * N + n_offset
            mask = n_offset < N
            inp_ptrs = inp + offset
            inp_vals = tl.load(inp_ptrs, mask=mask, other=min_value)
            q_vals = tl.load(q + offset, mask=mask, other=1.0)
            inp_vals = inp_vals / q_vals
            inp_vals = tl.where(mask, inp_vals, min_value)
            local_max, local_argmax = tl.max(
                inp_vals, 0, return_indices=True, return_indices_tie_break_left=True
            )
            update = local_max > max_values
            max_values = tl.where(update, local_max, max_values)
            argmax_values = tl.where(
                update, loop_time * TILE_N + local_argmax, argmax_values
            )

        out_index_ptrs = out_index + pid_m
        tl.store(out_index_ptrs, argmax_values)

def private_div_argmax(probs, q):
    rows, vocab = probs.shape
    out = torch.empty((rows, 1), dtype=torch.int64, device=probs.device)
    with torch.cuda.device(probs.device):
        private_div_argmax_kernel[(rows, 1, 1)](
            probs, out, q, rows, vocab,
            TILE_N=4096, ONE_TILE_PER_CTA=False, num_warps=16,
        )
    return out.view(-1)
