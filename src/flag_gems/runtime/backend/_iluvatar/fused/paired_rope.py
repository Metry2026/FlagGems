# SPDX-License-Identifier: Apache-2.0
"""Functional NeoX pair outputs for the bound task010 Llama RoPE layout.

One K-then-Q combo launch. Each CTA covers the same output indices as the
parent B0=512 configuration, with 256 pairs and the same four 64-lane warps.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def paired_rope_kernel(Q, K, Positions, Table, OutQ, OutK, T,
                       PAIRS: tl.constexpr):
    pid = tl.program_id(0)
    nk = tl.cdiv(T * 128, PAIRS)
    if pid < nk:
        p = pid * PAIRS + tl.arange(0, PAIRS)
        mask = p < T * 128
        d = p % 64
        h = (p // 64) % 2
        t = p // 128
        pos = tl.load(Positions + t, mask, other=0)
        pos = tl.where(pos < 0, pos + 131072, pos)
        x0 = tl.load(K + t * 2560 + h * 128 + d, mask, other=0).to(tl.float32)
        x1 = tl.load(K + t * 2560 + h * 128 + d + 64, mask, other=0).to(tl.float32)
        c = tl.load(Table + pos * 128 + d, mask, other=0).to(tl.float32)
        s = tl.load(Table + pos * 128 + d + 64, mask, other=0).to(tl.float32)
        y0 = x0 * c - x1 * s
        y1 = x1 * c + x0 * s
        o = t * 256 + h * 128 + d
        tl.store(OutK + o, y0, mask)
        tl.store(OutK + o + 64, y1, mask)
    else:
        p = (pid - nk) * PAIRS + tl.arange(0, PAIRS)
        mask = p < T * 1024
        d = p % 64
        h = (p // 64) % 16
        t = p // 1024
        pos = tl.load(Positions + t, mask, other=0)
        pos = tl.where(pos < 0, pos + 131072, pos)
        x0 = tl.load(Q + t * 2560 + h * 128 + d, mask, other=0).to(tl.float32)
        x1 = tl.load(Q + t * 2560 + h * 128 + d + 64, mask, other=0).to(tl.float32)
        c = tl.load(Table + pos * 128 + d, mask, other=0).to(tl.float32)
        s = tl.load(Table + pos * 128 + d + 64, mask, other=0).to(tl.float32)
        y0 = x0 * c - x1 * s
        y1 = x1 * c + x0 * s
        o = t * 2048 + h * 128 + d
        tl.store(OutQ + o, y0, mask)
        tl.store(OutQ + o + 64, y1, mask)


def launch(query, key, positions, table, out_q, out_k):
    t = query.shape[0]
    return paired_rope_kernel[(triton.cdiv(t * 256, 512) + triton.cdiv(t * 2048, 512),)](
        query, key, positions, table, out_q, out_k, t, PAIRS=256,
        num_warps=4, num_stages=1, enable_fp_fusion=True, enable_reflect_ftz=True,
        sanitize_overflow=False)


def paired_rope(query, key, positions, table):
    out_q = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    out_k = torch.empty(key.shape, device=key.device, dtype=key.dtype)
    launch(query, key, positions, table, out_q, out_k)
    return out_q, out_k
