# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unvalidated task009 scalar-page prototype for BF16 paged causal GQA (16 Q / 2 KV heads, D=128).

The caller owns validation and fallback. No KV writes, temporary tensors, host
metadata reads, or decode dispatch are performed here. Each program shares a
K/V tile across all eight heads and 2, 4 or 8 consecutive query positions.
"""

import triton
import triton.language as tl


@triton.jit
def _mapped_paged_gqa_single_stage(
    Q,
    K,
    V,
    OUT,
    CU,
    LEN,
    BT,
    MAP,
    scale,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ks2: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    vs2: tl.constexpr,
    os0: tl.constexpr,
    os1: tl.constexpr,
    bts: tl.constexpr,
    NSEQ: tl.constexpr,
    NMAP: tl.constexpr,
    PAGE: tl.constexpr,
    BQ: tl.constexpr,
    BN: tl.constexpr,
):
    bid = tl.program_id(0)
    kh = tl.program_id(1)
    seq = tl.load(MAP + bid * 2, bid < NMAP, 0)
    qb = tl.load(MAP + bid * 2 + 1, bid < NMAP, 0)
    if bid >= NMAP or seq < 0 or seq >= NSEQ:
        return
    qstart = tl.load(CU + seq)
    qlen = tl.load(CU + seq + 1) - qstart
    kvlen = tl.load(LEN + seq)
    r = tl.arange(0, BQ * 8)
    d = tl.arange(0, 128)
    n = tl.arange(0, BN)
    qp = qb * BQ + r // 8
    head = kh * 8 + r % 8
    q = tl.load(
        Q + (qstart + qp[:, None]).to(tl.int64) * qs0 + head[:, None] * qs1 + d[None, :],
        qp[:, None] < qlen,
        0,
    )
    m = tl.full((BQ * 8,), float("-inf"), tl.float32)
    z = tl.zeros((BQ * 8,), tl.float32)
    acc = tl.zeros((BQ * 8, 128), tl.float32)
    # Skip future tiles even for prefix/chunk prefill, using bottom-right causal.
    end = tl.minimum(kvlen, kvlen - qlen + (qb + 1) * BQ)
    for base in range(0, tl.maximum(end, 0), BN):
        pos = base + n
        # PAGE=16, BN=32 and base advances by 32: exactly two page IDs.
        # Preserve masked lanes too; never load the second page past kvlen.
        page0 = tl.load(BT + seq * bts + base // PAGE, base < kvlen, 0).to(tl.int64)
        page1 = tl.load(BT + seq * bts + base // PAGE + 1, base + 16 < kvlen, 0).to(tl.int64)
        page = tl.where(n < 16, page0, page1)
        page = tl.where(pos < kvlen, page, 0)
        slot = pos % PAGE
        k = tl.load(
            K + page[None, :] * ks0 + slot[None, :] * ks1 + kh * ks2 + d[:, None],
            pos[None, :] < kvlen,
            0,
        )
        v = tl.load(
            V + page[:, None] * vs0 + slot[:, None] * vs1 + kh * vs2 + d[None, :],
            pos[:, None] < kvlen,
            0,
        )
        scores = tl.dot(q, k) * scale
        valid = (pos[None, :] < kvlen) & (pos[None, :] <= (kvlen - qlen + qp[:, None]))
        scores = tl.where(valid, scores, float("-inf"))
        new_m = tl.maximum(m, tl.max(scores, 1))
        safe_m = tl.where(new_m == float("-inf"), 0.0, new_m)
        alpha = tl.exp(m - safe_m)
        p = tl.exp(scores - safe_m[:, None])
        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        z = z * alpha + tl.sum(p, 1)
        m = new_m
    result = acc / tl.where(z > 0, z, 1.0)[:, None]
    tl.store(
        OUT + (qstart + qp[:, None]).to(tl.int64) * os0 + head[:, None] * os1 + d[None, :],
        result,
        qp[:, None] < qlen,
    )


def mapped_paged_gqa_attention(q, k, v, out, cu_seqlens_q, seqused_k,
                              block_table, softmax_scale, program_map):
    """Caller validates exact current-call map and parent supported domain."""
    return _mapped_paged_gqa_single_stage[(program_map.shape[0], 2)](
        q, k, v, out, cu_seqlens_q, seqused_k, block_table, program_map,
        softmax_scale, *q.stride()[:2], *k.stride()[:3], *v.stride()[:3],
        *out.stride()[:2], block_table.stride(0), seqused_k.shape[0],
        program_map.shape[0], 16, 8, 32, num_warps=4, num_stages=1)
