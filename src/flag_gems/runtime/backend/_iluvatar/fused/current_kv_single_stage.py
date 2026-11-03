# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Fixed task024 current-K/V reader; caller owns no-history qualification.

Original persistent cache writer is retained. Arithmetic, masks, packed grid,
Q8/KV32, four warps and one stage match the selected single-stage parent.
No packing, cache writes, host tensor reads, or new decode domain here.
"""
import triton
import triton.language as tl


@triton.jit
def _current_kv_single_stage(
    Q,
    K,
    V,
    OUT,
    CU,
    LEN,
    scale,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    os0: tl.constexpr,
    os1: tl.constexpr,
    NSEQ: tl.constexpr,
    BQ: tl.constexpr,
    BN: tl.constexpr,
):
    bid = tl.program_id(0)
    kh = tl.program_id(1)
    # Packed upper-bound grid: sequence s starts at floor(cu[s]/BQ)+s.
    # Extra programs return without touching output; no CPU length extraction.
    lo = 0
    hi = NSEQ
    while lo < hi:
        mid = (lo + hi) // 2
        start = tl.load(CU + mid) // BQ + mid
        if start <= bid:
            lo = mid + 1
        else:
            hi = mid
    seq = lo - 1
    qstart = tl.load(CU + seq)
    qlen = tl.load(CU + seq + 1) - qstart
    qb = bid - (qstart // BQ + seq)
    if qb * BQ >= qlen:
        return
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
        # Guarded no-history call: the complete causal K/V range is in
        # this request's current token interval. Keep parent masks/order.
        token = (qstart + pos).to(tl.int64)
        k = tl.load(
            K + token[None, :] * ks0 + kh * ks1 + d[:, None],
            pos[None, :] < kvlen,
            0,
        )
        v = tl.load(
            V + token[:, None] * vs0 + kh * vs1 + d[None, :],
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


def current_kv_attention(q, k, v, out, cu_seqlens_q, seqused_k, softmax_scale):
    if not q.shape[0] or not seqused_k.shape[0]:
        return None
    return _current_kv_single_stage[(q.shape[0] // 8 + seqused_k.shape[0], 2)](
        q, k, v, out, cu_seqlens_q, seqused_k, softmax_scale,
        *q.stride()[:2], *k.stride()[:2], *v.stride()[:2],
        *out.stride()[:2], seqused_k.shape[0], 8, 32,
        num_warps=4, num_stages=1,
    )
