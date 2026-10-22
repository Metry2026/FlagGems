# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""O4 prototype: paged INT8 KV, per-token-head FP32 scales, BF16 math.

Shares each dequantized K/V tile across eight query heads and BQ positions.
The caller validates format/strides and owns writer/pool lifecycle admission.
No full-cache dequantization, scratch pool, metadata D2H or cache writes.
Component checks and complete-call costs are recorded by source manifest.
This default-off prototype has no full model or engine qualification.
"""
import triton
import triton.language as tl


@triton.jit
def _paged_gqa_int8(
    Q, K, V, KS, VS, OUT, CU, LEN, BT, scale,
    qs0: tl.constexpr, qs1: tl.constexpr,
    ks0: tl.constexpr, ks1: tl.constexpr, ks2: tl.constexpr,
    vs0: tl.constexpr, vs1: tl.constexpr, vs2: tl.constexpr,
    ss0: tl.constexpr, ss1: tl.constexpr, ss2: tl.constexpr,
    ts0: tl.constexpr, ts1: tl.constexpr, ts2: tl.constexpr,
    os0: tl.constexpr, os1: tl.constexpr, bts: tl.constexpr,
    NSEQ: tl.constexpr, PAGE: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr,
):
    bid = tl.program_id(0)
    kh = tl.program_id(1)
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
        Q + (qstart + qp[:, None]).to(tl.int64) * qs0
        + head[:, None] * qs1 + d[None, :],
        qp[:, None] < qlen, 0,
    )
    m = tl.full((BQ * 8,), float("-inf"), tl.float32)
    z = tl.zeros((BQ * 8,), tl.float32)
    acc = tl.zeros((BQ * 8, 128), tl.float32)
    end = tl.minimum(kvlen, kvlen - qlen + (qb + 1) * BQ)
    for base in range(0, tl.maximum(end, 0), BN):
        pos = base + n
        page = tl.load(BT + seq * bts + pos // PAGE, pos < kvlen, 0).to(tl.int64)
        slot = pos % PAGE
        sk = tl.load(KS + page * ss0 + slot * ss1 + kh * ss2, pos < kvlen, 1.0)
        sv = tl.load(VS + page * ts0 + slot * ts1 + kh * ts2, pos < kvlen, 1.0)
        ik = tl.load(
            K + page[None, :] * ks0 + slot[None, :] * ks1 + kh * ks2 + d[:, None],
            pos[None, :] < kvlen, 0,
        )
        iv = tl.load(
            V + page[:, None] * vs0 + slot[:, None] * vs1 + kh * vs2 + d[None, :],
            pos[:, None] < kvlen, 0,
        )
        # Multiply in FP32 before rounding to BF16. K and V scales are distinct;
        # scales are read with their own strides, including inline scale padding.
        k = (ik.to(tl.float32) * sk[None, :]).to(q.dtype)
        v = (iv.to(tl.float32) * sv[:, None]).to(q.dtype)
        scores = tl.dot(q, k) * scale
        valid = (pos[None, :] < kvlen) & (pos[None, :] <= kvlen - qlen + qp[:, None])
        scores = tl.where(valid, scores, float("-inf"))
        new_m = tl.maximum(m, tl.max(scores, 1))
        safe_m = tl.where(new_m == float("-inf"), 0.0, new_m)
        alpha = tl.exp(m - safe_m)
        p = tl.exp(scores - safe_m[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        z = z * alpha + tl.sum(p, 1)
        m = new_m
    result = acc / tl.where(z > 0, z, 1.0)[:, None]
    tl.store(
        OUT + (qstart + qp[:, None]).to(tl.int64) * os0
        + head[:, None] * os1 + d[None, :],
        result, qp[:, None] < qlen,
    )


def paged_gqa_attention_int8(
    q, k, v, k_scale_cache, v_scale_cache, out,
    cu_seqlens_q, seqused_k, block_table, softmax_scale, *, block_q=8, block_n=32,
):
    """Launch one reader call; out is caller-owned, including empty/masked rows.

    Valid CU/length/page metadata is a device-side engine precondition, as in B.
    K/V may include four inline scale bytes after D128; those bytes are not data.
    Wrapper guards must reject unsupported tensors before reaching this entry.
    """
    if block_q not in (2, 4, 8) or block_n not in (32, 64):
        raise ValueError("O4 tiles: Q=2/4/8, KV=32/64")
    if not q.shape[0] or not seqused_k.shape[0]:
        return None
    return _paged_gqa_int8[(q.shape[0] // block_q + seqused_k.shape[0], 2)](
        q, k, v, k_scale_cache, v_scale_cache, out,
        cu_seqlens_q, seqused_k, block_table, softmax_scale,
        *q.stride()[:2], *k.stride()[:3], *v.stride()[:3],
        *k_scale_cache.stride(), *v_scale_cache.stride(), *out.stride()[:2],
        block_table.stride(0), seqused_k.shape[0], k.shape[1], block_q, block_n,
        num_warps=4, num_stages=2,
    )
