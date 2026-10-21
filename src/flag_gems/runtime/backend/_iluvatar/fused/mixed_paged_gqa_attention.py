# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""W007 mixed Attention: unchanged B prefill arithmetic + indexed split Decode.

CU is the live request index: classify on device each call, never cache lengths.
No gather/scatter, no expanded KV, no host metadata read. Request slots with
zero queries do not write output; prefill and decode writers are disjoint.
"""
import threading
import torch
import triton
import triton.language as tl

@triton.jit
def _prefill(
    Q,
    K,
    V,
    OUT,
    CU,
    LEN,
    BT,
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
    PAGE: tl.constexpr,
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
    if qlen == 1 or qb * BQ >= qlen:
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
        page = tl.load(BT + seq * bts + pos // PAGE, pos < kvlen, 0).to(tl.int64)
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

@triton.jit
def _split_decode(
    Q,
    K,
    V,
    PART,
    MAX,
    SUM,
    CU,
    LEN,
    BT,
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
    PAGE: tl.constexpr,
    BQ: tl.constexpr,
    J: tl.constexpr,
    BN: tl.constexpr,
):
    seq = tl.program_id(0)
    kh = tl.program_id(1)
    segment = tl.program_id(2)
    qstart = tl.load(CU + seq)
    qlen = tl.load(CU + seq + 1) - qstart
    if qlen != 1:
        return
    qb = 0
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
    span = tl.cdiv(kvlen, J * BN) * BN
    begin = segment * span
    end = tl.minimum(kvlen, begin + span)
    for base in range(begin, end, BN):
        pos = base + n
        page = tl.load(BT + seq * bts + pos // PAGE, pos < kvlen, 0).to(tl.int64)
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
    # Unnormalized FP32 partials, including every empty segment. No stale reads.
    off = (seq.to(tl.int64) * 16 + head) * J + segment
    tl.store(PART + off[:, None] * 128 + d[None, :], acc, r[:, None] < 8)
    tl.store(MAX + off, m, r < 8)
    tl.store(SUM + off, z, r < 8)


@triton.jit
def _merge(PART, MAX, SUM, OUT, CU, os0: tl.constexpr, os1: tl.constexpr, J: tl.constexpr):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    token = tl.load(CU + seq)
    if tl.load(CU + seq + 1) - token != 1:
        return
    j = tl.arange(0, J)
    d = tl.arange(0, 128)
    off = (seq.to(tl.int64) * 16 + head) * J + j
    m = tl.load(MAX + off)
    z = tl.load(SUM + off)
    overall = tl.max(m, 0)
    safe = tl.where(overall == float("-inf"), 0.0, overall)
    weight = tl.exp(m - safe)
    denom = tl.sum(z * weight, 0)
    acc = tl.load(PART + off[:, None] * 128 + d[None, :])
    result = tl.sum(acc * weight[:, None], 0) / tl.where(denom > 0, denom, 1.0)
    tl.store(OUT + token.to(tl.int64) * os0 + head * os1 + d, result)


_local = threading.local()


def _workspace(q, nseq, segments):
    """One arena per host thread/device/current stream/J, reused in stream order.

    No metadata is retained. Growth replaces storage; PyTorch's stream-aware
    allocator preserves queued users. No sharing across streams or graph pools.
    Mixed graph capture is deliberately rejected by the public entry point.
    """
    pools = getattr(_local, "pools", None)
    if pools is None:
        pools = _local.pools = {}
    key = (q.device.index, torch.cuda.current_stream(q.device).cuda_stream, segments)
    arena = pools.get(key)
    if arena is None or arena[0].shape[0] < nseq:
        capacity = triton.next_power_of_2(nseq)
        arena = (
            torch.empty((capacity, 16, segments, 128), device=q.device, dtype=torch.float32),
            torch.empty((capacity, 16, segments), device=q.device, dtype=torch.float32),
            torch.empty((capacity, 16, segments), device=q.device, dtype=torch.float32),
        )
        pools[key] = arena
    return arena


def mixed_paged_gqa_attention(
    q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale,
    *, segments=16, block_q=8, block_n=32,
):
    """Complete eager mixed call, including classification/addressing and merge.

    Legal metadata follows B's contract. The plugin owns shape/semantic guards.
    FULL_DECODE_ONLY never captures this mixed path. For an unexpected capture
    the plugin uses B before allocating scratch; direct capture is an error.
    """
    if segments not in (1, 4, 8, 16) or block_q != 8 or block_n != 32:
        raise ValueError("W007 supports only Q8/KV32 and J=1/4/8/16")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("W007 mixed scratch is eager-only; use B during capture")
    nseq = seqused_k.shape[0]
    if not q.shape[0] or not nseq:
        return None
    part, maxima, sums = _workspace(q, nseq, segments)
    strides = (*q.stride()[:2], *k.stride()[:3], *v.stride()[:3],
               *out.stride()[:2], block_table.stride(0))
    prefill = _prefill[(q.shape[0] // 8 + nseq, 2)](
        q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale,
        *strides, nseq, k.shape[1], 8, 32, num_warps=4, num_stages=2,
    )
    split = _split_decode[(nseq, 2, segments)](
        q, k, v, part, maxima, sums, cu_seqlens_q, seqused_k, block_table, softmax_scale,
        *strides, nseq, k.shape[1], 2, segments, 32, num_warps=4, num_stages=2,
    )
    merge = _merge[(nseq, 16)](
        part, maxima, sums, out, cu_seqlens_q, *out.stride()[:2], segments,
        num_warps=4,
    )
    return prefill, split, merge

