# SPDX-License-Identifier: Apache-2.0
"""W009 component prototype: gate/up GEMMs with a BF16 SwiGLU epilogue.

Not exported or registered as an ATen implementation. Two bounded configurations
are for P1 selection only; device correctness and complete FFN costs are required.
"""
import torch
import triton
import triton.language as tl


CONFIGS = ((16, 64, 32), (32, 64, 32))


@triton.jit
def _gate_up(X, W, Y, Boundary, M: tl.constexpr, I: tl.constexpr, K: tl.constexpr,
             XM: tl.constexpr, XK: tl.constexpr, WN: tl.constexpr, WK: tl.constexpr,
             ROUND_SILU: tl.constexpr, DEBUG_BOUNDARY: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr,
             BK: tl.constexpr):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + kk
        x = tl.load(X + m[:, None] * XM + k[None, :] * XK,
                    (m[:, None] < M) & (k[None, :] < K), other=0)
        g = tl.load(W + n[None, :] * WN + k[:, None] * WK,
                    (n[None, :] < I) & (k[:, None] < K), other=0)
        u = tl.load(W + (I + n[None, :]) * WN + k[:, None] * WK,
                    (n[None, :] < I) & (k[:, None] < K), other=0)
        gate += tl.dot(x, g, allow_tf32=False)
        up += tl.dot(x, u, allow_tf32=False)
    # Preserve the original gate/up projection's BF16 storage boundary.
    g = gate.to(tl.bfloat16).to(tl.float32)
    u = up.to(tl.bfloat16).to(tl.float32)
    if DEBUG_BOUNDARY:
        mask = (m[:, None] < M) & (n[None, :] < I)
        tl.store(Boundary + m[:, None] * (2 * I) + n[None, :], g.to(tl.bfloat16), mask)
        tl.store(Boundary + m[:, None] * (2 * I) + I + n[None, :], u.to(tl.bfloat16), mask)
    silu = g / (1.0 + tl.exp(-g))
    # Vendor torch SiLU stores BF16 before mul; FlagGems fuses the two.
    if ROUND_SILU:
        silu = silu.to(tl.bfloat16).to(tl.float32)
    result = (silu * u).to(tl.bfloat16)
    tl.store(Y + m[:, None] * I + n[None, :], result,
             (m[:, None] < M) & (n[None, :] < I))


def gate_up_swiglu(x, weight, *, config=0, round_silu=False, boundary_out=None):
    """Return the SwiGLU intermediate; caller must include down GEMM in costs."""
    if config not in (0, 1):
        raise ValueError("W009 F permits exactly two P1 configurations")
    if (x.ndim != 2 or weight.ndim != 2 or not x.is_cuda
            or x.device != weight.device or x.dtype != torch.bfloat16
            or weight.dtype != torch.bfloat16 or weight.shape[1] != x.shape[1]
            or weight.shape[0] % 2 or x.stride(-1) != 1
            or weight.stride(-1) != 1 or x.stride(0) < x.shape[1]
            or weight.stride(0) < weight.shape[1]):
        raise ValueError("Unsupported W009 fused gate/up layout")
    m, k = x.shape
    i = weight.shape[0] // 2
    if boundary_out is not None and (
            tuple(boundary_out.shape) != (m, 2 * i) or boundary_out.dtype != x.dtype
            or boundary_out.device != x.device or not boundary_out.is_contiguous()
            or boundary_out.data_ptr() in (x.data_ptr(), weight.data_ptr())):
        raise ValueError("Invalid W009 audit-only boundary output")
    out = torch.empty((m, i), device=x.device, dtype=x.dtype)
    if m == 0:
        return out
    bm, bn, bk = CONFIGS[config]
    _gate_up[(triton.cdiv(m, bm), triton.cdiv(i, bn))](
        x, weight, out, boundary_out if boundary_out is not None else out,
        m, i, k, *x.stride(), *weight.stride(),
        bool(round_silu), boundary_out is not None, bm, bn, bk, num_warps=4, num_stages=2,
    )
    return out
