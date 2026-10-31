# SPDX-License-Identifier: Apache-2.0
"""Opt-in fixed-G8 launch using the original LibEntry's already selected config.

No autotuner, new output allocation, tensor-value guard, or custom-op boundary.
"""
import inspect
import os
import torch
import triton
import triton.language as tl

_ENABLED = os.environ.get('VLLM_FL_GROUPED_GATEUP_CTA', '0') == '1'
_MODEL_OK = False
_CACHE = {}
_AUDIT_PATH = os.environ.get('VLLM_FL_GROUPED_GATEUP_CTA_AUDIT_PATH', '')
_AUDIT_DONE = False


def enabled():
    return _ENABLED and _MODEL_OK


def configure_model(model_config, parallel_config):
    global _MODEL_OK
    hf = model_config.hf_config
    _MODEL_OK = (
        parallel_config.tensor_parallel_size == 1
        and getattr(hf, 'architectures', None) == ['LlamaForCausalLM']
        and hf.hidden_size == 2048 and hf.intermediate_size == 6144
        and hf.num_attention_heads == 16 and hf.num_key_value_heads == 2
        and model_config.dtype == torch.bfloat16
    )
    return _MODEL_OK


def eligible(a, b, c, M, N, K):
    return (
        enabled() and N == 12288 and K == 2048 and 0 < M <= 2048
        and a.is_cuda and b.device == c.device == a.device
        and a.dtype == b.dtype == c.dtype == torch.bfloat16
        and a.ndim == b.ndim == c.ndim == 2
        and a.shape == (M, K) and b.shape == (K, N) and c.shape == (M, N)
        and a.stride() == (K, 1) and b.stride() == (1, K)
        and c.stride() == (N, 1)
    )


def qualified(constants, M, N):
    return (constants.get('SPLIT_K') == 1
            and triton.cdiv(M, constants['BLOCK_M']) > 1
            and triton.cdiv(N, constants['BLOCK_N']) > 1)


def lookup(entry, args, kwargs):
    if getattr(entry, '_has_flagtune_tuner', True):
        return None
    spec, dns, const = [], [], []
    for i, value in enumerate(args):
        if i in entry.specialize_indices:spec.append(value)
        elif i in entry.do_not_specialize_indices:dns.append(value)
        else:const.append(value)
    for p in entry.jit_function.params[len(args):]:
        if p.name in kwargs:value = kwargs[p.name]
        elif p.default is inspect._empty:continue
        else:value = p.default
        if p.is_constexpr:const.append(value)
        elif p.do_not_specialize:dns.append(value)
        else:spec.append(value)
    key = entry.key(spec, dns, const)
    item = entry.kernel_cache[args[0].device.index].get(key)
    return None if item is None else (key, item)


def try_launch(entry, grouped_kernel, a, b, c, M, N, K):
    if not eligible(a, b, c, M, N, K):
        return False
    args = (a, b, c, M, N, K, *a.stride(), *b.stride(), *c.stride())
    kwargs = dict(acc_dtype=tl.float32, input_precision=None,
                  fp8_fast_accum=True, GROUP_M=8, AB_DTYPE=tl.bfloat16)
    native = lookup(entry, args, kwargs)
    if native is None:
        return False  # original path may prepare its own normal cache entry
    key, item = native
    parent, constants, _, _, pre_hooks = item
    if pre_hooks or not qualified(constants, M, N):
        return False  # preserve any original side effects and split-K behavior
    cache_key = (a.device.index, key, id(parent))
    cached = _CACHE.get(cache_key)
    gm = triton.cdiv(M, constants['BLOCK_M'])
    gn = triton.cdiv(N, constants['BLOCK_N'])
    if cached is None:
        if torch.cuda.is_current_stream_capturing():
            return False  # no compilation/cache preparation inside capture
        meta = dict(constants)
        meta.update(kwargs)
        for name in ('enable_fp_fusion', 'enable_reflect_ftz', 'sanitize_overflow'):
            meta[name] = getattr(parent.metadata, name)
        with torch.cuda.device(a.device):
            compiled = grouped_kernel[(gm * gn, 1)](*args, **meta)
        # Store separately from original LibEntry/tuner cache. Native callable
        # identity is part of the key, so stale parent entries cannot qualify.
        _CACHE[cache_key] = (compiled, dict(constants))
        _audit(a, b, c, M, N, K, constants, compiled)
        return True
    compiled, saved = cached
    all_args = list(args)
    values = {**kwargs, **saved}
    all_args.extend(values[p.name] for p in grouped_kernel.params[len(args):])
    with torch.cuda.device(a.device):
        compiled[(gm * gn, 1, 1)](*all_args)
    _audit(a, b, c, M, N, K, saved, compiled)
    return True


def _audit(a, b, c, M, N, K, constants, compiled):
    """One requested accuracy-only hit receipt; disabled for normal timing."""
    global _AUDIT_DONE
    if not _AUDIT_PATH or _AUDIT_DONE:
        return
    if torch.cuda.is_current_stream_capturing() or not os.path.exists(_AUDIT_PATH + '.armed'):
        return
    import json
    from pathlib import Path
    from vllm.forward_context import get_forward_context
    context = get_forward_context()
    record = dict(M=M, N=N, K=K, mapping='fixed_G8', model_guard=_MODEL_OK,
                  parent_constants=dict(constants), grouped_hash=compiled.hash,
                  tensors=[dict(shape=list(x.shape), stride=list(x.stride()),
                                offset=x.storage_offset(), dtype=str(x.dtype)) for x in (a,b,c)],
                  context_mode=str(context.cudagraph_runtime_mode),
                  batch_descriptor=str(getattr(context, 'batch_descriptor', None)),
                  armed_after_service_health=True, performance_audit=False)
    Path(_AUDIT_PATH).write_text(json.dumps(record, indent=2, default=str) + '\n')
    _AUDIT_DONE = True
