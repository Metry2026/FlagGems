"""O204 compiled LlamaMLP entry; keep the native GEMM selector unchanged.

Only an exact, already selected native SPLIT_K=1 specialization qualifies.
Two independent BF16 roundings precede the recovered FP32 activation chain.
The paired kernel is derived from the separately validated target prototype.
"""
import hashlib
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems import mm_out as _BOUND_MM_OUT

_MM_SHA256 = "4cb0cff3addfa452e85724bc400b39ee354f5c79168f3df63f1b0a6c1a708ae8"
_QUALIFIED = dict(
    BLOCK_M=64, BLOCK_N=256, BLOCK_K=64, SPLIT_K=1, EVEN_K=True,
    UPGRADE=False, UPGRADE_A_OFFS=False, UPGRADE_B_OFFS=False,
    UPGRADE_C_OFFS=False, num_warps=16, num_stages=1, num_ctas=1,
)
_MM_PATH = Path(__file__).resolve().parents[1] / "ops/mm.py"
_BOUND_MM_SOURCE_MATCH = Path(_BOUND_MM_OUT.__code__.co_filename).resolve()==_MM_PATH
_AUDIT_DIR = os.environ.get("VLLM_FL_O204_AUDIT_DIR")
_AUDIT_SEEN = set()


@libentry()
@triton.jit
def paired_gateup_kernel(
    A,B,Z,M,N,K,stride_am,stride_ak,stride_bk,stride_bn,stride_zm,stride_zn,
    acc_dtype:tl.constexpr,input_precision:tl.constexpr,fp8_fast_accum:tl.constexpr,
    BLOCK_M:tl.constexpr,BLOCK_N:tl.constexpr,BLOCK_K:tl.constexpr,
    GROUP_M:tl.constexpr,SPLIT_K:tl.constexpr,EVEN_K:tl.constexpr,
    AB_DTYPE:tl.constexpr,UPGRADE:tl.constexpr,UPGRADE_A_OFFS:tl.constexpr,
    UPGRADE_B_OFFS:tl.constexpr,UPGRADE_C_OFFS:tl.constexpr,HALF:tl.constexpr,
):
    tl.static_assert(SPLIT_K==1)
    tl.static_assert(not UPGRADE and not UPGRADE_A_OFFS and not UPGRADE_B_OFFS and not UPGRADE_C_OFFS)
    pid=tl.program_id(0)
    grid_n=tl.cdiv(HALF,BLOCK_N)
    pid_m=pid//grid_n;pid_n=pid%grid_n
    rm=pid_m*BLOCK_M+tl.arange(0,BLOCK_M)
    rn=pid_n*BLOCK_N+tl.arange(0,BLOCK_N)
    ram=tl.max_contiguous(tl.multiple_of(rm%M,BLOCK_M),BLOCK_M)
    rbn=tl.max_contiguous(tl.multiple_of(rn%HALF,BLOCK_N),BLOCK_N)
    rk=tl.arange(0,BLOCK_K)
    ap=A+ram[:,None]*stride_am+rk[None,:]*stride_ak
    gp=B+rk[:,None]*stride_bk+rbn[None,:]*stride_bn
    up=B+rk[:,None]*stride_bk+(HALF+rbn[None,:])*stride_bn
    gacc=tl.zeros((BLOCK_M,BLOCK_N),dtype=acc_dtype)
    uacc=tl.zeros((BLOCK_M,BLOCK_N),dtype=acc_dtype)
    if EVEN_K:
        for k in range(0,tl.cdiv(K,BLOCK_K*SPLIT_K)):
            a=tl.load(ap);g=tl.load(gp);u=tl.load(up)
            if AB_DTYPE is not None:
                a=a.to(AB_DTYPE);g=g.to(AB_DTYPE);u=u.to(AB_DTYPE)
            if fp8_fast_accum:
                gacc=tl.dot(a,g,gacc,out_dtype=acc_dtype,input_precision=input_precision)
                uacc=tl.dot(a,u,uacc,out_dtype=acc_dtype,input_precision=input_precision)
            else:
                gacc+=tl.dot(a,g,out_dtype=acc_dtype,input_precision=input_precision)
                uacc+=tl.dot(a,u,out_dtype=acc_dtype,input_precision=input_precision)
            ap+=BLOCK_K*SPLIT_K*stride_ak
            gp+=BLOCK_K*SPLIT_K*stride_bk
            up+=BLOCK_K*SPLIT_K*stride_bk
    else:
        loop_num=tl.cdiv(K,BLOCK_K*SPLIT_K)-1
        for k in range(0,loop_num):
            a=tl.load(ap);g=tl.load(gp);u=tl.load(up)
            if AB_DTYPE is not None:
                a=a.to(AB_DTYPE);g=g.to(AB_DTYPE);u=u.to(AB_DTYPE)
            if fp8_fast_accum:
                gacc=tl.dot(a,g,gacc,out_dtype=acc_dtype,input_precision=input_precision)
                uacc=tl.dot(a,u,uacc,out_dtype=acc_dtype,input_precision=input_precision)
            else:
                gacc+=tl.dot(a,g,out_dtype=acc_dtype,input_precision=input_precision)
                uacc+=tl.dot(a,u,out_dtype=acc_dtype,input_precision=input_precision)
            ap+=BLOCK_K*SPLIT_K*stride_ak
            gp+=BLOCK_K*SPLIT_K*stride_bk
            up+=BLOCK_K*SPLIT_K*stride_bk
        remaining=K-loop_num*(BLOCK_K*SPLIT_K)
        a=tl.load(ap,mask=rk[None,:]<remaining,other=0)
        g=tl.load(gp,mask=rk[:,None]<remaining,other=0)
        u=tl.load(up,mask=rk[:,None]<remaining,other=0)
        if fp8_fast_accum:
            gacc=tl.dot(a,g,gacc,out_dtype=acc_dtype,input_precision=input_precision)
            uacc=tl.dot(a,u,uacc,out_dtype=acc_dtype,input_precision=input_precision)
        else:
            gacc+=tl.dot(a,g,out_dtype=acc_dtype,input_precision=input_precision)
            uacc+=tl.dot(a,u,out_dtype=acc_dtype,input_precision=input_precision)
    g=gacc.to(tl.bfloat16).to(tl.float32)
    u=uacc.to(tl.bfloat16).to(tl.float32)
    sigmoid=tl.sigmoid(g)
    silu=g*sigmoid
    result=silu*u
    zp=Z+rm[:,None]*stride_zm+rn[None,:]*stride_zn
    tl.store(zp,result,(rm<M)[:,None] & (rn<HALF)[None,:])


def _pair_call(a,b,z,observed):
    assert observed['SPLIT_K']==1
    assert a.shape[1]==b.shape[0]==2048 and b.shape[1]==12288
    names=('acc_dtype','input_precision','fp8_fast_accum','BLOCK_M','BLOCK_N','BLOCK_K','GROUP_M','SPLIT_K','EVEN_K','AB_DTYPE','UPGRADE','UPGRADE_A_OFFS','UPGRADE_B_OFFS','UPGRADE_C_OFFS','num_warps','num_stages','num_ctas')
    meta={k:v for k,v in observed.items() if k in names}
    # One explained pairing: each accumulator gets half the original column
    # tile, keeping aggregate columns/accumulator elements perCTA unchanged.
    # BM/BK/warps/stages are taken unchanged from the observed original path.
    assert observed['BLOCK_N']>=32 and observed['BLOCK_N']%2==0
    meta['BLOCK_N']=observed['BLOCK_N']//2
    meta.update(acc_dtype=tl.float32,input_precision=None,fp8_fast_accum=True,GROUP_M=8,AB_DTYPE=tl.bfloat16,HALF=6144)
    args = (a,b,z,a.shape[0],12288,2048,*a.stride(),*b.stride(),*z.stride())
    grid = (triton.cdiv(a.shape[0],meta['BLOCK_M'])*triton.cdiv(6144,meta['BLOCK_N']),)
    return args,meta,grid


def launch(a,b,z,observed):
    args,meta,grid = _pair_call(a,b,z,observed)
    return paired_gateup_kernel[grid](*args,**meta)


@libentry()
@triton.jit
def _parent_activation(C,Z,xnumel,XBLOCK:tl.constexpr):
    ix=tl.program_id(0)*XBLOCK+tl.arange(0,XBLOCK)
    mask=ix<xnumel
    j=ix%6144;row=ix//6144
    g=tl.load(C+j+12288*row,mask,other=0).to(tl.float32)
    u=tl.load(C+6144+j+12288*row,mask,other=0).to(tl.float32)
    sig=tl.sigmoid(g)
    silu=g*sig
    value=silu*u
    tl.store(Z+ix,value,mask)


def _existing(entry, args, kwargs):
    """Mirror this LibEntry schema's key calculation, without invoking run."""
    if getattr(entry, "_has_flagtune_tuner", True):
        return None
    try:
        spec=[];dns=[];const=[]
        for i,arg in enumerate(args):
            if arg.__class__.__name__ == "TensorDescriptor":
                return None
            if i in entry.specialize_indices:spec.append(arg)
            elif i in entry.do_not_specialize_indices:dns.append(arg)
            else:const.append(arg)
        for p in entry.jit_function.params[len(args):]:
            if p.name in kwargs:val=kwargs[p.name]
            elif p.default is inspect._empty:continue
            else:val=p.default
            if p.is_constexpr:const.append(val)
            elif p.do_not_specialize:dns.append(val)
            else:spec.append(val)
        key=entry.key(spec,dns,const)
        item=entry.kernel_cache[args[0].device.index].get(key)
        if not isinstance(item,tuple) or len(item)!=5:
            return None
        return dict(key=key,compiled=item[0],constexprs=item[1],hooks=item[4])
    except (AttributeError,IndexError,KeyError,TypeError):
        return None


def _native_entry(mm_out,a,b,z):
    # Source/path verification is immutable for this loaded module. Retain an
    # exact exported-function guard on every call; read its live native cache.
    if mm_out is not _BOUND_MM_OUT:
        return None
    entry=mm_out.__globals__.get("mm_kernel")
    if entry is None or entry.__class__.__name__!="LibEntry":
        return None
    args=(a,b,z,a.shape[0],12288,2048,*a.stride(),*b.stride(),12288,1)
    kwargs=dict(acc_dtype=tl.float32,input_precision=None,fp8_fast_accum=True,
                GROUP_M=8,AB_DTYPE=tl.bfloat16)
    return _existing(entry,args,kwargs)


def _qualified(native):
    return (native is not None and not native["hooks"]
            and native["constexprs"] == _QUALIFIED)


def _activation_args(c,z):
    return (c,z,z.numel()),dict(XBLOCK=1024,num_warps=4)


def _audit(reason,a,native,compiled=None):
    if not _AUDIT_DIR or torch.cuda.is_current_stream_capturing():
        return
    context={}
    try:
        from vllm.forward_context import get_forward_context
        ctx=get_forward_context()
        context=dict(cudagraph_runtime_mode=str(ctx.cudagraph_runtime_mode),
                     attn_metadata_present=ctx.attn_metadata is not None,
                     attn_metadata_nonempty=(len(ctx.attn_metadata)>0
                         if isinstance(ctx.attn_metadata,dict) else None),
                     attn_metadata_type=type(ctx.attn_metadata).__name__)
    except (ImportError,AttributeError,AssertionError):
        pass
    key=(reason,a.shape[0],tuple(context.items()))
    if key in _AUDIT_SEEN:return
    _AUDIT_SEEN.add(key)
    record=dict(utc=datetime.now(timezone.utc).isoformat(),reason=reason,
                M=a.shape[0],N=12288,K=2048,dtype=str(a.dtype),
                stride_a=list(a.stride()),context=context,
                original_constexprs=None if native is None else native["constexprs"],
                original_launch_hooks=False if native is None else bool(native["hooks"]))
    if compiled is not None:
        record.update(paired_compiled_hash=getattr(compiled,"hash",None),
                      paired_metadata=getattr(compiled,"metadata",None)._asdict()
                      if hasattr(getattr(compiled,"metadata",None),"_asdict") else str(getattr(compiled,"metadata",None)))
    dest=Path(_AUDIT_DIR);dest.mkdir(parents=True,exist_ok=True)
    with (dest/f"dispatch-{os.getpid()}.jsonl").open("a") as f:
        f.write(json.dumps(record,default=str)+"\n")


_SOURCE_MATCH = (_BOUND_MM_SOURCE_MATCH
                 and hashlib.sha256(_MM_PATH.read_bytes()).hexdigest()==_MM_SHA256)


def gate_up_swiglu(a,weight):
    """Internal compiled-only API; outer LlamaMLP guard owns unsupported domains.

    A normal native execution of each shape prepares the original full chain.
    Captured execution may only launch already prepared kernels. A missing
    original warmup is reported explicitly, never hidden by tuning in capture.
    """
    import flag_gems

    assert a.is_cuda and a.dtype==torch.bfloat16 and a.ndim==2
    assert 0<a.shape[0]<=2048 and a.shape[1]==2048 and a.is_contiguous()
    assert weight.device==a.device and weight.dtype==a.dtype
    assert tuple(weight.shape)==(12288,2048) and weight.is_contiguous()
    assert not a.requires_grad and not weight.requires_grad
    with torch_device_fn.device(a.device):
        b=weight.t()
        z=torch.empty((a.shape[0],6144),device=a.device,dtype=a.dtype)
        mm_out=flag_gems.mm_out
        native=_native_entry(mm_out,a,b,z)
        # C is represented by an equally aligned BF16 pointer, with the
        # original contiguous C strides supplied explicitly for native keys.
        act_args,act_meta=_activation_args(z,z)
        activation_ready=_existing(_parent_activation,act_args,act_meta) is not None
        capture=torch.cuda.is_current_stream_capturing()
        aligned=all(t.data_ptr()%16==0 for t in (a,b,z))
        eligible=_SOURCE_MATCH and aligned and _qualified(native)
        pair_ready=False
        if eligible and capture:
            pair_args,pair_meta,_=_pair_call(a,b,z,native["constexprs"])
            pair_ready=_existing(paired_gateup_kernel,pair_args,pair_meta) is not None
        # First ordinary execution uses the entire original chain. This is
        # the requested computation, with no synthetic Tuner/cache warming.
        fallback=(not eligible or not activation_ready or (capture and not pair_ready))
        if fallback:
            if capture and (native is None or not activation_ready):
                raise RuntimeError("O204 original full chain was not prepared by native pre-capture execution")
            c=torch.empty((a.shape[0],12288),device=a.device,dtype=a.dtype)
            mm_out(a,b,out=c)
            args,meta=_activation_args(c,z)
            _parent_activation[(triton.cdiv(z.numel(),1024),)](*args,**meta)
            reason=("native_cache_or_config_fallback" if not eligible else
                    "original_activation_first_use" if not activation_ready else
                    "capture_unprepared_pair_fallback")
            _audit(reason,a,native)
        else:
            compiled,_=launch(a,b,z,native["constexprs"])
            _audit("paired",a,native,compiled)
        return z
