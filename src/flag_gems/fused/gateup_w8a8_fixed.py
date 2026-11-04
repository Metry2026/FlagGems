"""Task029 unregistered fixed W8A8 prototype; BM/BN/BK32, K2048."""
import triton
import triton.language as tl

@triton.jit
def quantize_rows(X,Q,S,M:tl.constexpr,K:tl.constexpr):
 r=tl.program_id(0);k=tl.arange(0,2048)
 x=tl.load(X+r*K+k).to(tl.float32)
 finite=tl.sum((tl.abs(x)<float('inf')).to(tl.int32),0)==2048
 a=tl.max(tl.abs(x),0)
 s=tl.where(a==0,1.0,tl.div_rn(a,127.0))
 s=tl.where(finite,s,float('nan'))
 z=tl.div_rn(x,s);lo=tl.floor(z);frac=z-lo
 rounded=lo+((frac>0.5)|((frac==0.5)&((lo.to(tl.int32)&1)!=0))).to(tl.float32)
 q=tl.minimum(127.0,tl.maximum(-127.0,rounded)).to(tl.int8)
 # Nonfinite rows are explicitly signalled by NaN scale/output, never silent zero.
 tl.store(Q+r*K+k,tl.where(finite,q,0));tl.store(S+r,s)

@triton.jit
def gateup_w8a8_kernel(A,B,SA,SB,Y,INTEGER,M:tl.constexpr,N:tl.constexpr,DEBUG_INTEGER:tl.constexpr):
 m=tl.program_id(0)*32+tl.arange(0,32)
 n=tl.program_id(1)*32+tl.arange(0,32)
 k=tl.arange(0,32)
 acc=tl.zeros((32,32),tl.int32)
 for base in range(0,2048,32):
  a=tl.load(A+m[:,None]*2048+(base+k[None,:]),m[:,None]<M,0)
  b=tl.load(B+(base+k[:,None])*N+n[None,:],n[None,:]<N,0)
  acc+=tl.dot(a,b,out_dtype=tl.int32)
 if DEBUG_INTEGER:tl.store(INTEGER+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))
 sa=tl.load(SA+m,m<M,0);sb=tl.load(SB+n,n<N,0)
 value=acc.to(tl.float32)*sa[:,None]
 value=value*sb[None,:]
 tl.store(Y+m[:,None]*N+n[None,:],value,(m[:,None]<M)&(n[None,:]<N))
