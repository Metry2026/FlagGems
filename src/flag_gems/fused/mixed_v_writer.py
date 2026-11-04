import triton
import triton.language as tl

@triton.jit
def mixed_v_writer(KIN,VIN,K,V,S,SLOTS,ki0:tl.constexpr,ki1:tl.constexpr,ki2:tl.constexpr,vi0:tl.constexpr,vi1:tl.constexpr,vi2:tl.constexpr,k0:tl.constexpr,k1:tl.constexpr,k2:tl.constexpr,k3:tl.constexpr,v0:tl.constexpr,v1:tl.constexpr,v2:tl.constexpr,v3:tl.constexpr,s0:tl.constexpr,s1:tl.constexpr,s2:tl.constexpr):
 row=tl.program_id(0);h=tl.program_id(1);d=tl.arange(0,128)
 slot=tl.load(SLOTS+row).to(tl.int64)
 if slot<0:return
 b=slot//16;t=slot%16
 kval=tl.load(KIN+row*ki0+h*ki1+d*ki2)
 x=tl.load(VIN+row*vi0+h*vi1+d*vi2).to(tl.float32)
 finite=tl.sum((tl.abs(x)<float('inf')).to(tl.int32),0)==128
 a=tl.max(tl.abs(x),0);scale=tl.where(a==0,1.0,tl.div_rn(a,127.0));scale=tl.where(finite,scale,float('nan'))
 z=tl.div_rn(x,scale);lo=tl.floor(z);f=z-lo
 r=lo+((f>0.5)|((f==0.5)&((lo.to(tl.int32)&1)!=0))).to(tl.float32)
 q=tl.minimum(127.0,tl.maximum(-127.0,r)).to(tl.int8)
 tl.store(K+b*k0+t*k1+h*k2+d*k3,kval)
 tl.store(V+b*v0+t*v1+h*v2+d*v3,tl.where(finite,q,0))
 tl.store(S+b*s0+t*s1+h*s2,scale)
