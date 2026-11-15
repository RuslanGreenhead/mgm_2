"""Shared geometry and data-independent neural generator for the new comparison."""
from __future__ import annotations

import math
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self,d_in,d_out,width=128,depth=3,zero_last=False):
        super().__init__()
        blocks=[]
        for i in range(depth):
            blocks.append(nn.Linear(d_in if i==0 else width,d_out if i==depth-1 else width))
            if i<depth-1:blocks.append(nn.SiLU())
        self.net=nn.Sequential(*blocks)
        if zero_last:
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):return self.net(x)


class TimeFeatures(nn.Module):
    def __init__(self,num_frequencies=8):
        super().__init__()
        self.register_buffer("frequencies",2.**torch.arange(num_frequencies))

    def forward(self,t):
        if t.ndim==1:t=t[:,None]
        a=2*math.pi*t*self.frequencies.to(t)
        return torch.cat([a.sin(),a.cos()],-1)


class ManifoldOps:
    def __init__(self,manifold,dim):
        name=(manifold if isinstance(manifold,str) else manifold.__class__.__name__).lower()
        self.kind="torus" if "torus" in name else "sphere" if "sphere" in name else None
        if self.kind is None:raise ValueError(name)
        self.dim=self.coord_dim=int(dim)
        self.intrinsic_dim=self.dim-1 if self.kind=="sphere" else self.dim
        self.embed_dim=self.dim if self.kind=="sphere" else 2*self.dim

    def embed(self,x):return torch.cat([x.cos(),x.sin()],-1) if self.kind=="torus" else x

    def proj(self,x):
        if self.kind=="torus":return x.remainder(2*math.pi)
        norm=x.norm(dim=-1,keepdim=True)
        fallback=torch.zeros_like(x);fallback[...,0]=1
        return torch.where(norm>1e-12,x/norm.clamp_min(1e-12),fallback)

    def tangent(self,x,v):return v-(x*v).sum(-1,keepdim=True)*x if self.kind=="sphere" else v

    def inner(self,x,u,v):return (u*v).sum(-1)

    def exp(self,x,v):
        if self.kind=="torus":return (x+v).remainder(2*math.pi)
        v=self.tangent(x,v);q=v.square().sum(-1,keepdim=True)
        r=q.clamp_min(1e-16).sqrt()
        # Polynomial branches keep mixed JVP/parameter derivatives finite at zero.
        c=torch.where(q<1e-8,1-q/2+q.square()/24,torch.cos(r))
        s=torch.where(q<1e-8,1-q/6+q.square()/120,torch.sin(r)/r)
        return self.proj(c*x+s*v)

    def log(self,x,y):
        if self.kind=="torus":
            delta=y-x
            return torch.atan2(delta.sin(),delta.cos())
        x=self.proj(x);y=self.proj(y)
        dot=(x*y).sum(-1,keepdim=True).clamp(-1,1)
        orth=y-dot*x;s=orth.square().sum(-1,keepdim=True).clamp_min(1e-16).sqrt()
        angle=torch.atan2(s,dot)
        basis=torch.nn.functional.one_hot(x.abs().argmin(-1),self.dim).to(x)
        fallback=self.tangent(x,basis)
        fallback=fallback/fallback.norm(dim=-1,keepdim=True).clamp_min(1e-12)*math.pi
        return torch.where((dot<0)&(orth.square().sum(-1,keepdim=True)<1e-14),fallback,orth*(angle/s))

    def distance2(self,x,y):
        if self.kind=="torus":return self.log(x,y).square().sum(-1)
        x=self.proj(x);y=self.proj(y)
        angle=2*torch.atan2((x-y).norm(dim=-1),(x+y).norm(dim=-1))
        return angle.square()

    def pairwise_distance2(self,x,y):return self.distance2(x[:,None],y[None])

    def base(self,n,rng,device,dtype=torch.float32):
        if self.kind=="sphere":return self.proj(torch.randn(n,self.dim,device=device,dtype=dtype,generator=rng))
        return torch.rand(n,self.dim,device=device,dtype=dtype,generator=rng)*(2*math.pi)

    def base_from_latent(self,z):
        if z.shape[-1]<self.dim:raise ValueError("Latent width smaller than representation")
        if self.kind=="sphere":return self.proj(z[...,:self.dim])
        return (0.5*(1+torch.erf(z[...,:self.dim]/math.sqrt(2)))*2*math.pi).clamp_max(2*math.pi-1e-6)

    def path(self,x0,x1,s,noise=None,amplitude=0.):
        delta=self.log(x0,x1)
        if self.kind=="torus":
            if noise is None or amplitude==0:return self.exp(x0,s*delta),delta
            return self.exp(x0,s*delta+amplitude*torch.sin(math.pi*s)*noise),delta+amplitude*math.pi*torch.cos(math.pi*s)*noise
        q=delta.square().sum(-1,keepdim=True);r=q.clamp_min(1e-16).sqrt()
        sr=s*r
        c=torch.cos(sr);sinc=torch.where(q<1e-8,s-s.pow(3)*q/6,torch.sin(sr)/r)
        point=self.proj(c*x0+sinc*delta)
        velocity=-r*torch.sin(sr)*x0+torch.cos(sr)*delta
        return point,self.tangent(point,velocity)


class PointGenerator(nn.Module):
    """Shared random-initialized residual map of a uniform manifold base."""
    def __init__(self,ops,latent_dim=16,width=128,depth=3):
        super().__init__();self.ops=ops;self.latent_dim=max(latent_dim,ops.dim)
        self.raw=MLP(ops.embed_dim,ops.dim,width,depth,zero_last=True)

    def forward(self,z):
        base=self.ops.base_from_latent(z)
        return self.ops.exp(base,self.ops.tangent(base,self.raw(self.ops.embed(base))))


def make_rng(seed,device="cpu"):
    return torch.Generator(device=device).manual_seed(int(seed))


def sample_base(ops,n,rng,device,dtype=torch.float32):return ops.base(n,rng,device,dtype)
