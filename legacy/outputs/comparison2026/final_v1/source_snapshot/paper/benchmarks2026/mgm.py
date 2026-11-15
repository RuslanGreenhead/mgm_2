"""Intrinsic hidden-flip MGM with periodic critic and configurable two-timescale updates."""
import math
import torch
from torch import nn
from .common import MLP,PointGenerator,TimeFeatures


class Critic(nn.Module):
    def __init__(self,ops,width=128,depth=3,harmonics=4):
        super().__init__();self.ops=ops;self.time=TimeFeatures(8)
        self.register_buffer("frequencies",torch.arange(1,harmonics+1,dtype=torch.float32))
        d=ops.dim if ops.kind=="sphere" else 2*ops.dim*harmonics
        self.net=MLP(d+16,ops.dim,width,depth)

    def forward(self,t,x):
        if self.ops.kind=="torus":
            a=x[...,None]*self.frequencies.to(x)
            features=torch.cat([a.cos(),a.sin()],-1).flatten(1)
        else:features=x
        return self.ops.tangent(x,self.net(torch.cat([features,self.time(t).to(x)],-1)))


class MGM:
    def __init__(self,ops,cfg,device):
        self.ops,self.cfg,self.device=ops,cfg,device
        self.model=PointGenerator(ops,cfg.get("latent_dim",16),cfg.get("width",128),cfg.get("depth",3)).to(device)
        self.critic=Critic(ops,cfg.get("critic_width",cfg.get("width",128)),cfg.get("depth",3),cfg.get("harmonics",4)).to(device)
        self.opt=torch.optim.AdamW(self.model.parameters(),lr=cfg.get("lr",2e-4),weight_decay=0)
        self.opt_c=torch.optim.AdamW(self.critic.parameters(),lr=cfg.get("lr_c",2e-4),weight_decay=0)
        self.iteration=0

    def observation(self,z,y,rng,detach=False):
        x0=self.model(z)
        if detach:x0=x0.detach()
        t=torch.rand(len(y),1,device=y.device,generator=rng)*.5
        b=torch.randint(2,(len(y),1),device=y.device,generator=rng)
        s=t+b*(1-2*t)
        amplitude=self.cfg.get("noise",0.) if self.ops.kind=="torus" else 0.
        noise=torch.randn(y.shape,device=y.device,dtype=y.dtype,generator=rng) if amplitude else None
        x,a=self.ops.path(x0,y,s,noise,amplitude)
        return t,x,a

    def update(self,real_batch,rng):
        self.model.train();self.critic.train();n=len(real_batch)
        steps=self.cfg.get("critic_steps",5)
        warm=self.cfg.get("critic_warmup",500)
        only=self.iteration<warm
        for _ in range(1 if only else steps):
            z=torch.randn(n,self.model.latent_dim,device=self.device,generator=rng)
            with torch.no_grad():t,x,a=self.observation(z,real_batch,rng,True)
            self.opt_c.zero_grad(set_to_none=True)
            residual=self.critic(t,x)-a
            lc=self.ops.inner(x,residual,residual).mean()/self.ops.dim
            lc.backward();self.opt_c.step()
        if only:
            self.iteration+=1
            return {"loss":lc.detach(),"critic_loss":lc.detach(),"generator_updated":0}
        z=torch.randn(n,self.model.latent_dim,device=self.device,generator=rng)
        self.critic.requires_grad_(False);self.opt.zero_grad(set_to_none=True)
        t,x,a=self.observation(z,real_batch,rng)
        f=self.critic(t,x)
        loss=(2*self.ops.inner(x,f,a)-self.ops.inner(x,f,f)).mean()/self.ops.dim
        loss.backward();self.opt.step();self.critic.requires_grad_(True)
        self.iteration+=1
        return {"loss":loss.detach(),"critic_loss":lc.detach(),"generator_updated":1}

    @torch.no_grad()
    def sample(self,n,rng,nfe=1):
        self.model.eval()
        return self.model(torch.randn(n,self.model.latent_dim,device=self.device,generator=rng))

    def checkpoint(self):
        return {"model":self.model.state_dict(),"critic":self.critic.state_dict(),"optimizer":self.opt.state_dict(),
                "critic_optimizer":self.opt_c.state_dict(),"iteration":self.iteration}

    def load_checkpoint(self,state):
        self.model.load_state_dict(state["model"]);self.critic.load_state_dict(state["critic"])
        self.opt.load_state_dict(state["optimizer"]);self.opt_c.load_state_dict(state["critic_optimizer"])
        self.iteration=state["iteration"]


def create_method(ops,cfg,device):return MGM(ops,cfg,device)
