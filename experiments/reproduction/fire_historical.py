#!/usr/bin/env python3
"""Portable, separately-labelled reconstruction of the controlled Fire protocol."""
from __future__ import annotations

import argparse, contextlib, datetime as dt, hashlib, json, math, random, subprocess, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
RFM_ROOT=ROOT/"external/riemannian-fm"
sys.path.insert(0,str(RFM_ROOT))
from manifm.manifolds.sphere import Sphere  # noqa:E402
from manifm.rmgm import build_critic,build_generator,critic_loss,generator_loss,metric_inner,constraint_error  # noqa:E402
from manifm.utils import cartesian_from_latlon  # noqa:E402
from rfm_baseline_train import BaseSampler,TrainConfig,rfm_path,sample_rfm  # noqa:E402


def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,default=ROOT/"configs/reproduction/fire_historical.json")
 p.add_argument("--method",choices=("rmgm_gmm","rmgm_mlp","mmd_mlp","rfm"),required=True);p.add_argument("--seed",type=int,required=True)
 p.add_argument("--steps",type=int);p.add_argument("--device",choices=("auto","mps","cpu"),default="auto");return p.parse_args()

def device(requested,preference):
 ok=torch.backends.mps.is_built() and torch.backends.mps.is_available()
 if requested=="mps" and not ok:raise RuntimeError("MPS requested but unavailable")
 return torch.device("cpu" if requested=="cpu" else "mps" if requested=="mps" or (requested=="auto" and preference=="mps_then_cpu" and ok) else "cpu")

def load(path):
 path=path if path.is_absolute() else ROOT/path;cfg=json.loads(path.read_text())
 csv=ROOT/cfg["data"]["csv"];a=np.loadtxt(csv,delimiter=",",skiprows=1,dtype=np.float64)
 if a.shape!=(cfg["data"]["expected_rows"],2):raise ValueError(f"Unexpected Fire data shape {a.shape}")
 raw=torch.tensor(a,dtype=torch.float32);x=cartesian_from_latlon(raw*(math.pi/180));x=x/x.norm(dim=-1,keepdim=True).clamp_min(1e-8)
 g=torch.Generator().manual_seed(cfg["data"]["split_seed"]);perm=torch.randperm(len(x),generator=g);n=len(x)//10
 split={"train":x[perm[:len(x)-2*n]],"val":x[perm[len(x)-2*n:len(x)-n]],"test":x[perm[len(x)-n:]]}
 if {k:len(v) for k,v in split.items()}!={"train":10249,"val":1280,"test":1280}:raise AssertionError("Fire split mismatch")
 return cfg,csv,split

def sphere_dist(x,y):
 x=x/x.norm(dim=-1,keepdim=True);y=y/y.norm(dim=-1,keepdim=True)
 return 2*torch.atan2((x[:,None]-y[None]).norm(dim=-1),(x[:,None]+y[None]).norm(dim=-1))

def reference(train):
 g=torch.Generator().manual_seed(72301);x=train[torch.randperm(len(train),generator=g)[:1024]].double();d2=torch.cdist(x,x).square();pos=d2[d2>1e-12]
 d=sphere_dist(x,x);d.fill_diagonal_(float("inf"));d[(x[:,None]==x[None]).all(-1)]=float("inf")
 return {"bandwidth2":float(pos.median()),"coverage_radius":float(d.min(1).values.median())}

def fixed_target(values,seed,n):
 g=torch.Generator().manual_seed(seed);return values[torch.randperm(len(values),generator=g)[:n]]

def scores(samples,target,ref):
 x,y=samples.detach().cpu().double(),target.detach().cpu().double();h=ref["bandwidth2"]
 m=(torch.exp(-torch.cdist(x,x).square()/h).mean()+torch.exp(-torch.cdist(y,y).square()/h).mean()-2*torch.exp(-torch.cdist(x,y).square()/h).mean())
 d=sphere_dist(x,y);a,b=d.min(1).values,d.min(0).values
 return {"mmd2_biased":float(m),"symmetric_nn":float((a.mean()+b.mean())/2),"coverage":float((b<=ref["coverage_radius"]).double().mean()),"constraint_mean":float((x.norm(dim=-1)-1).abs().mean())}

def kmeans_init(generator,train,model,dev):
 data=train[:min(model["kmeans_samples"],len(train))].to(dev);centers=generator.projected_centers().detach().clone();k=len(centers)
 for _ in range(model["kmeans_iterations"]):
  labels=torch.cdist(data,centers).square().argmin(1)
  for i in range(k):
   p=data[labels==i]
   if len(p):centers[i]=p.mean(0);centers[i]=centers[i]/centers[i].norm().clamp_min(1e-8)
 labels=torch.cdist(data,centers).square().argmin(1);scales=torch.empty_like(centers)
 manifold=Sphere()
 for i in range(k):
  p=data[labels==i]
  scales[i]=0.25 if len(p)<=1 else manifold.logmap(centers[i].expand_as(p),p).std(0,unbiased=False).abs().clamp(.03,1.)
 inv=lambda z:z+torch.log(-torch.expm1(-z))
 with torch.no_grad():
  generator.centers.copy_(centers);generator.log_scale.copy_(inv(scales));counts=torch.bincount(labels,minlength=k).to(scales);generator.component_logits.copy_(((counts+1)/(counts.sum()+k)).log())

def metrics_for(draw,target,ref):return [scores(draw(i),target,ref)|{"draw":i} for i in range(3)]
def finite(module):return all(bool(torch.isfinite(p).all()) for p in module.parameters())
def git():
 try:return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
 except Exception:return None
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def outdir(cfg,method,seed):
 base=(ROOT/cfg["output"]["root"]/method).resolve();stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ");p=base/f"seed{seed}-{stamp}";p.mkdir(parents=True,exist_ok=False);return p
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def main():
 a=args();cfg,csv,split=load(a.config);dev=device(a.device,cfg["runtime"]["device_preference"]);seed_all(a.seed);model=cfg["model"]
 steps=a.steps or (2000 if a.method in ("rmgm_gmm","rmgm_mlp","mmd_mlp") else 10000);out=outdir(cfg,a.method,a.seed);manifold=Sphere();target_val=fixed_target(split["val"],82010,cfg["data"]["evaluation_size"]);target_test=fixed_target(split["test"],82011,cfg["data"]["evaluation_size"]);ref=reference(split["train"])
 metadata={"scope":"portable reconstruction; separate from archived outputs", "method":a.method,"seed":a.seed,"steps":steps,"device":str(dev),"config":cfg,"csv_sha256":sha(csv),"source_commit":git(),"counts":{k:len(v) for k,v in split.items()},"reference":ref}
 (out/"config.json").write_text(json.dumps(metadata,indent=2)+"\n")
 log=[]
 if a.method in ("rmgm_gmm","rmgm_mlp"):
  generator_cfg={"generator":{"generator_type":"local_gmm","generator_components":model["mixture_components"],"latent_dim":model["latent_dim"],"hidden_dim":model["hidden_dim"],"num_layers":model["num_layers"]}} if a.method=="rmgm_gmm" else {"generator":{"generator_type":"mlp","latent_dim":model["latent_dim"],"hidden_dim":model["hidden_dim"],"num_layers":model["num_layers"]}}
  gen=build_generator(generator_cfg,manifold,3).to(dev)
  critic=build_critic({"critic":{"hidden_dim":model["hidden_dim"],"num_layers":model["num_layers"],"fourier":8}},manifold,3).to(dev)
  if a.method=="rmgm_gmm":kmeans_init(gen,split["train"],model,dev)
  og=torch.optim.AdamW(gen.parameters(),lr=model["learning_rate"]);oc=torch.optim.AdamW(critic.parameters(),lr=model["learning_rate"])
  def draw(i):g=torch.Generator(device=dev).manual_seed(100000+i);return gen(torch.randn(len(target_test),model["latent_dim"],device=dev,generator=g))
  initial={"val":metrics_for(draw,target_val,ref),"test":metrics_for(draw,target_test,ref)};torch.save({"step":0,"generator":gen.state_dict(),"critic":critic.state_dict()},out/"checkpoint_step0.pt")
  train=split["train"].to(dev);start=time.perf_counter()
  for step in range(1,steps+1):
   z=torch.randn(model["batch_size"],model["latent_dim"],device=dev);x=train[torch.randint(len(train),(model["batch_size"],),device=dev)];oc.zero_grad(set_to_none=True);lc,cm=critic_loss(manifold,gen,critic,z,x);lc.backward();oc.step()
   critic.requires_grad_(False);z=torch.randn(model["batch_size"],model["latent_dim"],device=dev);x=train[torch.randint(len(train),(model["batch_size"],),device=dev)];og.zero_grad(set_to_none=True);lg,gm=generator_loss(manifold,gen,critic,z,x);lg.backward();og.step();critic.requires_grad_(True)
   if step==1 or step%cfg["runtime"]["log_every"]==0 or step==steps:
    row={"step":step,"critic_loss":float(lc.detach().cpu()),"generator_loss":float(lg.detach().cpu())};log.append(row);print(json.dumps(row),flush=True)
  final={"val":metrics_for(draw,target_val,ref),"test":metrics_for(draw,target_test,ref)};torch.save({"step":steps,"generator":gen.state_dict(),"critic":critic.state_dict()},out/f"checkpoint_step{steps}.pt");extra={"generator_class":gen.__class__.__name__,"critic_class":critic.__class__.__name__,"finite_parameters":finite(gen) and finite(critic)}
 elif a.method=="mmd_mlp":
  gen=build_generator({"generator":{"generator_type":"mlp","latent_dim":model["latent_dim"],"hidden_dim":model["hidden_dim"],"num_layers":model["num_layers"]}},manifold,3).to(dev);opt=torch.optim.AdamW(gen.parameters(),lr=model["learning_rate"]);train=split["train"].to(dev);start=time.perf_counter()
  def draw(i):g=torch.Generator(device=dev).manual_seed(100000+i);return gen(torch.randn(len(target_test),model["latent_dim"],device=dev,generator=g))
  initial={"val":metrics_for(draw,target_val,ref),"test":metrics_for(draw,target_test,ref)}
  for step in range(1,steps+1):
   z=torch.randn(model["batch_size"],model["latent_dim"],device=dev);x=gen(z);y=train[torch.randint(len(train),(model["batch_size"],),device=dev)];h=ref["bandwidth2"];loss=(torch.exp(-torch.cdist(x,x).square()/h).mean()+torch.exp(-torch.cdist(y,y).square()/h).mean()-2*torch.exp(-torch.cdist(x,y).square()/h).mean());opt.zero_grad(set_to_none=True);loss.backward();opt.step()
   if step==1 or step%cfg["runtime"]["log_every"]==0 or step==steps:log.append({"step":step,"mmd_loss":float(loss.detach().cpu())})
  final={"val":metrics_for(draw,target_val,ref),"test":metrics_for(draw,target_test,ref)};torch.save({"step":steps,"generator":gen.state_dict()},out/f"checkpoint_step{steps}.pt");extra={"generator_class":gen.__class__.__name__,"finite_parameters":finite(gen)}
 else:
  field=build_critic({"critic":{"hidden_dim":model["hidden_dim"],"num_layers":model["num_layers"],"fourier":8}},manifold,3).to(dev);base=BaseSampler(manifold,3);opt=torch.optim.AdamW(field.parameters(),lr=model["learning_rate"])
  def draw(i):g=torch.Generator(device=dev).manual_seed(100000+i);tc=TrainConfig(experiment="fire",ode_steps=model["rfm_ode_steps"]);return sample_rfm(tc,manifold,base,field,len(target_test),dev,torch.float32)
  snapshots={}
  train=split["train"].to(dev);start=time.perf_counter()
  for step in range(1,steps+1):
   x0=base.sample(model["batch_size"],dev);x1=train[torch.randint(len(train),(model["batch_size"],),device=dev)];t=torch.rand(model["batch_size"],1,device=dev);xt,u=rfm_path(manifold,x0,x1,t);opt.zero_grad(set_to_none=True);pred=field(t,xt);loss=manifold.inner(xt,pred-u,pred-u).mean()/3;loss.backward();opt.step()
   if step==1 or step%cfg["runtime"]["log_every"]==0 or step in (2000,steps):log.append({"step":step,"rfm_loss":float(loss.detach().cpu())})
   if step in (2000,steps):snapshots[str(step)]={"val":metrics_for(draw,target_val,ref),"test":metrics_for(draw,target_test,ref)};torch.save({"step":step,"field":field.state_dict()},out/f"checkpoint_step{step}.pt")
  initial=None;final=snapshots;extra={"field_class":field.__class__.__name__,"finite_parameters":finite(field)}
 result={"metadata":metadata,"initial":initial,"final":final,"learning_curve":log,"train_seconds":time.perf_counter()-start,**extra}
 (out/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n");print(f"Completed: {out}")
if __name__=="__main__":main()
