#!/usr/bin/env python3
"""Run the existing RMF-MT adapter on Fire as a labelled one-shot extension."""
from __future__ import annotations
import argparse,json,datetime as dt,subprocess,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.reproduction.fire_historical import ROOT,load,reference,fixed_target,scores,device,seed_all
from paper.benchmarks2026.common import ManifoldOps,make_rng
from paper.benchmarks2026.meanflow import create_method

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=2000);p.add_argument('--width',type=int,default=128);p.add_argument('--device',choices=('auto','mps','cpu'),default='auto');p.add_argument('--output-root',type=Path);a=p.parse_args()
 cfg,csv,split=load(ROOT/'configs/reproduction/fire_historical.json');dev=device(a.device,'mps_then_cpu');seed_all(a.seed);ops=ManifoldOps('sphere',3)
 method=create_method(ops,{'variant':'rmf_mt','width':a.width,'depth':3,'batch_size':256,'lr':2e-4,'weight_decay':.01,'prop_equal_times':.75,'time_distribution':'uniform'},dev)
 tier='rmf_mt_capacity_matched' if a.width==2 else 'rmf_mt_native';stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ');base=a.output_root if a.output_root else ROOT/'artifacts/comparison2026/fire'/tier;out=base/f'seed{a.seed}-{stamp}';out.mkdir(parents=True,exist_ok=False)
 train=split['train'].to(dev);rng_target=make_rng(50000+a.seed,dev);rng_method=make_rng(70000+a.seed,dev);curve=[]
 for step in range(1,a.steps+1):
  batch=train[torch.randint(len(train),(256,),device=dev,generator=rng_target)];logs=method.update(batch,rng_method)
  if step==1 or step%100==0 or step==a.steps:curve.append({'step':step,**{k:float(v.cpu()) if torch.is_tensor(v) else v for k,v in logs.items()}})
 target=fixed_target(split['test'],82011,1024);ref=reference(split['train']);rows=[];saved=[]
 for draw in range(3):
  x=method.sample(len(target),make_rng(420000+draw,dev),nfe=1);rows.append(scores(x,target,ref)|{'draw':draw,'model_forward_calls':1});saved.append(x.cpu().numpy())
 result={'scope':'new RMF-MT one-shot extension; not part of the historical Table-2 reproduction','tier':tier,'seed':a.seed,'steps':a.steps,'device':str(dev),'method_metadata':method.metadata,'parameter_count':method.parameter_count,'test_metrics':rows,'curve':curve}
 (out/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');import numpy as np;np.savez_compressed(out/'samples.npz',generated=np.stack(saved),target=target.numpy());print(f'Completed: {out}')
if __name__=='__main__':main()
