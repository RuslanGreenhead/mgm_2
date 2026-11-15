"""Same-L CUDA equivalence/throughput check using train data only."""
import json
import time
import numpy as np
import torch
from .common import ManifoldOps,make_rng
from .data_metrics import OUT,read,write
from .gpu_guard import foreign_python
from .rwflow import sinkhorn_field


def main():
    if foreign_python():raise RuntimeError("Wait for the other project's GPU training")
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device("cuda");rows=[];iteration_choices={};iteration_checks=[]
    names={"fire":"sphere","general":"torus2","rna":"torus7"}
    for name in ["fire","general","rna"]:
        manifest=read(OUT/"data"/name/"manifest.json")
        ops=ManifoldOps(manifest["kind"],manifest["dim"]);rng=make_rng(85001,device)
        train=torch.from_numpy(np.load(OUT/"data"/name/"data.npz")["train"]).to(device)
        real=train[torch.randint(len(train),(256,),generator=rng,device=device)]
        x=ops.base(256,rng,device);own=ops.base(256,rng,device)
        for epsilon in [.3,1.]:
            reference,_=sinkhorn_field(ops,x,real,own,epsilon=epsilon,iterations=128,solver="log")
            for solver in ["log","scaling","scaling64","graph64"]:
                durations=[]
                for repeat in range(6):
                    torch.cuda.synchronize();start=time.perf_counter()
                    field,logs=sinkhorn_field(ops,x,real,own,epsilon=epsilon,iterations=128,solver=solver)
                    torch.cuda.synchronize()
                    if repeat>=1:durations.append(time.perf_counter()-start)
                error=float((field-reference).abs().max())
                assert error<1e-5,(name,epsilon,solver,error)
                rows.append({"data":name,"epsilon":epsilon,"solver":solver,"iterations":128,
                    "max_field_error":error,"median_field_seconds":float(np.median(durations)),
                    "solver_diagnostics":{k:float(v) for k,v in logs.items()}})
    choice={}
    for name,group in [("fire","sphere"),("general","torus2"),("rna","torus7")]:
        options={solver:sum(r["median_field_seconds"] for r in rows if r["data"]==name and r["solver"]==solver) for solver in ["log","scaling","scaling64","graph64"]}
        choice[group]=min(options,key=options.get)
    for name,group in names.items():
        manifest=read(OUT/"data"/name/"manifest.json")
        ops=ManifoldOps(manifest["kind"],manifest["dim"]);rng=make_rng(85002,device)
        train=torch.from_numpy(np.load(OUT/"data"/name/"data.npz")["train"]).to(device)
        banks=[train[torch.randint(len(train),(256,),generator=rng,device=device)] for _ in range(3)]
        cases=[(ops.base(256,rng,device),banks[1],ops.base(256,rng,device)),tuple(banks)]
        iteration_choices[group]={}
        for epsilon in [.3,1.]:
            for iterations in [128,256,512,1024,2048]:
                residuals=[]
                for a,b,c in cases:
                    _,logs=sinkhorn_field(ops,a,b,c,epsilon=epsilon,iterations=iterations,solver=choice[group])
                    residuals.extend(float(logs[k]) for k in ["target_row_error","target_column_error","self_row_error","self_column_error"])
                error=max(residuals)
                iteration_checks.append({"group":group,"epsilon":epsilon,"iterations":iterations,"max_relative_marginal_error":error})
                if error<=1e-3:
                    iteration_choices[group][str(epsilon)]=iterations;break
            else:raise RuntimeError(f"Sinkhorn train-only marginal check failed: {group}, epsilon={epsilon}, error={error}")
    write(OUT/"solver_probe.json",{"same_L":128,"no_test_access":True,"gpu":torch.cuda.get_device_name(),
        "policy":"Equivalent finite-L solver selected by training-only GPU throughput. Numerical L refinement on uniform and independent train-resample banks to relative marginal error <=1e-3; no entropy/step/model tuning.",
        "group_solvers":choice,"iteration_choices":iteration_choices,"iteration_checks":iteration_checks,"probes":rows})
    print(json.dumps(choice))


if __name__=="__main__":main()
