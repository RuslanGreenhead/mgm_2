"""Train-only resource scheduling probe; never scores held-out data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from .common import ManifoldOps,make_rng
from .data_metrics import ROOT,OUT,write,read
from .gpu_guard import foreign_python
from .mgm import create_method


def child(seed,path):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);device=torch.device("cuda")
    ops=ManifoldOps("sphere",3)
    model=create_method(ops,{"width":128,"depth":3,"critic_width":256,"critic_steps":10,
        "critic_warmup":50,"lr":2e-4,"lr_c":4e-4},device)
    train=torch.from_numpy(np.load(OUT/"data/fire/data.npz")["train"]).to(device)
    data_rng=make_rng(50000+seed,device);rng=make_rng(70000+seed,device)
    for _ in range(50):model.update(train[torch.randint(len(train),(256,),device=device,generator=data_rng)],rng)
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(500):model.update(train[torch.randint(len(train),(256,),device=device,generator=data_rng)],rng)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    digest=hashlib.sha256()
    for module in [model.model,model.critic]:
        for key,value in sorted(module.state_dict().items()):
            array=value.detach().cpu().contiguous().numpy()
            assert np.isfinite(array).all()
            digest.update(key.encode());digest.update(array.tobytes())
    write(path,{"seed":seed,"steps":500,"training_seconds":seconds,"weights_sha256":digest.hexdigest(),
        "peak_allocated_bytes":torch.cuda.max_memory_allocated()})


def main():
    p=argparse.ArgumentParser();p.add_argument("--child-seed",type=int);p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.child_seed is not None:return child(args.child_seed,args.output)
    if foreign_python():raise RuntimeError("Foreign GPU training is active")
    base=OUT/"throughput_probe";base.mkdir(parents=True,exist_ok=True);results={};walls={}
    for mode in ["serial","parallel"]:
        started=time.perf_counter();children=[]
        for seed in [1010,1011]:
            output=base/f"{mode}_s{seed}.json"
            argv=[sys.executable,"-u","-m","paper.benchmarks2026.throughput_probe","--child-seed",str(seed),"--output",str(output)]
            if mode=="serial":subprocess.run(argv,cwd=ROOT,check=True)
            else:children.append(subprocess.Popen(argv,cwd=ROOT))
        for process in children:
            if process.wait()!=0:raise RuntimeError("Concurrent resource probe failed")
        walls[mode]=time.perf_counter()-started
        results[mode]=[read(base/f"{mode}_s{seed}.json") for seed in [1010,1011]]
    exact=all(a["weights_sha256"]==b["weights_sha256"] for a,b in zip(results["serial"],results["parallel"]))
    speedup=walls["serial"]/walls["parallel"]
    write(OUT/"throughput_probe.json",{"no_test_access":True,"model_recipe":"MGM 128x3 G, 256x3 C, 10 critic steps",
        "wall_seconds":walls,"speedup":speedup,"bitwise_equal_models_and_critics":exact,
        "recommended_concurrency":2 if exact and speedup>1.15 else 1,"runs":results,
        "policy":"If concurrent training is selected, report contended training wall time and measure inference separately after all workers exit."})
    print(json.dumps({"speedup":speedup,"exact_weights":exact,"concurrency":2 if exact and speedup>1.15 else 1}))


if __name__=="__main__":main()
