"""One GPU job at a time, isolated RNGs, validation selection, final-only test scoring."""
import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paper.benchmarks2026.common import ManifoldOps,make_rng
from paper.benchmarks2026.data_metrics import read,write,sha,metrics


def synchronize():
    torch.cuda.synchronize();return time.perf_counter()


def main():
    p=argparse.ArgumentParser();p.add_argument("--job",type=Path,required=True);p.add_argument("--root",type=Path,required=True)
    args=p.parse_args();job=read(args.job);root=args.root.resolve()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    seed=job["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    device=torch.device("cuda");folder=root/job["output"];folder.mkdir(parents=True,exist_ok=True)
    if (folder/"completed.json").exists():raise RuntimeError("Completed job must not be overwritten")
    m=read(root/"outputs/comparison2026/data"/job["data"]/"manifest.json")
    datafile=root/"outputs/comparison2026/data"/job["data"]/"data.npz"
    assert sha(datafile)==m["arrays_sha256"]
    arrays=np.load(datafile);train=torch.from_numpy(arrays["train"]).to(device)
    val=torch.from_numpy(arrays["val_target"]);test=torch.from_numpy(arrays["test_target"])
    ops=ManifoldOps(m["kind"],m["dim"]);cfg=job["config"]
    module=importlib.import_module("paper.benchmarks2026."+job["method"])
    method=module.create_method(ops,cfg,device)
    target_rng=make_rng(50000+seed,device);train_rng=make_rng(70000+seed,device)
    seconds=0.;start_step=0;best=float("inf");best_step=0;curve=[];attempts=0
    def states():
        return {"target":target_rng.get_state(),"method":train_rng.get_state(),"cpu":torch.random.get_rng_state(),"cuda":torch.cuda.get_rng_state()}
    def save(path,step):
        torch.save({"method_state":method.checkpoint(),"rng":states(),"job":job,"step":step,"attempts":attempts,
            "train_seconds":seconds,"best_val":best,"best_step":best_step,"curve":curve},path)
    last=folder/"last.pt"
    if last.exists():
        s=torch.load(last,map_location=device,weights_only=False);method.load_checkpoint(s["method_state"])
        target_rng.set_state(s["rng"]["target"].cpu());train_rng.set_state(s["rng"]["method"].cpu())
        torch.random.set_rng_state(s["rng"]["cpu"].cpu());torch.cuda.set_rng_state(s["rng"]["cuda"].cpu())
        start_step=s["step"];seconds=s["train_seconds"];best=s["best_val"];best_step=s["best_step"];curve=s["curve"];attempts=s["attempts"]
    write(folder/"job.json",{**job,"data_manifest_sha256":sha(root/"outputs/comparison2026/data"/job["data"]/"manifest.json"),
        "model_parameters":sum(p.numel() for p in method.model.parameters()),"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,
        "executed_worker_sha256":sha(__file__),"method_source_sha256":sha(module.__file__)})

    @torch.no_grad()
    def evaluate(split,step,draws=3):
        before=states();rows=[];saved={};target=val if split=="val" else test
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            for draw in range(draws):
                rng=make_rng(420000+draw,device)
                calls=[0]
                hook=method.model.register_forward_hook(lambda *args:calls.__setitem__(0,calls[0]+1))
                samples=method.sample(len(target),rng,nfe=1)
                hook.remove()
                assert torch.isfinite(samples).all()
                saved[f"generated_draw_{draw}"]=samples.cpu().numpy()
                row=metrics(ops,samples,target,m["reference"])
                row.update(data=job["data"],method=job["method"],seed=seed,step=step,split=split,draw=draw,train_seconds=seconds,model_forward_calls=calls[0])
                rows.append(row)
        after=states();assert all(torch.equal(before[k],after[k]) for k in before),"Evaluation changed training RNG"
        write(folder/f"{split}_step{step}_metrics.json",rows)
        np.savez_compressed(folder/f"{split}_step{step}_samples.npz",**saved)
        return float(np.mean([r["kernel_mmd2_v"] for r in rows]))

    step=start_step
    if not start_step:
        best=evaluate("val",0,job.get("validation_draws",1));best_step=0
        save(folder/"best.pt",0)
    started=synchronize();eval_every=job.get("eval_every",1000)
    try:
        while step<job["steps"]:
            batch=train[torch.randint(len(train),(cfg.get("batch_size",256),),device=device,generator=target_rng)]
            logs=method.update(batch,train_rng);attempts+=1
            if logs.get("generator_updated",1):step+=1
            if attempts%100==0:
                item={"step":step,"attempts":attempts,**{k:float(v.detach()) if torch.is_tensor(v) else v for k,v in logs.items()}}
                assert all(np.isfinite(v) for v in item.values() if isinstance(v,(int,float))),item
                curve.append(item)
            if step and (step%eval_every==0 or step==job["steps"]) and logs.get("generator_updated",1):
                seconds+=synchronize()-started
                score=evaluate("val",step,job.get("validation_draws",1))
                if score<best:best,best_step=score,step;save(folder/"best.pt",step)
                save(last,step);write(folder/"curve.json",curve)
                write(folder/"progress.json",{"step":step,"planned":job["steps"],"train_seconds":seconds,"validation_mmd2":score,"best_val":best,"best_step":best_step})
                print(json.dumps({"event":"validation", "data":job["data"],"method":job["method"],"seed":seed,"step":step,"val":score,"seconds":seconds}),flush=True)
                started=synchronize()
        if job.get("score_test",True):
            selected=torch.load(folder/"best.pt",map_location=device,weights_only=False)
            method.load_checkpoint(selected["method_state"])
            score=evaluate("test",best_step,3)
            durations=[]
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                for draw in range(7):
                    started_sample=synchronize()
                    method.sample(1024,make_rng(530000+draw,device),nfe=1)
                    duration=synchronize()-started_sample
                    if draw>=2:durations.append(duration)
            write(folder/"latency.json",{"batch":1024,"gpu_seconds":durations,"warmups":2,"includes":"RNG, neural inference, geometric output map"})
        else:score=None
        write(folder/"completed.json",{"data":job["data"],"method":job["method"],"seed":seed,"steps":step,"attempts":attempts,
            "train_seconds":seconds,"selected_step":best_step,"selected_validation_mmd2":best,"test_mmd2":score,
            "test_evaluations":3 if score is not None else 0,"selection":"Lowest fixed-probe validation MMD; no test tuning"})
    except Exception:
        seconds+=synchronize()-started
        save(folder/"interrupted.pt",step)
        raise


if __name__=="__main__":main()
