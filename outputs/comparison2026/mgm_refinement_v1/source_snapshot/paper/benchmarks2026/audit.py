"""Verify frozen final artifacts, counts, selected scores and MMD numerically."""
import argparse
import numpy as np
from .data_metrics import ROOT,OUT,read,write,sha


def independent_mmd(x,y,kind,h2):
    if kind=="torus":
        x=np.concatenate((np.cos(x),np.sin(x)),axis=1)
        y=np.concatenate((np.cos(y),np.sin(y)),axis=1)
    def kernel(a,b):
        d2=((a[:,None,:]-b[None,:,:])**2).sum(-1)
        return np.exp(-d2/h2).mean()
    return max(0.,float(kernel(x,x)+kernel(y,y)-2*kernel(x,y)))


def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",default="final_v1");a=p.parse_args()
    phase=OUT/a.phase;protocol=read(phase/"protocol.json");plan=protocol["plan"]
    assert read(phase/"completed.json")["all_completed"]
    for rel,digest in protocol["source_sha256"].items():
        assert sha(phase/"source_snapshot"/rel)==digest,rel
    rows=[]
    for job in plan["jobs"]:
        f=ROOT/job["output"];done=read(f/"completed.json");meta=read(f/"job.json")
        assert done["steps"]==job["steps"] and done["test_evaluations"]==3
        assert done["attempts"]==job["steps"]+(job["config"].get("critic_warmup",500) if job["method"]=="mgm" else 0)
        expected_critic=job["config"].get("critic_warmup",500)+job["steps"]*job["config"].get("critic_steps",5) if job["method"]=="mgm" else 0
        assert done["critic_updates"]==expected_critic
        assert meta["executed_worker_sha256"]==protocol["source_sha256"]["paper/benchmarks2026/worker.py"]
        assert meta["method_source_sha256"]==protocol["source_sha256"][f"paper/benchmarks2026/{job['method']}.py"]
        data=OUT/"data"/job["data"];manifest=read(data/"manifest.json")
        assert sha(data/"manifest.json")==meta["data_manifest_sha256"]
        assert sha(data/"data.npz")==manifest["arrays_sha256"]
        target=np.load(data/"data.npz")["test_target"].astype(np.float64)
        testfiles=list(f.glob("test_step*_metrics.json"));assert len(testfiles)==1
        test=read(testfiles[0]);assert len(test)==3
        samples=np.load(f/f"test_step{done['selected_step']}_samples.npz")
        differences=[]
        for i,row in enumerate(test):
            assert row["model_forward_calls"]==1 and row["step"]==done["selected_step"]
            assert row["seed"]==job["seed"] and row["draw"]==i and row["split"]=="test"
            x=samples[f"generated_draw_{i}"].astype(np.float64)
            assert np.isfinite(x).all() and len(x)==len(target)
            if manifest["kind"]=="sphere":assert np.max(np.abs(np.linalg.norm(x,axis=1)-1))<1e-5
            else:assert np.all((x>=0)&(x<2*np.pi))
            error=abs(independent_mmd(x,target,manifest["kind"],manifest["reference"]["bandwidth2"])-row["kernel_mmd2_v"])
            assert error<1e-10,error;differences.append(error)
        val=read(f/f"val_step{done['selected_step']}_metrics.json")
        assert abs(np.mean([r["kernel_mmd2_v"] for r in val])-done["selected_validation_mmd2"])<1e-12
        assert abs(np.mean([r["kernel_mmd2_v"] for r in test])-done["test_mmd2"])<1e-12
        rows.append({"data":job["data"],"method":job["method"],"seed":job["seed"],"max_mmd_recompute_error":max(differences)})
    report={"passed":True,"job_count":len(rows),"checks":"Frozen source/data hashes, generator/critic counts, unique 3-draw test, 1 NFE, constraint bounds, selected-val/test arithmetic, independent NumPy MMD recomputation.","jobs":rows}
    write(phase/"audit.json",report);print(f"PASS: {len(rows)} jobs; max MMD error {max(r['max_mmd_recompute_error'] for r in rows):.3g}")


if __name__=="__main__":main()
