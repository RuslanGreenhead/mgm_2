"""Freeze a phase and run its jobs serially on the user's single GPU."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .data_metrics import read,write,sha,ROOT,OUT


def main():
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);args=p.parse_args()
    plan=read(args.plan);phase=OUT/plan["phase"]
    snapshot=phase/"source_snapshot";source=ROOT/"paper/benchmarks2026"
    phase.mkdir(parents=True,exist_ok=True)
    if not (phase/"protocol.json").exists():
        files=list(source.glob("*.py"))+list(source.glob("*NOTES.md"));hashes={}
        if any(j["method"]=="kgdm" for j in plan["jobs"]):
            files.append(ROOT/"external/kgdm-author/earth/drift_core.py")
        for path in files:
            target=snapshot/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target);hashes[str(path.relative_to(ROOT))]=sha(target)
        write(phase/"protocol.json",{"plan":plan,"plan_sha256":sha(args.plan),"source_sha256":hashes,
            "selection":"Validation only. Test scoring disabled for tuning; frozen final jobs are scored once after training.",
            "gpu_policy":"Single sequential queue; no sidecar GPU training.","created_unix":time.time()})
    protocol=read(phase/"protocol.json")
    assert protocol["plan"]==plan
    jobs=plan["jobs"];done=[]
    for i,job in enumerate(jobs):
        for rel,h in protocol["source_sha256"].items():assert sha(snapshot/rel)==h,rel
        folder=ROOT/job["output"]
        if (folder/"completed.json").exists():done.append(read(folder/"completed.json"));continue
        jpath=phase/"jobs"/f"{i:03d}.json";write(jpath,job)
        print(json.dumps({"event":"job_start","index":i+1,"total":len(jobs),"data":job["data"],"method":job["method"],"seed":job["seed"]}),flush=True)
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/"train.log").open("a",encoding="utf-8") as log:
            proc=subprocess.run([sys.executable,"-u",str(snapshot/"paper/benchmarks2026/worker.py"),"--root",str(ROOT),"--job",str(jpath)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode:
            write(phase/"queue_failure.json",{"index":i,"job":job,"return_code":proc.returncode})
            raise RuntimeError(f"Job failed; inspect {folder/'train.log'}")
        done.append(read(folder/"completed.json"))
        write(phase/"progress.json",{"finished":len(done),"planned":len(jobs),"last":done[-1]})
        print(json.dumps({"event":"job_finished","finished":len(done),"planned":len(jobs),**done[-1]}),flush=True)
    write(phase/"completed.json",{"all_completed":True,"jobs":done,"summed_training_seconds":sum(r["train_seconds"] for r in done)})


if __name__=="__main__":main()
