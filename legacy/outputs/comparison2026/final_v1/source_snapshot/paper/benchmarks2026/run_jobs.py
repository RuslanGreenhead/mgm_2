"""Freeze a phase and run a bounded, process-isolated CUDA queue."""
import argparse
from concurrent.futures import CancelledError,ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from .data_metrics import read,write,sha,ROOT,OUT
from .gpu_guard import wait_until_free


def main():
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);args=p.parse_args()
    plan=read(args.plan);phase=OUT/plan["phase"]
    workers=plan.get("max_workers",1)
    if workers not in (1,2):raise ValueError("Only validated concurrency 1 or 2 is supported")
    snapshot=phase/"source_snapshot";source=ROOT/"paper/benchmarks2026"
    phase.mkdir(parents=True,exist_ok=True)
    if not (phase/"protocol.json").exists():
        files=list(source.glob("*.py"))+list(source.glob("*NOTES.md"));hashes={}
        if any(j["method"]=="kgdm" for j in plan["jobs"]):
            files.append(ROOT/"external/kgdm-author/earth/drift_core.py")
            files.append(ROOT/"external/kgdm-author/LICENSE")
        for path in files:
            target=snapshot/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target);hashes[str(path.relative_to(ROOT))]=sha(target)
        write(phase/"protocol.json",{"plan":plan,"plan_sha256":sha(args.plan),"source_sha256":hashes,
            "selection":"Validation only. Test scoring disabled for tuning; frozen final jobs are scored once after training.",
            "gpu_policy":f"At most {workers} isolated training processes. Sampling latency is profiled separately when concurrent.","created_unix":time.time()})
    protocol=read(phase/"protocol.json")
    assert protocol["plan"]==plan
    jobs=plan["jobs"];done=[];pending=[];running={};lock=threading.Lock();abort=threading.Event();started=time.time()
    def runtime(status="running"):
        write(phase/"runtime.json",{"status":status,"queue_pid":os.getpid(),"workers":dict(running),
            "max_workers":workers,"started_unix":started})
    for i,job in enumerate(jobs):
        folder=ROOT/job["output"]
        if (folder/"completed.json").exists():done.append(read(folder/"completed.json"))
        else:pending.append((i,job))

    def execute(i,job):
        if abort.is_set():raise CancelledError("Queue stopped after another worker failed")
        for rel,h in protocol["source_sha256"].items():assert sha(snapshot/rel)==h,rel
        folder=ROOT/job["output"]
        wait_until_free(stop_event=abort)
        jpath=phase/"jobs"/f"{i:03d}.json";write(jpath,job)
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/"train.log").open("a",encoding="utf-8") as log:
            proc=None;returncode=None
            try:
                with lock:
                    if abort.is_set():raise CancelledError("Queue stopped before spawning a new worker")
                    proc=subprocess.Popen([sys.executable,"-u",str(snapshot/"paper/benchmarks2026/worker.py"),"--root",str(ROOT),"--job",str(jpath)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    running[str(i)]={"worker_pid":proc.pid,"started_unix":time.time(),"job":job,
                        "worker_path":str(snapshot/"paper/benchmarks2026/worker.py")};runtime()
                    print(json.dumps({"event":"job_start","index":i+1,"total":len(jobs),"data":job["data"],"method":job["method"],"seed":job["seed"]}),flush=True)
                returncode=proc.wait()
            except BaseException:
                with lock:abort.set()
                raise
            finally:
                if proc is not None:
                    if returncode is None:returncode=proc.wait()
                    with lock:
                        if returncode!=0:abort.set()
                        running.pop(str(i),None);runtime()
        if returncode:
            write(folder/"failure.json",{"index":i,"job":job,"return_code":returncode})
            raise RuntimeError(f"Job failed; inspect {folder/'train.log'}")
        return read(folder/"completed.json")

    def run_one(i,job):
        try:return execute(i,job)
        except Exception:
            with lock:abort.set()
            raise

    failure=None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_one,i,job):(i,job) for i,job in pending}
        for future in as_completed(futures):
            if future.cancelled():continue
            try:row=future.result()
            except CancelledError:continue
            except Exception as error:
                if failure is None:
                    failure=error;i,job=futures[future]
                    write(phase/"queue_failure.json",{"index":i,"job":job,"error":repr(error)})
                    for other in futures:other.cancel()
                continue
            done.append(row)
            write(phase/"progress.json",{"finished":len(done),"planned":len(jobs),"last":row})
            print(json.dumps({"event":"job_finished","finished":len(done),"planned":len(jobs),**row}),flush=True)
    runtime("failed" if failure else "completed")
    if failure is not None:raise failure
    assert len(done)==len(jobs)
    write(phase/"completed.json",{"all_completed":True,"jobs":done,"max_workers":workers,
        "queue_elapsed_seconds":time.time()-started,"summed_training_seconds":sum(r["train_seconds"] for r in done),
        "training_times":"Per-job wall time can include contention; not an isolated efficiency comparison."})


if __name__=="__main__":main()
