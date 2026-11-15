"""Read-only GPU contention guard: never suspends or kills another project."""
import csv
from concurrent.futures import CancelledError
import io
import json
import os
import subprocess
import time


def foreign_python(include_project=False):
    result=subprocess.run(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],
        check=True,capture_output=True,text=True)
    blocked=[]
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row)!=2:continue
        pid,name=(v.strip() for v in row)
        lower=name.lower().replace("\\","/")
        if int(pid)==os.getpid():continue
        if lower.endswith(("/python.exe","/pythonw.exe")) and (include_project or "/riemmanian midpoint generative models/" not in lower):
            blocked.append({"pid":int(pid),"process":name})
    return blocked


def wait_until_free(stop_event=None):
    while True:
        if stop_event is not None and stop_event.is_set():raise CancelledError("Queue stopped after another worker failed")
        blocked=foreign_python()
        if not blocked:return
        print(json.dumps({"event":"waiting_for_foreign_gpu_training","processes":blocked,
            "policy":"User requested waiting for ParetoSR; no other process is modified."}),flush=True)
        if stop_event is None:time.sleep(60)
        else:stop_event.wait(60)


if __name__=="__main__":print(json.dumps({"blocking_gpu_python":foreign_python()},indent=2))
