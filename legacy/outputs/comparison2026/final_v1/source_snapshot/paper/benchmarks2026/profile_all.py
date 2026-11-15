"""Profile frozen, selected models serially only after all training has exited."""
import argparse
import subprocess
import sys
from pathlib import Path
from .data_metrics import ROOT,OUT,read,write,sha
from .gpu_guard import foreign_python


def main():
    p=argparse.ArgumentParser();p.add_argument("--phase",default="final_v1");args=p.parse_args()
    phase=OUT/args.phase;protocol=read(phase/"protocol.json")
    assert read(phase/"completed.json")["all_completed"]
    if foreign_python(include_project=True):raise RuntimeError("Other GPU Python workloads are active")
    snapshot=phase/"source_snapshot"
    for rel,digest in protocol["source_sha256"].items():assert sha(snapshot/rel)==digest
    for i,job in enumerate(protocol["plan"]["jobs"]):
        folder=ROOT/job["output"]
        if (folder/"isolated_latency.json").exists():
            previous=read(folder/"isolated_latency.json")
            assert previous["isolated"] and previous["selected_checkpoint_sha256"]==sha(folder/"best.pt")
            continue
        with (folder/"latency.log").open("a",encoding="utf-8") as log:
            subprocess.run([sys.executable,"-u",str(snapshot/"paper/benchmarks2026/worker.py"),"--root",str(ROOT),
                "--job",str(phase/"jobs"/f"{i:03d}.json"),"--latency-only"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f"Isolated profiles {i+1}/{len(protocol['plan']['jobs'])}",flush=True)
    write(phase/"profiles_completed.json",{"all_completed":True,"jobs":len(protocol["plan"]["jobs"]),"isolated":True})


if __name__=="__main__":main()
