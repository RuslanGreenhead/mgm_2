"""Read-only GPU resource metadata, without controlling unrelated processes."""
import argparse
import csv
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def query(fields, kind):
    command = ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--our-pid", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2] / "outputs/paper_icml2026/controlled_independent"
    gpu = query("name,utilization.gpu,memory.used,memory.total", "gpu")
    processes = query("pid,process_name,used_memory", "compute-apps")
    python = [{"pid": int(p[0]), "process_name": p[1], "memory_mib": p[2], "ours": int(p[0]) == args.our_pid}
              for p in processes if len(p) >= 3 and "python" in p[1].lower()]
    event = {"observed_at": datetime.now(timezone.utc).isoformat(), "gpu": gpu, "python_gpu_processes": python,
             "other_python_gpu_processes": sum(not p["ours"] for p in python),
             "interpretation": "Contention invalidates an isolated timing interpretation, not the fixed-sample metrics or RNG stream invariance"}
    with (root / "resource_contention.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    print(json.dumps({"observed_at": event["observed_at"], "other_python_gpu_processes": event["other_python_gpu_processes"], "gpu": gpu}))


if __name__ == "__main__":
    main()
