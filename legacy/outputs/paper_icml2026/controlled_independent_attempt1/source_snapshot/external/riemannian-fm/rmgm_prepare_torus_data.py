"""Prepare Top500 protein and RNA torus datasets for the RFM experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


BASE_URL = "https://files.rcsb.org/download"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--datasets", nargs="*", choices=("top500", "rna"), default=["top500", "rna"])
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    root = Path(args.root)
    for name in args.datasets:
        prepare_one(root / name, args.retries, args.timeout)


def prepare_one(path: Path, retries: int, timeout: float) -> None:
    list_path = path / "list_file.txt"
    if not list_path.exists():
        raise FileNotFoundError(list_path)

    pdb_ids = [item.strip() for item in list_path.read_text(encoding="utf-8").split(",") if item.strip()]
    print(f"\n=== preparing {path.name}: {len(pdb_ids)} PDB ids ===", flush=True)
    failures = []
    for i, pdb_id in enumerate(pdb_ids, start=1):
        target = path / f"{pdb_id}.pdb.gz"
        if target.exists() and target.stat().st_size > 0:
            continue
        url = f"{BASE_URL}/{pdb_id}.pdb.gz"
        ok = download(url, target, retries=retries, timeout=timeout)
        if not ok:
            failures.append(pdb_id)
        if i % 50 == 0 or i == len(pdb_ids):
            print(f"{path.name}: downloaded/checkpoint {i}/{len(pdb_ids)}; failures={len(failures)}", flush=True)

    if failures:
        fail_path = path / "download_failures.txt"
        fail_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"{path.name}: failed downloads written to {fail_path}", flush=True)

    script = "get_torsion_angle.py" if path.name == "top500" else "get_torsion_angles.py"
    print(f"{path.name}: running {script}", flush=True)
    subprocess.run([sys.executable, script], cwd=path, check=True)
    aggregated = path / "aggregated_angles.tsv"
    lines = sum(1 for _ in aggregated.open("r", encoding="utf-8")) if aggregated.exists() else 0
    print(f"{path.name}: wrote {aggregated} with {lines} rows", flush=True)


def download(url: str, target: Path, *, retries: int, timeout: float) -> bool:
    last_error = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                target.write_bytes(response.read())
            return True
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    print(f"failed {url}: {last_error}", flush=True)
    return False


if __name__ == "__main__":
    main()
