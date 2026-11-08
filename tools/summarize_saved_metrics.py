"""Reaggregate recorded metrics using JSON only, without samples, weights or GPU."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/comparison2026"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    protocol = read(BASE / "final_v1/protocol.json")
    groups, completed = defaultdict(dict), 0
    for job in protocol["plan"]["jobs"]:
        folder = ROOT / job["output"].replace("\\", "/")
        if not (folder / "completed.json").is_file():
            continue
        done = read(folder / "completed.json")
        assert done["steps"] == job["steps"] and done["test_evaluations"] == 3
        rows = read(folder / f"test_step{done['selected_step']}_metrics.json")
        assert sorted(r["draw"] for r in rows) == [0, 1, 2]
        assert all(r["model_forward_calls"] == 1 and r["split"] == "test" for r in rows)
        key = job["data"], job["method"]
        assert job["seed"] not in groups[key]
        groups[key][job["seed"]] = mean(r["kernel_mmd2_v"] for r in rows)
        completed += 1
    computed = {}
    print("Main test: biased kernel MMD squared x1000; draws averaged within training seed")
    print("| Dataset | Method | Seeds | Mean | Seed SD |")
    print("|---|---|---:|---:|---:|")
    for key, seeds in sorted(groups.items()):
        values = list(seeds.values())
        assert sorted(seeds) == [0, 1, 2]
        computed[key] = mean(values), stdev(values)
        print(f"| {key[0]} | {key[1]} | {len(values)} | {1000*mean(values):.3f} | {1000*stdev(values):.3f} |")
    grasp = defaultdict(lambda: defaultdict(list))
    for row in read(BASE / "grasp/final_summary.json")["rows"]:
        if row["split"] == "test":
            grasp[row["method"], row["nfe"]][row["seed"]].append(row)
    print("\nSE(3) test: different, unbiased product-kernel MMD squared x1000")
    print("| Method | NFE | MMD mean | Seed SD | Full collision percent |")
    print("|---|---:|---:|---:|---:|")
    for (method, nfe), seeds in sorted(grasp.items()):
        assert sorted(seeds) == [0, 1, 2] and all(len(rows) == 2 for rows in seeds.values())
        values = [mean(r["mmd2"] for r in rows) for rows in seeds.values()]
        collision = [mean(r["official_franka_collision_fraction"] for r in rows) for rows in seeds.values()]
        print(f"| {method} | {nfe} | {1000*mean(values):.3f} | {1000*stdev(values):.3f} | {100*mean(collision):.2f} |")
    if args.check:
        partial = read(BASE / "paused_2026_10_26/partial_audit_manifest.json")
        assert completed == partial["completed_test_jobs_verified"] == 18
        assert len(computed) == len(partial["summary"]) == 6
        for row in partial["summary"]:
            mu, sd = computed[row["data"], row["method"]]
            assert math.isclose(mu, row["kernel_mmd2_v_mean"], abs_tol=1e-12)
            assert math.isclose(sd, row["kernel_mmd2_v_seed_sd"], abs_tol=1e-12)
        assert len(grasp) == 4 and read(BASE / "grasp/final_audit.json")["status"] == "PASSED"
        print("\nPASS: 18 jobs / 54 test draws / 6 main aggregates; 4 grasp method-NFE aggregates.")
        print("Recorded audits are evidence only; no raw-sample or physical-success audit was rerun.")


if __name__ == "__main__":
    main()
