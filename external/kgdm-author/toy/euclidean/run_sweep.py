"""Run the (method × ν × seed) grid for Fig. 5.2 and write a single
results/swissroll/paper-fig/metrics.json that make_paper_fig.py reads.

All hyperparameters come from config.yml. Only the three sweep axes
(method, ν, seed) are varied here, since those *are* the paper grid.
"""
import json
import math
from copy import deepcopy
from pathlib import Path

from train import load_config, train_and_snapshot

METHODS = ["base", "grad"]
NUS = [0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 100.0]
N_SEEDS = 3
COLORS = {"base": "#2ca02c", "grad": "#1f77b4"}


def _nu_label(nu: float) -> str:
    return "inf" if math.isinf(nu) else f"{nu:g}"


def main():
    base_cfg = load_config("config.yml")
    base_seed = int(base_cfg.get("seed", 0))
    n_steps = int(base_cfg["training"]["n_steps"])

    out_dir = (
        Path(base_cfg.get("results_dir", "results"))
        / base_cfg["task"]["name"]
        / "paper-fig"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"sweep: methods={METHODS}  nus={[_nu_label(nu) for nu in NUS]}  "
          f"seeds={N_SEEDS}  n_steps={n_steps}")
    print(f"out: {out_dir}\n")

    results = []
    for method in METHODS:
        per_nu = []
        print(f"### method = {method} ###")
        for nu in NUS:
            print(f"  nu={_nu_label(nu)}")
            c2sts = []
            for seed_idx in range(N_SEEDS):
                cfg = deepcopy(base_cfg)
                cfg["drift"]["update_type"] = method
                cfg["drift"]["nu"] = float(nu)
                cfg["seed"] = base_seed + seed_idx
                # We only need the final-step C2ST.
                cfg["eval"]["snapshot_steps"] = [n_steps]
                snaps = train_and_snapshot(cfg, verbose=False)
                c = float(snaps[-1]["c2st"])
                c2sts.append(c)
                print(f"    seed {seed_idx}: C2ST={c:.4f}", flush=True)
            per_nu.append({"nu": _nu_label(nu), "values_c2st": c2sts})
        results.append({"name": method, "color": COLORS[method], "data": per_nu})

    out = out_dir / "metrics.json"
    with open(out, "w") as f:
        json.dump(
            {
                "config": {
                    "methods": METHODS,
                    "nus": [_nu_label(nu) for nu in NUS],
                    "n_seeds": N_SEEDS,
                    "n_steps": n_steps,
                    "batch_size": base_cfg["training"]["batch_size"],
                    "hidden_sizes": base_cfg["backbone"]["hidden_sizes"],
                    "temp": base_cfg["drift"]["temp"],
                    "max_step": base_cfg["drift"]["max_step"],
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
