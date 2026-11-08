"""Parameterization-matched direct MMD objective, not a named-method reproduction."""
import json
import time

import torch

import controlled_suite as s


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    jobs = [(name, "mmd_mlp", seed) for name in ("fire", "general", "rna") for seed in (0, 1, 2)]
    jobs += [("rna", "mgm_mlp", seed) for seed in (0, 1, 2)]
    s.write_json(s.OUT / "mmd_protocol.json", {"created_at": s.utc(), "jobs": jobs,
                "steps": 2000, "batch_size": 256, "data_seed": 0,
                "label": "MMD objective, natural-embedding fixed-training-bandwidth biased V-statistic",
                "not_reproduced": ["KGDM", "RW-Flow", "R-WGF", "Riemannian MeanFlow"],
                "control": "Same constrained MLP initial weights as random-init MGM MLP; no critic; same final-step protocol",
                "source_sha256": {str(p.relative_to(s.ROOT)): s.sha(p) for p in (s.Path(__file__), s.Path(s.__file__))}})
    done = []
    started = time.perf_counter()
    for name, method, seed in jobs:
        print(json.dumps({"event": "control_start", "data": name, "method": method, "seed": seed,
                          "finished": len(done), "planned": len(jobs), "at": s.utc()}), flush=True)
        row = s.run_training(name, method, seed, 2000, [2000], draws=3)
        done.append(row)
        s.write_json(s.OUT / "mmd_progress.json", {"started_jobs": len(done), "finished_jobs": len(done),
                     "planned_jobs": len(jobs), "elapsed_seconds": time.perf_counter() - started, "runs": done})
        print(json.dumps({"event": "control_finished", "finished": len(done), "planned": len(jobs), **row}), flush=True)
    s.summarize()


if __name__ == "__main__":
    main()
