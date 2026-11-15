"""Real-data CPU integration and exact-resume verification, not a benchmark."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from . import ROOT
from .data import write_json
from .runner import evaluate, train
from .visualization import visualize


def assert_state_equal(a, b, context="checkpoint"):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            raise AssertionError(f"Resume tensor differs: {context}")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"Resume keys differ: {context}")
        for key in a:
            assert_state_equal(a[key], b[key], f"{context}.{key}")
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise AssertionError(f"Resume lengths differ: {context}")
        for index, (left, right) in enumerate(zip(a, b)):
            assert_state_equal(left, right, f"{context}.{index}")
    else:
        import numpy as np
        if not np.array_equal(a, b):
            raise AssertionError(f"Resume value differs: {context}")


def smoke(manifest, destination, iterations=30):
    if iterations < 3:
        raise ValueError("Need at least three CPU integration iterations")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    dataset_manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    split = "val" if dataset_manifest.get("protocol") == "within_object_heldout_pose" else "test"
    results, metrics = {}, []
    for method in ("mgm", "rmf_mt", "rfm", "rmf_ccadapt", "glsd"):
        args = SimpleNamespace(command="train", manifest=str(manifest), method=method, seed=0,
                               device="cpu", gpu_release=False, cpu_threads=2, allow_smoke_data=True,
                               iterations=iterations, batch=64, width=64, lr=2e-4, translation_weight=1.,
                               critic_steps=2, warmup=5, save_every=iterations, resume=False,
                               out=str(destination / method / "uninterrupted"))
        uninterrupted = train(args)
        args.out = str(destination / method / "resumed")
        args.iterations = 2
        partial = train(args)
        args.iterations = iterations
        args.resume = True
        resumed = train(args)
        a = torch.load(uninterrupted, map_location="cpu", weights_only=False)
        b = torch.load(resumed, map_location="cpu", weights_only=False)
        for key in ("step", "model", "critic", "optimizer", "critic_optimizer", "rng", "torch_rng", "python_rng", "numpy_rng", "cuda_rng"):
            assert_state_equal(a[key], b[key], f"{method}.{key}")
        ev = SimpleNamespace(command="evaluate", checkpoint=uninterrupted, device="cpu", gpu_release=False,
                             cpu_threads=2, allow_smoke_data=True, split=split, samples=64, steps=1,
                             eval_seed=96231, collision_samples=32, skip_collision=False,
                             official_gripper=str(ROOT / "external/acronym-data/public/data/franka_gripper_collision_mesh.stl"),
                             out=str(destination / method / f"eval_{split}_nfe1"))
        method_metrics = []
        for steps in ((50, 100) if method == "rfm" else (1,)):
            ev.steps = steps
            ev.out = str(destination / method / f"eval_{split}_nfe{steps}")
            method_metrics.append(evaluate(ev))
        from .audit import audit_report
        from .data import PreparedData
        audit_data = PreparedData(manifest, "cpu", True)
        for metric in method_metrics:
            audit_report(json.loads(Path(metric).read_text()), audit_data)
        if method in ("mgm", "rmf_mt", "rfm"):
            metrics.extend(method_metrics)
        status = json.loads((Path(uninterrupted).parent / "train_status.json").read_text())
        results[method] = {"exact_resume_model_optimizer_and_all_rng": True, "cpu_seconds": a["training_seconds"],
                           "iterations": iterations, "batch": args.batch, "width": args.width,
                           "cpu_threads": 2, "parameter_count": a["config"]["model_parameter_count"],
                           "critic_parameter_count": a["config"]["critic_parameter_count"],
                           "seconds_per_iteration_including_warmup": a["training_seconds"]/iterations,
                           "linear_10k_cpu_estimate_seconds_same_smoke_config": a["training_seconds"]*10000/iterations,
                           "metrics_paths": method_metrics, "final_losses": status["losses"]}
    figures = []
    for record in dataset_manifest["objects"]:
        if record["object_id"] not in {o["object_id"] for o in json.loads(Path(metrics[0]).read_text())["objects"]}:
            continue
        figures.append(visualize(metrics, destination / f"{record['object_id']}_comparison.png", count=12, object_id=record["object_id"]))
    summary = {"stage": "Real ACRONYM two-object CPU integration; NOT a training comparison or physical success study",
               "protocol": dataset_manifest.get("protocol", "object_heldout"), "integration_eval_split": split,
               "primary_test_quality_peeked": False if split == "val" else "Legacy object-transfer smoke only",
               "gpu_used": False, "physical_success_rate": None, "runs": results,
               "figures": figures, "gpu_timing": "Not measured; parent queue release required",
               "performance_limitations": "CPU timings use width64, batch64 and two MGM critic updates; do not extrapolate to the width128/batch256/five-critic GPU schedule."}
    write_json(destination / "summary.json", summary)
    return str(destination / "summary.json")
