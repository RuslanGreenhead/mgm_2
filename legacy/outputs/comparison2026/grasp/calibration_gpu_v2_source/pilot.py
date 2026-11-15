"""One sequential, fail-closed, resumable pilot. Never spawns training workers."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from . import OUTPUT
from .audit import audit_report
from .data import PreparedData, sha256, write_json
from .runner import check_device, evaluate, source_hashes, train
from .source_mt import load_source
from .visualization import GRIPPER, visualize


def parsed(argv, command):
    from .cli import parser
    if argv[1:3] != ["-m", "paper.benchmarks2026.grasp.cli"] or argv[3] != command:
        raise ValueError("Unexpected frozen-plan command")
    return parser().parse_args(argv[3:])


def load_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan["source_sha256"] != source_hashes() or plan["manifest_sha256"] != sha256(plan["manifest"]):
        raise ValueError("Frozen plan source/data hashes changed; do not resume with a different implementation")
    if plan["official_gripper_sha256"] != sha256(GRIPPER):
        raise ValueError("Official gripper mesh changed")
    if [(j["method"], j["seed"]) for j in plan["jobs"]] != [(m, s) for s in (0, 1, 2) for m in ("mgm", "rmf_mt", "rfm")]:
        raise ValueError("Exactly nine ordered MGM/RMF-MT/RFM jobs and seeds 0/1/2 required")
    for job in plan["jobs"]:
        args = parsed(job["train_argv"], "train")
        expected = {"method": job["method"], "seed": job["seed"], "iterations": 10000, "batch": 256,
                    "width": 128, "lr": .0002, "critic_steps": 5, "warmup": 100, "save_every": 2000,
                    "translation_weight": 1., "device": "cuda", "cpu_threads": 2, "allow_smoke_data": True}
        if any(getattr(args, k) != v for k, v in expected.items()) or args.manifest != plan["manifest"]:
            raise ValueError("Frozen training budget/configuration differs from the registered pilot")
        if not Path(args.out).resolve().is_relative_to(OUTPUT.resolve()):
            raise ValueError("Pilot outputs must remain in the owned grasp subtree")
        pairs = []
        for argv in job["evaluate_argvs"]:
            ev = parsed(argv, "evaluate")
            pairs.append((ev.split, ev.steps))
            if (ev.checkpoint != str(Path(args.out)/"last.pt") or ev.samples != 512 or ev.collision_samples != 100 or
                    ev.eval_seed != 96231 or ev.skip_collision or ev.device != "cuda" or ev.cpu_threads != 2 or
                    not ev.allow_smoke_data or ev.out or Path(ev.official_gripper) != GRIPPER):
                raise ValueError("Frozen evaluation configuration changed")
        if pairs != [(s, n) for s in ("val", "test") for n in ((50, 100) if args.method == "rfm" else (1,))]:
            raise ValueError("Pilot requires both held-out splits and both RFM 50/100 evaluations")
    data = PreparedData(plan["manifest"], "cpu", True)
    if not data.within_object or len(data.records) != 2 or data.manifest["seed"] != 2026:
        raise ValueError("Only the real two-object within-object held-out-pose pilot is authorized")
    _, _, provenance = load_source()
    if plan["methods"]["rmf_mt"] != provenance:
        raise ValueError("Native-source RMF-MT provenance changed")
    return plan, data


def preflight(path):
    plan, data = load_plan(path)
    report = {"status": "CPU_READY_AWAITING_EXPLICIT_PARENT_GPU_RELEASE", "gpu_used": False,
              "plan": str(Path(path).resolve()), "plan_sha256": sha256(path),
              "manifest_sha256": plan["manifest_sha256"], "source_sha256": source_hashes(),
              "source_verification": data.source_verification,
              "pose_partitions": [{"object_id": r["object_id"], "counts": {s: len(v) for s, v in r["pose_splits_source_indices"].items()},
                                  "pairwise_disjoint": True, "positive_only_complete_coverage": True} for r in data.records],
              "jobs": 9, "evaluations": sum(len(j["evaluate_argvs"]) for j in plan["jobs"]),
              "physical_success_rate": None}
    destination = OUTPUT / "verification_ready/preflight.json"
    write_json(destination, report)
    return str(destination)


def checkpoint_step(path):
    return torch.load(path, map_location="cpu", weights_only=False)["step"] if Path(path).exists() else 0


def evaluation_complete(path, args):
    if not Path(path).exists():
        return False
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {k: v for k, v in vars(args).items() if k != "func"}
    if (report.get("evaluation_config") != expected or report["checkpoint_sha256"] != sha256(args.checkpoint) or
            report["evaluation_source_sha256"] != source_hashes() or report["training_iteration"] != 10000):
        raise ValueError("Existing evaluation is incompatible with the frozen final checkpoint")
    if len(report["objects"]) != 2 or report["physical_success_rate"] is not None:
        raise ValueError("Incomplete or invalid final evaluation")
    for obj in report["objects"]:
        if sha256(obj["samples_path"]) != obj["samples_sha256"] or "static_geometry" not in obj:
            raise ValueError("Existing evaluation sample/proxy evidence changed")
    return True


def runtime_estimate(plan):
    estimates = {}
    for job in plan["jobs"]:
        if job["seed"] != 0:
            continue
        status_path = Path(parsed(job["train_argv"], "train").out)/"train_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            estimates[job["method"]] = {"measured_updates": status["step"],
                "seconds_per_update": status["seconds_per_iteration_excluding_warmup"],
                "linear_10k_training_seconds": status["warmup_seconds"]+10000*status["seconds_per_iteration_excluding_warmup"]}
    write_json(OUTPUT/"public_within_pilot/runtime_estimate.json", {
        "method_estimates": estimates, "three_seed_training_seconds": 3*sum(v["linear_10k_training_seconds"] for v in estimates.values()) if len(estimates) == 3 else None,
        "limitation": "Linear measured CUDA training estimate only, excludes evaluation, FCL and rendering; no quality tuning"})


def aggregate(plan, reports):
    if len(reports) != 24:
        raise ValueError("All 24 fixed final evaluations must finish before aggregation")
    data = PreparedData(plan["manifest"], "cpu", True)
    audits = []
    rows, reference_baselines = [], {}
    for path in reports:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        audits.append({"metrics_path": str(path), "metrics_sha256": sha256(path), "objects": audit_report(report, data)})
        for baseline in report["reference_baseline"]:
            key = (baseline["split"], baseline["object_id"])
            if key in reference_baselines and reference_baselines[key] != baseline:
                raise ValueError("Held-out reference baseline changed across methods/seeds/NFE")
            reference_baselines[key] = baseline
        for obj in report["objects"]:
            geometry = obj["static_geometry"]
            reference = obj["heldout_reference_static_geometry"]
            rows.append({"method": report["method"], "seed": report["seed"], "nfe": report["nfe"], "split": report["split"],
                         "object_id": obj["object_id"], "mmd2": obj["distribution"]["mmd2_unbiased_fixed_chordal_product_kernel"],
                         "static_proxy_pose_fraction": geometry["author_static_feasible_pose_fraction"],
                         "static_proxy_any_at_20": geometry["author_static_feasible_any_at_20"],
                         "official_franka_collision_fraction": geometry["official_franka_mesh_collision_fraction"],
                         "heldout_reference_static_proxy_pose_fraction": reference["author_static_feasible_pose_fraction"],
                         "heldout_reference_static_proxy_any_at_20": reference["author_static_feasible_any_at_20"],
                         "heldout_reference_official_franka_collision_fraction": reference["official_franka_mesh_collision_fraction"],
                         "heldout_reference_proxy_count": len(obj["heldout_reference_proxy_source_indices"]),
                         "so3_orthogonality_max": obj["validity"]["orthogonality_max"],
                         "so3_determinant_error_max": obj["validity"]["determinant_error_max"],
                         "sampling_seconds": obj["sampling_seconds"], "metrics_path": str(path), "metrics_sha256": sha256(path)})
    groups = []
    for split in ("val", "test"):
        for method, nfe in (("mgm", 1), ("rmf_mt", 1), ("rfm", 50), ("rfm", 100)):
            selected = [r for r in rows if (r["split"], r["method"], r["nfe"]) == (split, method, nfe)]
            if {(r["seed"], r["object_id"]) for r in selected} != {(s, o["id"]) for s in (0, 1, 2) for o in plan["objects"]} or len(selected) != 6:
                raise ValueError("Missing or duplicate final per-object/seed metrics")
            means = [np.mean([r["mmd2"] for r in selected if r["seed"] == s]) for s in (0, 1, 2)]
            group = {"split": split, "method": method, "nfe": nfe, "seed_object_mean_mmd2": means,
                     "mean_mmd2": float(np.mean(means)), "seed_std_mmd2_ddof1": float(np.std(means, ddof=1))}
            for metric in ("static_proxy_pose_fraction", "static_proxy_any_at_20", "official_franka_collision_fraction", "sampling_seconds"):
                values = [float(np.mean([r[metric] for r in selected if r["seed"] == s])) for s in (0, 1, 2)]
                group[metric] = {"seed_object_means": values, "mean": float(np.mean(values)), "seed_std_ddof1": float(np.std(values, ddof=1))}
            groups.append(group)
    figures = []
    for seed in (0, 1, 2):
        paths = [p for p in reports if (r := json.loads(Path(p).read_text(encoding="utf-8")))["seed"] == seed and r["split"] == "test"]
        for record in plan["objects"]:
            image = visualize(paths, OUTPUT/f"figures/{record['id']}_seed{seed}.png", object_id=record["id"])
            figures.append({"png": image, "pdf": str(Path(image).with_suffix(".pdf")), "provenance": str(Path(image).with_suffix(".json"))})
    runs = []
    for job in plan["jobs"]:
        directory = Path(parsed(job["train_argv"], "train").out)
        status = json.loads((directory/"train_status.json").read_text(encoding="utf-8"))
        runs.append({"method": job["method"], "seed": job["seed"], **status})
    summary = {"status": "COMPLETE_FIXED_FINAL_GPU_PILOT", "protocol": plan["protocol"], "plan_source_sha256": plan["source_sha256"],
               "physical_success_rate": None, "selection": plan["selection"], "rows": rows, "seed_summaries": groups,
               "reference_baseline": list(reference_baselines.values()),
               "reference_baseline_note": "First 100 unfiltered held-out reference poses per object/split, same static proxy; FleX-positive labels do not imply proxy passes. Fixed indices shared by all methods/seeds/NFE.",
               "runs": runs, "figures": figures, "limitations": ["Only two known public objects; no unseen-object generalization",
               "Static surface collision/overlap proxy is separate from distribution MMD and is not physical success",
               "RMF-MT uses pinned native loss and PCGrad with common local architecture/path adapters, not native SDF reproduction",
               "Three seeds; fixed update budget, not equal optimizer work or a convergence claim"]}
    write_json(OUTPUT/"final_summary.json", summary)
    write_json(OUTPUT/"final_audit.json", {"status": "PASSED", "physical_success_evaluated": False, "reports": audits})
    with (OUTPUT/"final_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(OUTPUT/"final_summary.json")


def run_pilot(path, gpu_release=False, calibrate_only=False):
    check_device("cuda", gpu_release)
    plan, _ = load_plan(path)
    reports = []
    state_path = OUTPUT/"public_within_pilot/pilot_status.json"
    for job in plan["jobs"]:
        if calibrate_only and job["seed"] != 0:
            continue
        args = parsed(job["train_argv"], "train")
        args.gpu_release, args.resume = gpu_release, True
        checkpoint = Path(args.out)/"last.pt"
        write_json(state_path, {"status": "TRAINING", "method": job["method"], "seed": job["seed"], "plan_sha256": sha256(path)})
        if calibrate_only and checkpoint_step(checkpoint) < 100:
            args.iterations = 100
            train(args)
            shutil.copyfile(checkpoint, checkpoint.parent/"calibration_step100.pt")
            shutil.copyfile(checkpoint.parent/"train_status.json", checkpoint.parent/"calibration_step100_status.json")
        if job["seed"] == 0 and checkpoint_step(checkpoint) < 200:
            args.iterations = 200
            train(args)
        runtime_estimate(plan)
        if calibrate_only:
            continue
        args.iterations = 10000
        train(args)
        if checkpoint_step(checkpoint) != 10000:
            raise ValueError("Only the exact final 10000-update checkpoint may be evaluated")
        for argv in job["evaluate_argvs"]:
            ev = parsed(argv, "evaluate")
            ev.gpu_release = gpu_release
            metrics = Path(args.out)/f"eval_{ev.split}_nfe{ev.steps}/metrics.json"
            write_json(state_path, {"status": "EVALUATING", "method": job["method"], "seed": job["seed"], "split": ev.split, "nfe": ev.steps})
            check_device(ev.device, gpu_release)
            if not evaluation_complete(metrics, ev):
                evaluate(ev)
            reports.append(metrics)
    if calibrate_only:
        write_json(state_path, {"status": "CALIBRATED_NO_QUALITY_EVALUATION", "updates_per_method_seed0": 200})
        return str(state_path)
    result = aggregate(plan, reports)
    runtime_estimate(plan)
    write_json(state_path, {"status": "COMPLETE", "final_summary": result, "plan_sha256": sha256(path)})
    return result
