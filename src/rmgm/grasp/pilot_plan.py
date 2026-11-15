"""Build an explicit plan; never launch a process or request a GPU implicitly."""

from pathlib import Path
import sys
from types import SimpleNamespace
import json

from . import OUTPUT, ROOT
from .data import PreparedData, sha256, write_json
from .runner import source_hashes, train
from .source_mt import load_source


def build_plan(destination=None):
    manifest = OUTPUT / "data/public_within_object/manifest.json"
    data = PreparedData(manifest, "cpu", True)
    _, _, mt_provenance = load_source()
    run_root = OUTPUT / "public_within_pilot"
    python = str(Path(sys.executable))
    jobs = []
    for seed in (0, 1, 2):
        for method in ("mgm", "rmf_mt", "rfm"):
            out = run_root / method / f"seed{seed}"
            argv = [python, "-m", "rmgm.grasp.cli", "train", "--manifest", str(manifest),
                    "--method", method, "--seed", str(seed), "--iterations", "10000", "--batch", "256",
                    "--width", "128", "--lr", "0.0002", "--critic-steps", "5", "--warmup", "100",
                    "--save-every", "2000", "--device", "cuda", "--cpu-threads", "2", "--gpu-release", "--allow-smoke-data", "--out", str(out)]
            evaluations = []
            for split in ("val", "test"):
                for steps in ((50, 100) if method == "rfm" else (1,)):
                    evaluations.append([python, "-m", "rmgm.grasp.cli", "evaluate",
                         "--checkpoint", str(out / "last.pt"), "--split", split, "--samples", "512",
                         "--collision-samples", "100", "--steps", str(steps), "--eval-seed", "96231",
                         "--device", "cuda", "--cpu-threads", "2", "--gpu-release", "--allow-smoke-data"])
            jobs.append({"method": method, "seed": seed, "train_argv": argv, "evaluate_argvs": evaluations})
    plan = {"protocol": "within-object held-out-pose pilot, NOT unseen-object ACRONYM",
            "requires_explicit_parent_gpu_queue_release": True, "launches_processes": False,
            "manifest": str(manifest), "manifest_sha256": sha256(manifest), "source_sha256": source_hashes(),
            "official_gripper_sha256": sha256(ROOT / "external/acronym-data/public/data/franka_gripper_collision_mesh.stl"),
            "split_seed": 2026, "training_seeds": [0, 1, 2], "positive_grasps": sum(r["n_positive"] for r in data.records),
            "objects": [{"id": r["object_id"], "pose_split_counts": {s: len(v) for s, v in r["pose_splits_source_indices"].items()},
                         "pose_split_sha256": r["pose_split_sha256"]} for r in data.records],
            "selection": "Fixed final 10000 generator/field updates per seed. No early stopping, hyperparameter selection or checkpoint selection on validation/test.",
            "budgets": {"training_jobs": 9, "total_generator_or_field_updates": 90000,
                        "total_mgm_critic_updates": 150300, "total_optimizer_updates": 240300,
                        "cpu_profile_command": "python -m rmgm.grasp.cli profile-cpu",
                        "gpu_runtime": "Unmeasured. First 200 fixed-configuration CUDA steps after release should calibrate the nine-job ETA; no test-based tuning.",
                        "cuda_requirement": "CUDA-enabled PyTorch (existing build reports CUDA 12.1), compatible NVIDIA driver; no CPU-only PyTorch install changes needed."},
            "methods": {"mgm": "Direct conditional MGM, five critic updates/generator update; 1 NFE",
                        "rfm": "Local common-architecture reverse-geodesic RFM, exponential Euler at 50 and 100 NFE",
                        "rmf_mt": mt_provenance},
            "metrics": ["Per-object unbiased pose MMD2 with fixed chordal-product kernels", "SO3 validity residuals",
                        "Released-author static geometric feasibility pose fraction and object any-at-20",
                        "Separate collision fraction for official Franka collision mesh", "Measured optimizer updates, wall time, peak CUDA bytes"],
            "physical_success_rate": None,
            "physical_evaluation": "Not evaluated. Historical FleX target labels are not predictions of generated grasp success.",
            "visualization": "Each object and all three seeds: reference/MGM/RMF-MT/RFM-50/RFM-100, actual Franka meshes, first 12 draws unfiltered. Full range covers every evaluated draw; mesh-centered detail discloses off-frame counts. Identical camera and shared bounds within each row; PNG+PDF+provenance JSON.",
            "jobs": jobs}
    destination = destination or OUTPUT / "pilot_plan_ready.json"
    write_json(destination, plan)
    return str(destination)


def profile_cpu(destination=None, iterations=100):
    destination = Path(destination or OUTPUT / "cpu_profile")
    rows = {}
    for method in ("mgm", "rmf_mt", "rfm"):
        args = SimpleNamespace(command="train", manifest=str(OUTPUT / "data/public_within_object/manifest.json"),
                 method=method, seed=0, device="cpu", gpu_release=False, cpu_threads=2, allow_smoke_data=True,
                 iterations=iterations, batch=256, width=128, lr=2e-4, translation_weight=1., critic_steps=5,
                 warmup=100, save_every=iterations, resume=False, out=str(destination / method))
        checkpoint = train(args)
        status = json.loads((Path(checkpoint).parent / "train_status.json").read_text())
        rows[method] = {"measured_cpu_seconds": status["training_seconds"],
                        "warmup_seconds": status["warmup_seconds"],
                        "cpu_seconds_per_update": status["seconds_per_iteration_excluding_warmup"],
                        "linear_10000_update_cpu_seconds": status["warmup_seconds"]+10000*status["seconds_per_iteration_excluding_warmup"]}
    summary = {"stage": "CPU runtime calibration only; no validation/test sampling or quality selection",
               "iterations": iterations, "batch": 256, "width": 128, "threads": 2, "gpu_used": False,
               "methods": rows, "three_seed_total_cpu_seconds_linear": 3*sum(r["linear_10000_update_cpu_seconds"] for r in rows.values()),
               "gpu_estimate": "Not measured; CPU timing is not a GPU speedup prediction"}
    write_json(destination / "summary.json", summary)
    return str(destination / "summary.json")
