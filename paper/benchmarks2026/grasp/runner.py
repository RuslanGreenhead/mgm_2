"""Single-process training, exact-state resumes, and fixed held-out evaluation."""

import json
import importlib.metadata
import csv
import io
import os
import random
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import OUTPUT
from .data import PreparedData, sha256, write_json
from .evaluation import distribution_metrics, static_geometry
from .geometry import rotation_residual
from .models import Field, Generator, flow_objective, generate, mgm_objective


def source_hashes():
    return {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}


def dependency_versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "numpy", "scipy", "roma", "h5py", "trimesh", "python-fcl", "rtree", "matplotlib")}


def make_rng(seed, device):
    return torch.Generator(device=device).manual_seed(seed)


def check_device(device, gpu_release):
    if not str(device).startswith("cuda"):
        return {"device": str(device), "gpu_inspected": False}
    if not gpu_release:
        raise ValueError("GPU queue is locked. Pass --gpu-release ONLY after the parent explicitly grants the shared queue.")
    # WDDM reports desktop C+G contexts here too; only foreign Python is training.
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=15, check=True)
        processes = []
        desktop = {"dwm.exe", "explorer.exe", "searchapp.exe", "lockapp.exe", "textinputhost.exe",
                   "shellexperiencehost.exe", "phoneexperiencehost.exe", "video.ui.exe", "avpui.exe",
                   "redshieldvpn.exe", "systemsettings.exe", "codex-computer-use-swift.exe",
                   "chatgpt.exe", "steamwebhelper.exe", "gcc.exe", "client.exe"}
        for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
            if not row or not any(v.strip() for v in row):
                continue
            if len(row) != 2:
                raise ValueError("Expected NVIDIA PID/process_name CSV")
            pid, name = int(row[0].strip()), row[1].strip()
            base = name.lower().replace("\\", "/").rsplit("/", 1)[-1]
            python = bool(re.fullmatch(r"python(?:w|\d+(?:\.\d+)*)?(?:\.exe)?", base))
            classification = ("own_process" if pid == os.getpid() else "foreign_python" if python else
                              "desktop_gui" if base in desktop else "unknown_nonpython")
            processes.append({"pid": pid, "process_name": name, "classification": classification})
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=True)
        gpu_rows = []
        for row in csv.reader(io.StringIO(gpu.stdout), skipinitialspace=True):
            if len(row) != 5:
                raise ValueError("Expected NVIDIA GPU memory/utilization CSV")
            gpu_rows.append(dict(zip(("name", "driver", "memory_used_mib", "memory_total_mib", "utilization_percent"),
                                     [value.strip() for value in row])))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError("GPU ownership could not be verified; refusing CUDA") from exc
    snapshot = {"device": str(device), "gpu_inspected": True, "processes": processes,
                "raw_compute_apps_csv": result.stdout, "raw_gpu_csv": gpu.stdout, "gpus": gpu_rows,
                "unknown_entries": [p for p in processes if p["classification"] == "unknown_nonpython"],
                "guard_scope": "Block every foreign Python/pythonw GPU PID, including this project's other workers; WDDM desktop/unknown entries logged, not classified as training"}
    print(json.dumps({"event": "gpu_guard_snapshot", **snapshot}), flush=True)
    foreign = sorted(p["pid"] for p in processes if p["classification"] == "foreign_python")
    if foreign:
        raise RuntimeError(f"Foreign GPU Python process(es) {foreign}; refusing train/evaluate. No process was stopped.")
    return snapshot


def check_cpu_threads(threads):
    if threads not in (1, 2):
        raise ValueError("This pilot is restricted to one or two CPU threads")
    torch.set_num_threads(threads)


def assert_resume_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        equal = torch.equal(actual.cpu(), expected.cpu())
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError("Resume state keys differ")
        for key in expected:
            assert_resume_state(actual[key], expected[key])
        return
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise ValueError("Resume state lengths differ")
        for left, right in zip(actual, expected):
            assert_resume_state(left, right)
        return
    else:
        equal = np.array_equal(actual, expected)
    if not equal:
        raise ValueError("Restored model/optimizer/RNG state differs from checkpoint")


def train(args):
    guard = check_device(args.device, args.gpu_release)
    check_cpu_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = PreparedData(args.manifest, args.device, args.allow_smoke_data)
    out = Path(args.out or OUTPUT / "runs" / args.method / f"seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ("func",)}
    config.update({"manifest_sha256": sha256(args.manifest), "source_sha256": source_hashes(),
                   "dependency_versions": dependency_versions(),
                   "gpu_guard": guard, "source_verification": data.source_verification,
                   "local_implementation": True, "author_architecture_reproduction": False,
                   "geometry": "SO3 x R3 product metric; body angular and object-frame translation velocities",
                   "method_note": "MGM hidden-fold tangent minimax; RFM reverse geodesic regression; RMF-MT pinned two-loss/PCGrad adaptation distinct from RMF-CC; G-LSD endpoint JVP + FM regularizer",
                   "data_status": data.manifest.get("protocol_label", "PUBLIC_TWO_OBJECT_INTEGRATION_ONLY") if data.manifest["smoke_only"] else "licensed_object_heldout",
                   "physical_success_evaluation": False})
    config["method_label"] = {"rmf_ccadapt": "RMF-CC adaptation", "rmf_mt": "RMF-MT source-loss adaptation",
                              "mgm": "MGM", "rfm": "RFM", "glsd": "G-LSD local adaptation"}[args.method]
    pcgrad = None
    if args.method == "rmf_mt":
        from .source_mt import load_source
        _, pcgrad, provenance = load_source()
        config["method_provenance"] = provenance
    model = Generator(args.width) if args.method == "mgm" else Field(args.width)
    model.to(data.device)
    critic = Field(args.width).to(data.device) if args.method == "mgm" else None
    config["model_parameter_count"] = sum(p.numel() for p in model.parameters())
    config["critic_parameter_count"] = sum(p.numel() for p in critic.parameters()) if critic else 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    copt = torch.optim.Adam(critic.parameters(), lr=args.lr) if critic else None
    rngs = {"data": make_rng(10000+args.seed, data.device), "generator": make_rng(20000+args.seed, data.device)}
    start, elapsed_before, warmup_seconds, peak_before = 0, 0., 0., 0
    checkpoint_path = out / "last.pt"
    if checkpoint_path.exists():
        if not args.resume:
            raise ValueError(f"Run already exists: {out}; use --resume or a new --out")
        saved = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        for key in ("method", "seed", "device", "cpu_threads", "width", "batch", "lr", "critic_steps", "warmup", "translation_weight", "manifest_sha256", "source_sha256", "dependency_versions"):
            if saved["config"][key] != config[key]:
                raise ValueError(f"Resume mismatch: {key}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if critic:
            critic.load_state_dict(saved["critic"])
            copt.load_state_dict(saved["critic_optimizer"])
        for key, rng in rngs.items():
            rng.set_state(saved["rng"][key].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        if data.device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), data.device)
        restored = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "critic": critic.state_dict() if critic else None, "critic_optimizer": copt.state_dict() if copt else None,
                    "rng": {k: r.get_state() for k, r in rngs.items()}, "torch_rng": torch.get_rng_state(),
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "cuda_rng": torch.cuda.get_rng_state(data.device) if data.device.type == "cuda" else None}
        assert_resume_state(restored, {key: saved[key] for key in restored})
        write_json(out/f"resume_step{saved['step']}.json", {"restored_step": saved["step"], "verified": True,
                   "checks": list(restored), "device": str(data.device), "checkpoint_sha256": sha256(checkpoint_path)})
        print(json.dumps({"event": "resume_state_verified", "method": args.method, "step": saved["step"], "device": str(data.device)}), flush=True)
        start, elapsed_before = saved["step"], saved["training_seconds"]
        warmup_seconds = saved.get("warmup_seconds", 0.)
        peak_before = saved.get("peak_cuda_bytes") or 0
        if start > args.iterations:
            raise ValueError("Checkpoint exceeds requested fixed update budget")
        del saved
        del restored
    write_json(out / "config.json", config)
    if data.device.type == "cuda":
        torch.cuda.synchronize(data.device)
        torch.cuda.reset_peak_memory_stats(data.device)
    wall_start = time.perf_counter()
    losses = {}

    def update(loss, module, opt):
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; no checkpoint accepted")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 10., error_if_nonfinite=True)
        opt.step()
        return float(norm)

    if critic and start == 0:
        warmup_start = time.perf_counter()
        for _ in range(args.warmup):
            loss = mgm_objective(model, critic, data, rngs["data"], args.batch, True, args.translation_weight)
            update(loss, critic, copt)
        if data.device.type == "cuda":
            torch.cuda.synchronize(data.device)
        warmup_seconds = time.perf_counter()-warmup_start
    for iteration in range(start+1, args.iterations+1):
        if critic:
            for _ in range(args.critic_steps):
                loss = mgm_objective(model, critic, data, rngs["data"], args.batch, True, args.translation_weight)
                cnorm = update(loss, critic, copt)
            losses["critic_loss"] = float(loss)
            losses["critic_grad_norm"] = cnorm
            critic.requires_grad_(False)
            loss = mgm_objective(model, critic, data, rngs["generator"], args.batch, False, args.translation_weight)
            gnorm = update(loss, model, optimizer)
            critic.requires_grad_(True)
        elif args.method == "rmf_mt":
            from .source_mt import objective_parts
            l1, l2 = objective_parts(model, data, rngs["data"], args.batch)
            if not torch.isfinite(l1+l2):
                raise FloatingPointError("Non-finite source RMF-MT terms")
            pcgrad(optimizer).pc_backward([l1, l2])
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True))
            optimizer.step()
            loss = l1 + l2
            losses.update({"native_loss_1": float(l1), "native_loss_2": float(l2)})
        else:
            loss = flow_objective(args.method, model, data, rngs["data"], args.batch, args.translation_weight)
            gnorm = update(loss, model, optimizer)
        losses.update({"loss": float(loss), "grad_norm": gnorm})
        if iteration % args.save_every == 0 or iteration == args.iterations:
            if data.device.type == "cuda":
                torch.cuda.synchronize(data.device)
            elapsed = elapsed_before + time.perf_counter()-wall_start
            state = {"step": iteration, "config": config, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "critic": critic.state_dict() if critic else None,
                     "critic_optimizer": copt.state_dict() if copt else None,
                     "rng": {k: r.get_state() for k, r in rngs.items()}, "torch_rng": torch.get_rng_state(),
                     "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                     "cuda_rng": torch.cuda.get_rng_state(data.device) if data.device.type == "cuda" else None,
                     "training_seconds": elapsed}
            state["warmup_seconds"] = warmup_seconds
            state["peak_cuda_bytes"] = max(peak_before, torch.cuda.max_memory_allocated(data.device)) if data.device.type == "cuda" else None
            temporary = out / "last.tmp.pt"
            torch.save(state, temporary)
            temporary.replace(checkpoint_path)
            write_json(out / "train_status.json", {"step": iteration, "training_seconds": elapsed, "losses": losses,
                        "warmup_seconds": warmup_seconds,
                        "seconds_per_iteration_excluding_warmup": (elapsed-warmup_seconds)/iteration,
                        "generator_updates": iteration, "critic_updates": args.warmup+iteration*args.critic_steps if critic else 0,
                        "peak_cuda_bytes": state["peak_cuda_bytes"],
                        "checkpoint_sha256": sha256(checkpoint_path)})
            print(json.dumps({"step": iteration, "seconds": round(elapsed, 3), **losses}), flush=True)
    return str(checkpoint_path)


def load_run(checkpoint, device, allow_smoke):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = saved["config"]
    if cfg["source_sha256"] != source_hashes():
        raise ValueError("Implementation changed since training; evaluation refused")
    if cfg["dependency_versions"] != dependency_versions():
        raise ValueError("Dependency versions changed since training")
    if sha256(cfg["manifest"]) != cfg["manifest_sha256"]:
        raise ValueError("Dataset manifest changed since training")
    data = PreparedData(cfg["manifest"], device, allow_smoke)
    model = Generator(cfg["width"]) if cfg["method"] == "mgm" else Field(cfg["width"])
    model.to(device).load_state_dict(saved["model"])
    model.eval()
    return cfg, data, model, saved["step"]


def evaluate(args):
    guard = check_device(args.device, args.gpu_release)
    if not args.skip_collision and (args.samples < 20 or args.collision_samples < 20):
        raise ValueError("Any-at-20 geometry evaluation requires at least 20 generated poses")
    check_cpu_threads(args.cpu_threads)
    cfg, data, model, iteration = load_run(args.checkpoint, args.device, args.allow_smoke_data)
    if args.split == "train":
        raise ValueError("Train-set evaluation is not a held-out result")
    indices = data.splits[args.split]
    if not indices:
        raise ValueError(f"No {args.split} objects; the public two-object smoke split has no validation set")
    out = Path(args.out or Path(args.checkpoint).parent / f"eval_{args.split}_nfe{args.steps}")
    out.mkdir(parents=True, exist_ok=True)
    if data.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(data.device)
    results = []
    for index in indices:
        record, target = data.records[index], data.objects[index]
        rng = make_rng(args.eval_seed + index, data.device)
        if data.device.type == "cuda":
            torch.cuda.synchronize(data.device)
        begin = time.perf_counter()
        r, p = generate(cfg["method"], model, data, index, rng, args.samples, args.steps)
        if data.device.type == "cuda":
            torch.cuda.synchronize(data.device)
        sampling_seconds = time.perf_counter()-begin
        validity = rotation_residual(r)
        r, p = r.cpu().numpy(), p.cpu().numpy()
        if not np.isfinite(r).all() or not np.isfinite(p).all() or max(validity.values()) > 1e-3:
            raise ValueError("Generated poses are not valid SE(3)")
        target_rng = np.random.default_rng(args.eval_seed+100000+index)
        allowed = target["pose_indices"][args.split].cpu().numpy()
        choose = target_rng.choice(allowed, min(args.samples, len(allowed)), replace=False)
        tr, tp = target["r"][choose].cpu().numpy(), target["p"][choose].cpu().numpy()
        physical_p = p*record["radius_m"] + np.array(record["center_m"])
        transforms = np.broadcast_to(np.eye(4), (len(r), 4, 4)).copy()
        transforms[:, :3, :3], transforms[:, :3, 3] = r, physical_p
        target_transforms = np.broadcast_to(np.eye(4), (len(tr), 4, 4)).copy()
        target_transforms[:, :3, :3] = tr
        target_transforms[:, :3, 3] = tp*record["radius_m"]+np.array(record["center_m"])
        path = out / (record["object_id"] + ".npz")
        np.savez_compressed(path, rotations=r, translations=p, translations_m=physical_p,
                            transforms_m=transforms,
                            target_rotations=tr, target_translations=tp,
                            target_transforms_m=target_transforms,
                            reference_source_indices=target["source_indices"][choose])
        metrics = {"object_id": record["object_id"], "mesh_identity": record["mesh_identity"],
                   "split": args.split, "samples_path": str(path.resolve()), "samples_sha256": sha256(path),
                   "pose_frame": "Original object mesh coordinate frame, mesh scaled by object/scale; transforms_m uses meters and column vectors",
                   "sampling_seconds": sampling_seconds, "validity": validity,
                   "distribution": distribution_metrics(r, p, tr, tp, record["radius_m"])}
        if not args.skip_collision:
            official = args.official_gripper
            n = min(args.collision_samples, len(r))
            metrics["static_geometry"] = static_geometry(record, r[:n], physical_p[:n], official)
            # Original FleX-labeled poses are a geometry sanity reference, not generated successes.
            metrics["heldout_reference_static_geometry"] = static_geometry(
                record, tr[:n], tp[:n]*record["radius_m"]+np.array(record["center_m"]), official)
            metrics["heldout_reference_proxy_source_indices"] = target["source_indices"][choose[:n]].tolist()
            metrics["heldout_reference_proxy_note"] = "Same static_geometry proxy on first min(collision_samples, heldout_reference_count) unfiltered held-out reference draws; historical FleX-positive labels do not imply proxy passes or generated physical success."
        results.append(metrics)
    summary = {"method": cfg["method"], "seed": cfg["seed"], "training_iteration": iteration,
               "method_label": cfg.get("method_label", cfg["method"]), "method_provenance": cfg.get("method_provenance"),
               "nfe": 1 if cfg["method"] == "mgm" else args.steps,
               "split": args.split, "eval_seed": args.eval_seed, "n_objects": len(results),
               "evaluation_config": {k: v for k, v in vars(args).items() if k not in ("func",)},
               "gpu_guard": guard,
               "peak_cuda_bytes": torch.cuda.max_memory_allocated(data.device) if data.device.type == "cuda" else None,
               "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha256(args.checkpoint),
               "manifest": str(data.path.resolve()), "manifest_sha256": sha256(data.path),
               "data_status": cfg["data_status"], "local_implementation": True,
               "evaluation_source_sha256": source_hashes(), "dependency_versions": dependency_versions(),
               "nfe_definition": "One pose-network call per step; geometry encoding is computed once and amortized across draws/steps, not counted as pose NFE.",
               "physical_success_rate": None, "physical_success_status": "NOT EVALUATED; static geometric proxy only",
               "mean_object_mmd2": float(np.mean([m["distribution"]["mmd2_unbiased_fixed_chordal_product_kernel"] for m in results])),
               "objects": results}
    summary["reference_baseline"] = [{"object_id": m["object_id"], "split": args.split,
        "source_indices": m["heldout_reference_proxy_source_indices"], "static_geometry": m["heldout_reference_static_geometry"]}
        for m in results if "heldout_reference_static_geometry" in m]
    geometric = [r["static_geometry"] for r in results if "static_geometry" in r]
    if geometric:
        summary["author_static_feasible_object_fraction_at_20"] = float(np.mean([r["author_static_feasible_any_at_20"] for r in geometric]))
        summary["mean_object_static_feasible_pose_fraction"] = float(np.mean([r["author_static_feasible_pose_fraction"] for r in geometric]))
    write_json(out / "metrics.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "objects"}), flush=True)
    return str(out / "metrics.json")
