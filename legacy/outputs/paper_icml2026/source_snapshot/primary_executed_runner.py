"""Bounded, final-checkpoint empirical supplement. Does not change legacy code."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "external" / "riemannian-fm"
OUT = ROOT / "outputs" / "paper_icml2026"
sys.path.insert(0, str(REPO))

from manifm.datasets import _get_dataset
from manifm.manifolds.mesh import Mesh, closest_point, face_normal
from manifm.manifolds.sphere import Sphere
from manifm.manifolds.torus import FlatTorus
from manifm.rmgm import build_critic, build_generator, critic_loss, generator_loss
from rfm_baseline_train import BaseSampler, rfm_path, sample_rfm
from rmgm_train_real import (
    EmpiricalSampler,
    RealTrainConfig,
    maybe_init_local_gmm_real,
    maybe_init_torus_mixture_real,
)

REAL = ["volcano", "earthquake", "fire", "flood", "general", "glycine", "proline", "prepro", "rna"]
MESH = ["bunny10", "bunny50", "bunny100", "spot10", "spot50", "spot100"]
DEVICE = torch.device("cuda")


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(x):
    return hashlib.sha256(x.contiguous().cpu().numpy().tobytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synced_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def config(name, seed=0):
    torus = name in REAL[4:]
    return RealTrainConfig(
        data=name, seed=seed, data_seed=0, hidden_dim=128, num_layers=3,
        batch_size=256, latent_dim=16, steps=2000, lr=2e-4, lr_c=2e-4,
        lr_g=2e-4, critic_steps=1, weight_decay=0.0, ode_steps=128,
        earth_datadir=str(REPO / "data/earth_data"),
        top500_datadir=str(REPO / "data/top500"),
        rna_datadir=str(REPO / "data/rna"), mesh_datadir=str(REPO / "data/mesh"),
        generator_type="local_gmm", generator_components=12 if name == "volcano" else 24,
        generator_kmeans_init=not torus, generator_kmeans_samples=8192,
        torus_generator="gmm" if torus else "mlp",
        torus_components=64 if name == "general" else 32,
        torus_kmeans_init=torus, torus_kmeans_samples=8192,
    )


def load_data(name):
    cfg = config(name)
    # Only fixed, index-addressable real datasets are allowed in the primary suite.
    from rmgm_train_real import DotConfig
    dataset, _ = _get_dataset(DotConfig(asdict(cfg)))
    if name not in REAL:
        raise ValueError("Primary suite requires a fixed real dataset")
    n = len(dataset)
    p = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    nv = n // 10
    nt = n - 2 * nv
    indices = {"train": p[:nt], "val": p[nt:nt + nv], "test": p[nt + nv:]}
    data = {k: dataset.data[v].float().clone() for k, v in indices.items()}
    return dataset.manifold, dataset.dim, data, indices


def embed(manifold, x):
    return torch.cat([x.cos(), x.sin()], -1) if isinstance(manifold, FlatTorus) else x


def pairwise(manifold, x, y):
    if isinstance(manifold, Sphere):
        return (x @ y.T).clamp(-1, 1).acos()
    if isinstance(manifold, FlatTorus):
        d = x[:, None] - y[None]
        return torch.atan2(d.sin(), d.cos()).square().sum(-1).sqrt()
    return torch.cdist(x, y)


@torch.no_grad()
def reference(manifold, train):
    g = torch.Generator().manual_seed(72301)
    x = train[torch.randperm(len(train), generator=g)[:1024]].cpu()
    e = embed(manifold, x).double()
    d2 = torch.cdist(e, e).square()
    positive = d2[d2 > 1e-12]
    bandwidth2 = float(positive.median()) if positive.numel() else 1.0
    distances = pairwise(manifold, x.double(), x.double())
    distances.fill_diagonal_(float("inf"))
    distances[distances <= 1e-7] = float("inf")
    nearest = distances.min(1).values
    nearest = nearest[torch.isfinite(nearest)]
    radius = float(nearest.median()) if nearest.numel() else 1e-7
    return {"bandwidth2": bandwidth2, "coverage_radius": radius,
            "reference_sha256": tensor_sha(x), "reference_n": len(x)}, x


@torch.no_grad()
def metrics(manifold, samples, target, ref):
    x, y = samples.cpu().double(), target.cpu().double()
    ex, ey = embed(manifold, x), embed(manifold, y)
    b = ref["bandwidth2"]
    # Biased V-statistic, including diagonals; kernel is fixed using training data.
    mmd = (torch.exp(-torch.cdist(ex, ex).square() / b).mean()
           + torch.exp(-torch.cdist(ey, ey).square() / b).mean()
           - 2 * torch.exp(-torch.cdist(ex, ey).square() / b).mean())
    dist = pairwise(manifold, x, y)
    s2d, d2s = dist.min(1).values, dist.min(0).values
    result = {"mmd2_biased": float(mmd), "symmetric_nn": float((s2d.mean() + d2s.mean()) / 2),
              "nn_sample_to_data": float(s2d.mean()), "nn_data_to_sample": float(d2s.mean()),
              "coverage": float((d2s <= ref["coverage_radius"]).double().mean()),
              "generated_n": len(x), "target_n": len(y)}
    if isinstance(manifold, Sphere):
        result["constraint_mean"] = float((x.norm(dim=-1) - 1).abs().mean())
    elif isinstance(manifold, FlatTorus):
        result["constraint_mean"] = float(((x < 0) | (x >= 2 * np.pi)).any(-1).double().mean())
    return result


def field_model(manifold, dim):
    return build_critic({"critic": {"hidden_dim": 128, "num_layers": 3, "fourier": 8,
                                    "metric_normalize": False}}, manifold, dim)


def generator_model(cfg, manifold, dim, method, train):
    d = asdict(cfg)
    if method == "mgm_mlp":
        d.update(generator_type="mlp", torus_generator="mlp")
    model = build_generator({"generator": d}, manifold, dim).to(DEVICE)
    if method == "mgm_gmm":
        sampler = EmpiricalSampler(train, manifold)
        maybe_init_torus_mixture_real(cfg, manifold, sampler, model, DEVICE, torch.float32)
        maybe_init_local_gmm_real(cfg, manifold, sampler, model, DEVICE, torch.float32)
    return model


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def draw(model, manifold, dim, cfg, method, n, draw_seed, device, nfe):
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(draw_seed)
        if devices:
            torch.cuda.manual_seed_all(draw_seed)
        if method == "rfm":
            cc = copy.copy(cfg)
            cc.ode_steps = nfe
            counter = [0]
            hook = model.register_forward_hook(lambda *args: counter.__setitem__(0, counter[0] + 1))
            try:
                result = sample_rfm(cc, manifold, BaseSampler(manifold, dim), model, n,
                                    device, torch.float32)
            finally:
                hook.remove()
            if counter[0] != nfe:
                raise AssertionError((counter[0], nfe))
            return result
        z = torch.randn(n, cfg.latent_dim, device=device)
        return model(z)


def checkpoint(path, model, critic, cfg, method, step, seconds):
    state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "config": asdict(cfg), "method": method, "step": step,
             "train_seconds": seconds, "selection": "fixed_final_step"}
    if critic is not None:
        state["critic"] = {k: v.detach().cpu() for k, v in critic.state_dict().items()}
    torch.save(state, path)


def evaluation(run_dir, model, manifold, dim, cfg, method, step, data, ref, seconds, draws):
    model.eval()
    rows = []
    for nfe in ([128, 1] if method == "rfm" else [1]):
        arrays = {}
        n = min(1024, len(data["val_target"]), len(data["test_target"]))
        # Warmup excluded from synchronized wall-clock sampling timings.
        for _ in range(2):
            draw(model, manifold, dim, cfg, method, n, 900000, DEVICE, nfe)
        for i in range(draws):
            t0 = synced_time()
            sample = draw(model, manifold, dim, cfg, method, n, 100000 + i, DEVICE, nfe)
            elapsed = synced_time() - t0
            sample = sample.cpu()
            arrays[f"generated_draw_{i}"] = sample.numpy()
            for split in ("val", "test"):
                row = metrics(manifold, sample, data[split + "_target"], ref)
                row.update(data=cfg.data, method=method, training_seed=cfg.seed, step=step,
                           draw=i, draw_seed=100000 + i, split=split, sampler_nfe=nfe,
                           vector_field_nfe=nfe if method == "rfm" else 0,
                           generator_calls=0 if method == "rfm" else 1,
                           gpu_sample_seconds=elapsed, train_seconds=seconds)
                rows.append(row)
        cpu_model = copy.deepcopy(model).cpu()
        cpu_manifold = getattr(cpu_model, "manifold", copy.deepcopy(manifold).cpu())
        draw(cpu_model, cpu_manifold, dim, cfg, method, n, 900000, torch.device("cpu"), nfe)
        cpu_times = []
        for i in range(3):
            start = time.perf_counter()
            draw(cpu_model, cpu_manifold, dim, cfg, method, n, 110000 + i, torch.device("cpu"), nfe)
            cpu_times.append(time.perf_counter() - start)
        for row in rows:
            if row["sampler_nfe"] == nfe:
                row["cpu_sample_seconds_mean"] = float(np.mean(cpu_times))
                row["cpu_sample_seconds_sd"] = float(np.std(cpu_times, ddof=1))
        np.savez_compressed(run_dir / f"samples_step{step}_nfe{nfe}.npz", **arrays)
    write_json(run_dir / f"evaluation_step{step}.json", rows)
    model.train()
    return rows


def prepare(name, save=True):
    manifold, dim, data, indices = load_data(name)
    ref, rx = reference(manifold, data["train"])
    for split in ("val", "test"):
        g = torch.Generator().manual_seed(82010 if split == "val" else 82011)
        ii = torch.randperm(len(data[split]), generator=g)[:1024]
        data[split + "_target"] = data[split][ii]
    if save:
        dest = OUT / "datasets" / name
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "split_indices_and_targets.npz", **{k + "_indices": v.numpy() for k, v in indices.items()},
                            val_target=data["val_target"].numpy(), test_target=data["test_target"].numpy(),
                            kernel_reference_train=rx.numpy())
        write_json(dest / "manifest.json", {"data_seed": 0, "counts": {k: len(v) for k, v in indices.items()},
                   "split_indices_sha256": {k: tensor_sha(v) for k, v in indices.items()},
                   "data_tensor_sha256": {k: tensor_sha(data[k]) for k in indices}, **ref,
                   "target_selection": "without replacement; min(1024, split size); independent fixed val/test RNG"})
    return manifold, dim, data, ref


def run_training(name, method, seed, steps, checkpoints, draws=3, benchmark=False):
    setup_started = synced_time()
    seed_all(seed)
    cfg = config(name, seed)
    cfg.steps = steps
    manifold, dim, data, ref = prepare(name, save=not benchmark)
    model = field_model(manifold, dim).to(DEVICE) if method == "rfm" else generator_model(cfg, manifold, dim, method, data["train"])
    critic = None if method == "rfm" else field_model(manifold, dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0)
    opt_c = None if critic is None else torch.optim.AdamW(critic.parameters(), lr=2e-4, weight_decay=0)
    train = data["train"].to(DEVICE)
    target_rng = torch.Generator(device=DEVICE).manual_seed(50000 + seed)
    latent_rng = torch.Generator(device=DEVICE).manual_seed(60000 + seed)
    curve, rows = [], []
    dest = OUT / "controlled" / name / method / f"seed{seed}"
    setup_seconds = synced_time() - setup_started
    initial_parameters = {k: p.detach().cpu().clone() for k, p in model.named_parameters()}
    if not benchmark:
        dest.mkdir(parents=True, exist_ok=True)
        write_json(dest / "config.json", {**asdict(cfg), "method": method, "checkpoints": checkpoints,
                    "model_parameters": parameter_count(model), "critic_parameters": parameter_count(critic) if critic else 0,
                    "setup_seconds_including_data_and_initialization": setup_seconds,
                    "split_seed": 0, "selection": "fixed_final_step", "status": "running", "started_at": utc()})
        if method == "mgm_gmm":
            checkpoint(dest / "step0.pt", model, critic, cfg, method, 0, 0.0)
            rows += evaluation(dest, model, manifold, dim, cfg, method, 0, data, ref, 0.0, draws)

    def target():
        return train[torch.randint(len(train), (cfg.batch_size,), generator=target_rng, device=DEVICE)]

    started = synced_time()
    train_seconds = 0.0
    for step in range(1, steps + 1):
        if method == "rfm":
            x1 = target()
            x0 = BaseSampler(manifold, dim).sample(cfg.batch_size, DEVICE, torch.float32)
            t = torch.rand(cfg.batch_size, 1, device=DEVICE)
            xt, ut = rfm_path(manifold, x0, x1, t)
            opt.zero_grad(set_to_none=True)
            pred = model(t, xt)
            loss = manifold.inner(xt, pred - ut, pred - ut).mean() / dim
            loss.backward()
            opt.step()
            lc = loss
        else:
            z = torch.randn(cfg.batch_size, cfg.latent_dim, device=DEVICE, generator=latent_rng)
            opt_c.zero_grad(set_to_none=True)
            lc, _ = critic_loss(manifold, model, critic, z, target())
            lc.backward()
            opt_c.step()
            critic.requires_grad_(False)
            z = torch.randn(cfg.batch_size, cfg.latent_dim, device=DEVICE, generator=latent_rng)
            opt.zero_grad(set_to_none=True)
            loss, _ = generator_loss(manifold, model, critic, z, target())
            loss.backward()
            opt.step()
            critic.requires_grad_(True)
        if step == 1 or step % 100 == 0:
            curve.append({"step": step, "loss": float(loss.detach().cpu()), "critic_loss": float(lc.detach().cpu())})
            if not np.isfinite(curve[-1]["loss"]) or not np.isfinite(curve[-1]["critic_loss"]):
                raise FloatingPointError((name, method, seed, step))
        if step in checkpoints and not benchmark:
            train_seconds += synced_time() - started
            checkpoint(dest / f"step{step}.pt", model, critic, cfg, method, step, train_seconds)
            rows += evaluation(dest, model, manifold, dim, cfg, method, step, data, ref, train_seconds, draws)
            write_json(dest / "learning_curve.json", curve)
            print(json.dumps({"event": "checkpoint_finished", "data": name, "method": method,
                              "seed": seed, "step": step, "train_seconds": train_seconds, "at": utc()}), flush=True)
            started = synced_time()
    train_seconds += synced_time() - started
    if not benchmark:
        write_json(dest / "completed.json", {"finished_at": utc(), "steps": steps, "train_seconds": train_seconds,
                    "setup_seconds_including_data_and_initialization": setup_seconds,
                    "evaluation_rows": len(rows), "finite_parameters": all(torch.isfinite(p).all().item() for p in model.parameters()),
                    "parameter_change_l2": {k: float((p.detach().cpu() - initial_parameters[k]).norm()) for k, p in model.named_parameters()}})
    return {"data": name, "method": method, "seed": seed, "steps": steps, "train_seconds": train_seconds,
            "seconds_per_step": train_seconds / steps, "setup_seconds": setup_seconds, "model_parameters": parameter_count(model),
            "critic_parameters": parameter_count(critic) if critic else 0}


def benchmark():
    rows = []
    for name, method in [("fire", "mgm_gmm"), ("fire", "mgm_mlp"), ("fire", "rfm"), ("rna", "mgm_gmm"), ("rna", "rfm")]:
        row = run_training(name, method, 7, 100, [], benchmark=True)
        rows.append(row)
        print(json.dumps(row), flush=True)
    write_json(OUT / "benchmark.json", rows)


def train(args):
    jobs = [(name, method, seed) for name in args.datasets for seed in args.seeds
            for method in (["mgm_gmm", "mgm_mlp", "rfm"] if name in args.mlp_datasets else ["mgm_gmm", "rfm"])]
    protocol = {"created_at": utc(), "datasets": args.datasets, "seeds": args.seeds,
                "data_seed": 0, "mgm_steps": args.mgm_steps, "rfm_steps": args.rfm_steps,
                "checkpoints": {"mgm": [args.mgm_steps], "rfm": sorted(set([args.mgm_steps, args.rfm_steps]))},
                "jobs": jobs, "total_jobs": len(jobs), "draws_per_checkpoint": args.draws,
                "metric": "biased MMD2 V-statistic, k=exp(-squared embedded distance / median positive train distance2)",
                "coverage": "threshold is median leave-one-out NN distance on fixed 1024 training reference points",
                "selection": "No validation or test based checkpoint/run/hyperparameter selection. Fixed final steps.",
                "historical_test_exposure": "Legacy test set has been inspected; new training protocol locked before new evaluations. Not a pristine unseen benchmark.",
                "budget": "Same iterations and batch primary; MGM has two updates/iteration vs RFM one. Report measured compute separately.",
                "sampler": "Legacy projected/expmap Euler, midpoint-in-time evaluation, ONE field call/step: 128 steps = 128 NFE.",
                "host": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                         "cpu_threads": torch.get_num_threads()},
                "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), REPO / "manifm/rmgm/geometry.py",
                                    REPO / "manifm/rmgm/losses.py", REPO / "manifm/rmgm/generators.py", REPO / "rfm_baseline_train.py"]}}
    write_json(OUT / "protocol.json", protocol)
    completed = []
    started = time.perf_counter()
    for i, (name, method, seed) in enumerate(jobs):
        existing = OUT / "controlled" / name / method / f"seed{seed}" / "completed.json"
        if existing.exists():
            old = json.loads(existing.read_text())
            steps = args.rfm_steps if method == "rfm" else args.mgm_steps
            if old["steps"] != steps:
                raise ValueError("Refusing to reuse a completed run with different steps")
            completed.append({"data": name, "method": method, "seed": seed, "steps": steps,
                              "train_seconds": old["train_seconds"], "reused_completed_job": True})
            print(json.dumps({"event": "reused_completed_run", "data": name, "method": method, "seed": seed}), flush=True)
            continue
        if time.perf_counter() - started > args.max_seconds:
            print(json.dumps({"event": "budget_stop", "completed": len(completed), "planned": len(jobs)}), flush=True)
            break
        print(json.dumps({"event": "run_start", "index": i + 1, "total": len(jobs), "data": name, "method": method, "seed": seed, "at": utc()}), flush=True)
        steps = args.rfm_steps if method == "rfm" else args.mgm_steps
        checks = sorted(set([args.mgm_steps, steps]))
        row = run_training(name, method, seed, steps, checks, draws=args.draws)
        completed.append(row)
        write_json(OUT / "progress.json", {"started_jobs": i + 1, "finished_jobs": len(completed), "planned_jobs": len(jobs),
                     "elapsed_seconds": time.perf_counter() - started, "runs": completed, "updated_at": utc()})
        print(json.dumps({"event": "run_finished", "finished": len(completed), "planned": len(jobs), **row}), flush=True)
    summarize()


def mesh(args):
    results = []
    for name in MESH:
        dataset_cfg = config("fire")
        dataset_cfg.data = name
        from rmgm_train_real import DotConfig
        dataset, _ = _get_dataset(DotConfig(asdict(dataset_cfg)))
        n = len(dataset)
        # Match the checkpoint's effective data seed, never the evaluation RNG seed.
        p = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        train_idx, test_idx = p[:n - 2 * (n // 10)], p[n - n // 10:]
        g = torch.Generator().manual_seed(82011)
        chosen = test_idx[torch.randperm(len(test_idx), generator=g)[:1024]]
        target = dataset.data[chosen].float()
        train = dataset.data[train_idx[:1024]].float()
        dest = OUT / "legacy_mesh" / name
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "target.npz", target=target.numpy(), target_indices=chosen.numpy())
        for variant in ("projection", "sphere_map", "sphere_map_gmm"):
            paths = list((ROOT / "outputs/rmgm_mesh_bunny_spot_pilot" / variant / name / "rmgm").glob("seed0_*/checkpoints/last.pt"))
            if len(paths) != 1:
                raise ValueError(paths)
            path = paths[0]
            ck = torch.load(path, map_location="cpu", weights_only=True)
            c = ck["config"]
            if ck["step"] != c["steps"] or (c.get("data_seed") if c.get("data_seed") is not None else c["seed"]) != 0:
                raise ValueError("Unexpected legacy step/split")
            # All geometric buffers are restored from the checkpoint. Avoid a new ARPACK solve.
            state = ck["generator"]
            manifold = Mesh.__new__(Mesh)
            torch.nn.Module.__init__(manifold)
            for key, value in state.items():
                if key.startswith("manifold."):
                    manifold.register_buffer(key.split(".", 1)[1], value.clone())
            model = build_generator({"generator": c}, manifold, 3)
            model.load_state_dict(state, strict=True)
            ref, _ = reference(manifold, train)
            model = model.to(DEVICE).eval()
            cfg = RealTrainConfig(**c)
            arrays = {}
            for i in range(args.draws):
                # Chunk brute-force mesh projection to bound N x faces memory.
                torch.manual_seed(100000 + i)
                z = torch.randn(1024, cfg.latent_dim, device=DEVICE)
                start = synced_time()
                with torch.no_grad():
                    samples = torch.cat([model(z[j:j + 64]) for j in range(0, 1024, 64)])
                elapsed = synced_time() - start
                samples = samples.cpu()
                arrays[f"generated_draw_{i}"] = samples.numpy()
                row = metrics(manifold, samples, target, ref)
                row.update(data=name, variant=variant, step=ck["step"], training_seed=c["seed"], data_seed=0,
                           draw=i, checkpoint=str(path), checkpoint_sha256=sha(path), target_sha256=tensor_sha(target),
                           gpu_sample_seconds=elapsed, metric_geometry="ambient Euclidean (NOT intrinsic mesh)", **ref)
                results.append(row)
            np.savez_compressed(dest / f"{variant}_samples.npz", **arrays)
            # Verify tangency and local path velocities on a small held-out probe.
            # Legacy meshes used FOUR layers rather than the supplement's three.
            critic = build_critic({"critic": {"hidden_dim": c["hidden_dim"], "num_layers": c["num_layers"], "fourier": 8}}, manifold, 3).to(DEVICE)
            critic.load_state_dict(ck["critic"], strict=True)
            with torch.no_grad():
                x = target[:32].to(DEVICE)
                v = critic(torch.full((32, 1), 0.25, device=DEVICE), x)
                _, fi = closest_point(x, manifold.v, manifold.f)
                vs = manifold.v[manifold.f[fi]]
                normal = face_normal(vs[:, 0], vs[:, 1], vs[:, 2])
                tangency = ((normal * v).sum(-1).abs() / v.norm(dim=-1).clamp_min(1e-8)).max().item()
            write_json(dest / f"{variant}_tangency.json", {"max_relative_normal_component": tangency, "n": 32})
            print(json.dumps({"event": "mesh_finished", "data": name, "variant": variant, "draws": args.draws,
                              "step": ck["step"], "max_relative_normal_component": tangency}), flush=True)
        write_json(OUT / "legacy_mesh_results.json", results)


def summarize():
    rows = []
    for path in (OUT / "controlled").glob("*/*/seed*/evaluation_step*.json"):
        rows.extend(json.loads(path.read_text()))
    write_json(OUT / "results.json", rows)
    groups = {}
    for row in rows:
        key = (row["data"], row["method"], row["step"], row["sampler_nfe"], row["split"])
        groups.setdefault(key, {}).setdefault(row["training_seed"], []).append(row)
    summary = []
    fields = ["mmd2_biased", "symmetric_nn", "coverage", "gpu_sample_seconds", "cpu_sample_seconds_mean", "train_seconds"]
    for key, seeds in sorted(groups.items()):
        item = dict(zip(["data", "method", "step", "sampler_nfe", "split"], key))
        item.update(training_seeds=sorted(seeds), n_training_seeds=len(seeds),
                    mc_draws_per_seed=[len(v) for v in seeds.values()])
        for field in fields:
            means = [float(np.mean([r[field] for r in v])) for v in seeds.values()]
            item[field] = {"mean": float(np.mean(means)), "sd_across_training_seeds": float(np.std(means, ddof=1)) if len(means) > 1 else None,
                           "per_seed_mc_means": dict(zip(map(str, seeds), means))}
        summary.append(item)
    write_json(OUT / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["benchmark", "train", "mesh", "summarize"])
    parser.add_argument("--datasets", nargs="+", default=REAL)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--mlp-datasets", nargs="*", default=["fire", "general"])
    parser.add_argument("--mgm-steps", type=int, default=2000)
    parser.add_argument("--rfm-steps", type=int, default=10000)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--max-seconds", type=int, default=1050)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.action == "benchmark":
        benchmark()
    elif args.action == "train":
        train(args)
    elif args.action == "mesh":
        mesh(args)
    else:
        summarize()


if __name__ == "__main__":
    main()
