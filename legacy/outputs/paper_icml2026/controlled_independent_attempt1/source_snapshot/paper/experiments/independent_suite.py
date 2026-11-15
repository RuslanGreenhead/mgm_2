"""Frozen-source RNG-isolated rerun, preserving the earlier controlled bundle."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import pathlib
import shutil
import sys


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def prepare(root, output):
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    files = list((root / "paper/experiments").glob("*.py"))
    files += list((root / "paper/experiments").glob("*.json"))
    files += [p for p in (root / "external/riemannian-fm").rglob("*.py")
              if not any(part.startswith(".") or part == "__pycache__" for part in p.relative_to(root).parts)]
    hashes = {}
    for src in files:
        relative = src.relative_to(root)
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        hashes[str(relative)] = digest(dest)
    payload = {"workspace_root": str(root), "output_root": str(output), "source_sha256": hashes,
               "execution": "Run the worker script from this immutable snapshot; original source is not imported."}
    (output / "frozen_sources.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"snapshot_files": len(hashes), "worker": str(snapshot / "paper/experiments/independent_suite.py"),
                      "root": str(root), "output": str(output)}), flush=True)


def worker(root, output):
    import numpy as np
    import torch
    import controlled_suite as s
    from manifm.rmgm.geometry import midpoint_path, sample_midpoint_flip_time
    from manifm.rmgm.losses import metric_inner

    s.ROOT = root
    s.REPO = root / "external/riemannian-fm"
    s.OUT = output
    frozen = json.loads((output / "frozen_sources.json").read_text())
    snapshot = output / "source_snapshot"
    if not pathlib.Path(__file__).is_relative_to(snapshot):
        raise RuntimeError("Worker must execute from the frozen snapshot")

    def verify_sources():
        for relative, expected in frozen["source_sha256"].items():
            if digest(snapshot / relative) != expected:
                raise RuntimeError(f"Frozen source changed: {relative}")

    def executed_sources():
        result = []
        for name, module in sorted(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if not path:
                continue
            p = pathlib.Path(path).resolve()
            if p.is_relative_to(snapshot) and p.suffix == ".py":
                relative = str(p.relative_to(snapshot))
                expected = frozen["source_sha256"][relative]
                actual = digest(p)
                assert actual == expected
                result.append({"module": name, "path": relative, "sha256": actual, "verified": True})
            elif p.is_relative_to(root / "external/riemannian-fm") or p.is_relative_to(root / "paper/experiments"):
                raise RuntimeError(f"Imported live project source rather than snapshot: {p}")
        return result

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    verify_sources()
    old = root / "outputs/paper_icml2026/controlled"
    cases = ["fire", "general", "rna", "volcano", "earthquake", "flood", "glycine", "proline", "prepro"]
    jobs = [(name, method, seed) for name in cases for seed in (0, 1, 2) for method in ("mgm_gmm", "rfm")]
    copied = [(name, method, seed) for name in ("fire", "general", "rna") for method in ("mgm_mlp", "mmd_mlp") for seed in (0, 1, 2)]
    protocol = {"created_at": s.utc(), "data_seed": 0, "training_seeds": [0, 1, 2], "rerun_jobs": jobs,
                "copied_jobs": copied, "fixed_hyperparameters": "Exactly the original 2000-step MGM / 10000-step RFM, batch256, width128, layers3, LR2e-4; no test-based tuning",
                "selection": "Final steps only, RFM 2000/10000, GMM 0/2000", "source_snapshot": str(snapshot),
                "rng_streams": {"initialization": "training_seed", "target": "50000+seed", "latent": "60000+seed",
                                "path_flip_or_rfm_time": "70000+seed", "rfm_base": "80000+seed"},
                "rng_policy": "Private explicit torch.Generators; every stream and global CPU/CUDA state saved at checkpoints; evaluation invariance asserted",
                "sampler": "128 actual field calls for 128 Euler steps; 1-call optional baseline", "draws_per_checkpoint": 3,
                "historical_test_exposure": "Same historically inspected benchmark test; correction of RNG contamination, not a fresh unseen test",
                "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
                "source_sha256": frozen["source_sha256"]}
    s.write_json(output / "protocol.json", protocol)
    provenance = []
    for name, method, seed in copied:
        src = old / name / method / f"seed{seed}"
        dest = output / "controlled" / name / method / f"seed{seed}"
        hashes = {str(p.relative_to(src)): digest(p) for p in src.rglob("*") if p.is_file()}
        if not dest.exists():
            shutil.copytree(src, dest)
        assert all(digest(dest / rel) == h for rel, h in hashes.items())
        item = {"data": name, "method": method, "training_seed": seed, "origin": "copied_unchanged",
                "source_directory": str(src), "new_training_seconds": 0.0, "copied_sha256": hashes,
                "reason": "No pre-final training evaluation; original initialization/target/latent/path streams unaffected by the CPU timing reset",
                "saved_private_rng_states": False, "source_provenance": "Original source_snapshot and mmd_protocol.json remain untouched"}
        s.write_json(dest / "provenance.json", item)
        provenance.append(item)
    s.write_json(output / "provenance.json", provenance)

    def objective(manifold, model, critic, z, x1, path_rng, generator_update):
        with contextlib.nullcontext() if generator_update else torch.no_grad():
            x0 = model(z)
            times = sample_midpoint_flip_time(len(x0), device=x0.device, dtype=x0.dtype, generator=path_rng)
            xt, at = midpoint_path(manifold, x0, x1, times.s)
        pred = critic(times.t, xt)
        if generator_update:
            value = 2 * metric_inner(manifold, xt, pred, at) - metric_inner(manifold, xt, pred, pred)
        else:
            value = metric_inner(manifold, xt, pred - at, pred - at)
        return value.mean() / x1.shape[-1]

    # The stream repair must not change either midpoint objective or its gradient.
    s.seed_all(123)
    probe_manifold = s.FlatTorus()
    probe_cfg = s.asdict(s.config("general"))
    probe_cfg.update(torus_generator="mlp", generator_type="mlp")
    probe_model = s.build_generator({"generator": probe_cfg}, probe_manifold, 2)
    probe_critic = s.field_model(probe_manifold, 2)
    probe_z, probe_x1 = torch.randn(32, 16), torch.rand(32, 2) * (2 * np.pi)
    from manifm.rmgm.losses import critic_loss as original_c, generator_loss as original_g
    torch.random.default_generator.manual_seed(9876)
    old_c, _ = original_c(probe_manifold, probe_model, probe_critic, probe_z, probe_x1)
    new_c = objective(probe_manifold, probe_model, probe_critic, probe_z, probe_x1,
                      torch.Generator().manual_seed(9876), False)
    torch.random.default_generator.manual_seed(9876)
    old_g, _ = original_g(probe_manifold, probe_model, probe_critic, probe_z, probe_x1)
    old_grad = torch.autograd.grad(old_g, tuple(probe_model.parameters()))
    new_g = objective(probe_manifold, probe_model, probe_critic, probe_z, probe_x1,
                      torch.Generator().manual_seed(9876), True)
    new_grad = torch.autograd.grad(new_g, tuple(probe_model.parameters()))
    eq = {"critic_loss_abs_diff": float((old_c - new_c).detach()),
          "generator_loss_abs_diff": float((old_g - new_g).detach()),
          "generator_gradient_max_abs_diff": max(float((a - b).abs().max()) for a, b in zip(old_grad, new_grad))}
    assert max(abs(x) for x in eq.values()) < 1e-6
    s.write_json(output / "objective_equivalence.json", eq)

    def train_case(name, method, seed):
        s.seed_all(seed)
        cfg = s.config(name, seed)
        cfg.steps = 10000 if method == "rfm" else 2000
        manifold, dim, data, ref = s.prepare(name)
        model = s.field_model(manifold, dim).to(s.DEVICE) if method == "rfm" else s.generator_model(cfg, manifold, dim, method, data["train"])
        critic = None if method == "rfm" else s.field_model(manifold, dim).to(s.DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0)
        opt_c = None if critic is None else torch.optim.AdamW(critic.parameters(), lr=2e-4, weight_decay=0)
        rng = {key: torch.Generator(device=s.DEVICE).manual_seed(base + seed)
               for key, base in (("target", 50000), ("latent", 60000), ("path", 70000), ("base", 80000))}
        train = data["train"].to(s.DEVICE)
        dest = output / "controlled" / name / method / f"seed{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        s.write_json(dest / "config.json", {**s.asdict(cfg), "method": method, "source": "frozen_snapshot",
                     "model_parameters": s.parameter_count(model), "critic_parameters": s.parameter_count(critic) if critic else 0,
                     "rng_stream_seeds": {k: base + seed for k, base in (("target", 50000), ("latent", 60000), ("path", 70000), ("base", 80000))},
                     "started_at": s.utc()})
        previous_cfg = json.loads((old / name / method / f"seed{seed}" / "config.json").read_text())
        assert all(previous_cfg[k] == value for k, value in s.asdict(cfg).items() if k != "method")
        initial = {k: p.detach().cpu().clone() for k, p in model.named_parameters()}
        curve, elapsed = [], 0.0

        def states():
            return {**{k: v.get_state() for k, v in rng.items()}, "global_cpu": torch.random.get_rng_state(), "global_cuda": torch.cuda.get_rng_state()}

        def target():
            return train[torch.randint(len(train), (256,), device=s.DEVICE, generator=rng["target"])]

        def latent():
            return torch.randn(256, cfg.latent_dim, device=s.DEVICE, generator=rng["latent"])

        def save(step):
            before = states()
            torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "critic": {k: v.detach().cpu() for k, v in critic.state_dict().items()} if critic else None,
                        "method": method, "config": s.asdict(cfg), "step": step, "train_seconds": elapsed,
                        "rng_states": before, "optimizer": opt.state_dict(), "critic_optimizer": opt_c.state_dict() if opt_c else None,
                        "selection": "fixed_final_step", "source_sha256": frozen["source_sha256"]}, dest / f"step{step}.pt")
            s.evaluation(dest, model, manifold, dim, cfg, method, step, data, ref, elapsed, 3)
            after = states()
            if not all(torch.equal(before[k], after[k]) for k in before):
                raise RuntimeError(f"Evaluation altered training RNG state: {name}/{method}/{seed}/{step}")
            s.write_json(dest / f"rng_invariance_step{step}.json", {"all_streams_unchanged": True, "streams": list(before)})

        if method == "mgm_gmm":
            save(0)
        started = s.synced_time()
        for step in range(1, cfg.steps + 1):
            if method == "rfm":
                if isinstance(manifold, s.FlatTorus):
                    raw = torch.rand(256, dim, device=s.DEVICE, generator=rng["base"]) * (2 * np.pi)
                else:
                    raw = torch.randn(256, dim, device=s.DEVICE, generator=rng["base"])
                x0 = manifold.projx(manifold.projx(raw))
                tt = torch.rand(256, 1, device=s.DEVICE, generator=rng["path"])
                xt, ut = s.rfm_path(manifold, x0, target(), tt)
                opt.zero_grad(set_to_none=True)
                pred = model(tt, xt)
                loss = metric_inner(manifold, xt, pred - ut, pred - ut).mean() / dim
                loss.backward()
                opt.step()
                lc = loss
            else:
                opt_c.zero_grad(set_to_none=True)
                lc = objective(manifold, model, critic, latent(), target(), rng["path"], False)
                lc.backward()
                opt_c.step()
                critic.requires_grad_(False)
                opt.zero_grad(set_to_none=True)
                loss = objective(manifold, model, critic, latent(), target(), rng["path"], True)
                loss.backward()
                opt.step()
                critic.requires_grad_(True)
            if step == 1 or step % 100 == 0:
                item = {"step": step, "loss": float(loss.detach().cpu()), "critic_loss": float(lc.detach().cpu())}
                if not all(np.isfinite(item[k]) for k in ("loss", "critic_loss")):
                    raise FloatingPointError(item)
                curve.append(item)
            if step in (2000, cfg.steps):
                elapsed += s.synced_time() - started
                save(step)
                s.write_json(dest / "learning_curve.json", curve)
                started = s.synced_time()
        result = {"data": name, "method": method, "seed": seed, "steps": cfg.steps, "train_seconds": elapsed,
                  "finished_at": s.utc(), "finite_parameters": all(torch.isfinite(p).all().item() for p in model.parameters()),
                  "evaluation_rng_invariance_verified": True,
                  "parameter_change_l2": {k: float((p.detach().cpu() - initial[k]).norm()) for k, p in model.named_parameters()}}
        s.write_json(dest / "completed.json", result)
        s.write_json(dest / "provenance.json", {"origin": "rerun_with_independent_saved_rng_streams", "old_directory": str(old / name / method / f"seed{seed}"),
                                                "new_directory": str(dest), "measurement_fix_only": True, "source_sha256": frozen["source_sha256"]})
        return result

    completed = []
    for index, (name, method, seed) in enumerate(jobs):
        verify_sources()
        print(json.dumps({"event": "independent_run_start", "index": index + 1, "planned": 54, "data": name, "method": method, "seed": seed, "at": s.utc()}), flush=True)
        result = train_case(name, method, seed)
        verify_sources()
        completed.append(result)
        provenance.append({"data": name, "method": method, "training_seed": seed,
                           "origin": "rerun_with_independent_saved_rng_streams",
                           "old_directory": str(old / name / method / f"seed{seed}"),
                           "new_directory": str(output / "controlled" / name / method / f"seed{seed}"),
                           "fixed_hyperparameters_equal_original": True})
        s.write_json(output / "provenance.json", provenance)
        s.write_json(output / "progress.json", {"started_reruns": index + 1, "finished_reruns": len(completed), "planned_reruns": 54,
                     "copied_unchanged": 18, "runs": completed, "updated_at": s.utc()})
        print(json.dumps({"event": "independent_run_finished", "finished": len(completed), "planned": 54, **result}), flush=True)
    s.summarize()
    s.write_json(output / "executed_source_hashes.json", executed_sources())
    s.write_json(output / "run_manifest.json", {"all_completed": len(completed) == 54, "rerun_jobs": 54, "copied_unchanged_jobs": 18,
                 "all_evaluations_rng_invariant": True, "executed_source_hashes_verified": True,
                 "new_train_seconds": sum(r["train_seconds"] for r in completed), "completed_at": s.utc(), "jobs": completed})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--root", type=pathlib.Path)
    parser.add_argument("--output-root", type=pathlib.Path)
    args = parser.parse_args()
    root = (args.root or pathlib.Path(__file__).resolve().parents[2]).resolve()
    output = (args.output_root or root / "outputs/paper_icml2026/controlled_independent").resolve()
    if not output.is_relative_to(root / "outputs/paper_icml2026"):
        raise ValueError("Output must be inside the authorized empirical directory")
    if args.prepare == args.worker:
        raise ValueError("Choose exactly one of --prepare or --worker")
    if args.prepare:
        prepare(root, output)
    else:
        worker(root, output)


if __name__ == "__main__":
    main()
