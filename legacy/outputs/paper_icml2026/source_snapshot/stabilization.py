"""General-only validation stabilization study. Never evaluates the test split."""
import argparse
import json
import time

import numpy as np
import torch

import controlled_suite as s
from manifm.rmgm.critics import MLP, TimeFourier

DEST = s.OUT / "stabilization_general"


class PeriodicTangentCritic(torch.nn.Module):
    """One shared periodic architecture for the MGM critic and RFM field."""

    def __init__(self, manifold, dim=2, harmonics=4, width=128, layers=3):
        super().__init__()
        self.manifold = manifold
        self.register_buffer("frequencies", torch.arange(1, harmonics + 1, dtype=torch.float32))
        self.time_features = TimeFourier(8)
        self.net = MLP(2 * dim * harmonics + 16, dim, hidden_dim=width, num_layers=layers, activation="silu")

    def forward(self, t, x):
        if t.ndim == 1:
            t = t[:, None]
        phases = x[..., None] * self.frequencies.to(x)
        spatial = torch.cat([phases.cos(), phases.sin()], dim=-1).flatten(1)
        raw = self.net(torch.cat([spatial, self.time_features(t).to(x)], dim=-1))
        return self.manifold.proju(x, raw)


def make_field(protocol, manifold, periodic):
    return (PeriodicTangentCritic(manifold, harmonics=protocol["spatial_harmonics"],
                                 width=protocol["hidden_dim"], layers=protocol["num_layers"])
            if periodic else s.field_model(manifold, 2)).to(s.DEVICE)


@torch.no_grad()
def evaluate(dest, phase, step, model, manifold, cfg, method, train_target, val_target, ref, draws):
    model.eval()
    rows, arrays = [], {}
    for i in range(draws):
        x = s.draw(model, manifold, 2, cfg, method, len(val_target), 420000 + i, s.DEVICE, 128)
        arrays[f"generated_draw_{i}"] = x.cpu().numpy()
        for split, target in (("train", train_target), ("val", val_target)):
            row = s.metrics(manifold, x, target, ref)
            row.update(trial=dest.name, phase=phase, generator_step=step if method != "rfm" else None,
                       field_updates=step if method == "rfm" else None, step=step,
                       split=split, draw=i, draw_seed=420000 + i, training_seed=cfg.seed,
                       sampler_nfe=128 if method == "rfm" else 1)
            rows.append(row)
    np.savez_compressed(dest / f"{phase}_step{step}_samples.npz", **arrays)
    s.write_json(dest / f"{phase}_step{step}_metrics.json", rows)
    model.train()
    return rows


def run_trial(protocol, trial):
    s.seed_all(protocol["training_seed"])
    cfg = s.config("general", protocol["training_seed"])
    cfg.batch_size = protocol["batch_size"]
    cfg.steps = protocol["generator_steps"]
    cfg.lr_g, cfg.lr_c = protocol["lr_g"], protocol["lr_c"]
    manifold, dim, data, _ = s.load_data("general")
    # Test tensor is not used, sampled, scored or saved by the study.
    del data["test"]
    ref, train_target = s.reference(manifold, data["train"])
    val_g = torch.Generator().manual_seed(82010)
    val_target = data["val"][torch.randperm(len(data["val"]), generator=val_g)[:1024]]
    dest = DEST / trial["id"]
    dest.mkdir(parents=True, exist_ok=True)
    is_rfm = trial["method"] == "rfm"
    if is_rfm:
        model = make_field(protocol, manifold, True)
        critic = None
        method = "rfm"
    else:
        if trial["initialization"] == "random_mlp":
            method = "mgm_mlp"
        else:
            method = "mgm_gmm"
            cfg.torus_mixture_init = "random" if trial["initialization"] == "random_gmm" else "grid"
            cfg.torus_kmeans_init = trial["initialization"] == "train_kmeans_gmm"
        model = s.generator_model(cfg, manifold, dim, method, data["train"])
        critic = make_field(protocol, manifold, trial["periodic"])
    model = model.to(s.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=protocol["lr_rfm"] if is_rfm else cfg.lr_g, weight_decay=0)
    opt_c = None if is_rfm else torch.optim.AdamW(critic.parameters(), lr=cfg.lr_c, weight_decay=0)
    train_data = data["train"].to(s.DEVICE)
    target_rng = torch.Generator(device=s.DEVICE).manual_seed(50000 + cfg.seed)
    latent_rng = torch.Generator(device=s.DEVICE).manual_seed(60000 + cfg.seed)
    field = model if is_rfm else critic
    with torch.no_grad():
        x = train_data[:32]
        t = torch.full((32, 1), 0.25, device=s.DEVICE)
        f0 = field(t, x)
        seam_error = (f0 - field(t, x + 2 * torch.pi)).abs().max().item()
    s.write_json(dest / "config.json", {"protocol": protocol, "trial": trial, "generator_config": s.asdict(cfg),
                 "generator_parameters": s.parameter_count(model), "critic_parameters": s.parameter_count(critic) if critic else 0,
                 "initial_seam_max_absolute": seam_error, "validation_target_sha256": s.tensor_sha(val_target), **ref})
    original = {k: p.detach().cpu().clone() for k, p in model.named_parameters()}
    curves, all_rows = [], []
    train_seconds = 0.0
    c_updates = 0

    def target():
        return train_data[torch.randint(len(train_data), (cfg.batch_size,), generator=target_rng, device=s.DEVICE)]

    def latent():
        return torch.randn(cfg.batch_size, cfg.latent_dim, device=s.DEVICE, generator=latent_rng)

    def critic_update():
        nonlocal c_updates
        opt_c.zero_grad(set_to_none=True)
        loss_c, cm = s.critic_loss(manifold, model, critic, latent(), target())
        loss_c.backward()
        opt_c.step()
        c_updates += 1
        return loss_c, cm

    def save(phase, step):
        nonlocal all_rows
        payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                   "critic": {k: v.detach().cpu() for k, v in critic.state_dict().items()} if critic else None,
                   "protocol": protocol, "trial": trial, "phase": phase, "step": step,
                   "generator_config": s.asdict(cfg), "critic_updates": c_updates,
                   "train_seconds": train_seconds, "cpu_rng": torch.random.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state(), "target_rng": target_rng.get_state(),
                   "latent_rng": latent_rng.get_state(), "optimizer": opt.state_dict(),
                   "critic_optimizer": opt_c.state_dict() if opt_c else None}
        torch.save(payload, dest / f"{phase}_step{step}.pt")
        rr = evaluate(dest, phase, step, model, manifold, cfg, method, train_target, val_target, ref, protocol["evaluation_draws"])
        all_rows += rr
        vm = float(np.mean([r["mmd2_biased"] for r in rr if r["split"] == "val"]))
        print(json.dumps({"event": "validation_probe", "trial": trial["id"], "phase": phase, "step": step,
                          "validation_mmd2": vm, "critic_updates": c_updates, "train_seconds": train_seconds}), flush=True)

    save("initial", 0)
    if not is_rfm:
        model.requires_grad_(False)
        started = s.synced_time()
        for _ in range(protocol["critic_warmup"]):
            lc, cm = critic_update()
        train_seconds += s.synced_time() - started
        model.requires_grad_(True)
        save("warmup", 0)
    nsteps = trial["updates"] if is_rfm else protocol["generator_steps"]
    probes = trial["evaluation_updates"] if is_rfm else protocol["evaluation_generator_steps"]
    started = s.synced_time()
    for step in range(1, nsteps + 1):
        if is_rfm:
            x0 = s.BaseSampler(manifold, 2).sample(cfg.batch_size, s.DEVICE, torch.float32)
            tt = torch.rand(cfg.batch_size, 1, device=s.DEVICE)
            xt, ut = s.rfm_path(manifold, x0, target(), tt)
            opt.zero_grad(set_to_none=True)
            pred = model(tt, xt)
            loss = manifold.inner(xt, pred - ut, pred - ut).mean() / 2
            loss.backward()
            opt.step()
            lc = loss
        else:
            for _ in range(trial["critic_steps"]):
                lc, cm = critic_update()
            critic.requires_grad_(False)
            opt.zero_grad(set_to_none=True)
            loss, gm = s.generator_loss(manifold, model, critic, latent(), target())
            loss.backward()
            opt.step()
            critic.requires_grad_(True)
        if step == 1 or step % 100 == 0:
            item = {"step": step, "loss": float(loss.detach().cpu()), "critic_loss": float(lc.detach().cpu()), "critic_updates": c_updates}
            if not np.isfinite(item["loss"]) or not np.isfinite(item["critic_loss"]):
                raise FloatingPointError(item)
            curves.append(item)
        if step in probes:
            train_seconds += s.synced_time() - started
            save("final" if step == nsteps else "joint", step)
            s.write_json(dest / "learning_curve.json", curves)
            started = s.synced_time()
    result = {"trial": trial["id"], "finished_at": s.utc(), "training_seed": cfg.seed,
              "steps": nsteps, "critic_updates": c_updates, "train_seconds": train_seconds,
              "total_optimizer_updates": nsteps + c_updates, "test_evaluations": 0,
              "parameter_change_l2": {k: float((p.detach().cpu() - original[k]).norm()) for k, p in model.named_parameters()}}
    initial = [r["mmd2_biased"] for r in all_rows if r["phase"] == "initial" and r["split"] == "val"]
    final = [r["mmd2_biased"] for r in all_rows if r["phase"] == "final" and r["split"] == "val"]
    result.update(initial_validation_mmd2_mean=float(np.mean(initial)), final_validation_mmd2_mean=float(np.mean(final)),
                  final_validation_mmd2_sd_mc=float(np.std(final, ddof=1)),
                  validation_mmd2_change=float(np.mean(final) - np.mean(initial)))
    s.write_json(dest / "completed.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=s.Path, default=s.Path(__file__).with_name("stabilization_general.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = json.loads(args.config.read_text())
    DEST.mkdir(parents=True, exist_ok=True)
    s.write_json(DEST / "protocol.json", {"created_at": s.utc(), "config": protocol, "config_sha256": s.sha(args.config),
                 "source_sha256": s.sha(s.Path(__file__)), "test_policy": "NO test evaluation; validation-only seed-0 exploratory study"})
    results = []
    for trial in protocol["trials"]:
        print(json.dumps({"event": "trial_start", "trial": trial["id"], "finished": len(results), "planned": len(protocol["trials"]), "at": s.utc()}), flush=True)
        results.append(run_trial(protocol, trial))
        s.write_json(DEST / "results.json", results)
        print(json.dumps({"event": "trial_finished", "finished": len(results), "planned": len(protocol["trials"]), **results[-1]}), flush=True)
    candidates = {t["id"] for t in protocol["trials"] if t["candidate"]}
    selected = min((r for r in results if r["trial"] in candidates), key=lambda r: r["final_validation_mmd2_mean"])
    s.write_json(DEST / "selection.json", {"selected_at": s.utc(), "rule": protocol["selection"],
                 "selected_trial": selected["trial"], "selected_final_step": protocol["generator_steps"],
                 "validation_summary": selected, "new_test_evaluations": 0,
                 "interpretation": "Single training seed, exploratory validation selection; requires locked independent-seed confirmation. No superiority claim."})
    print(json.dumps({"event": "selection_locked", "trial": selected["trial"], "test_evaluations": 0}), flush=True)


if __name__ == "__main__":
    main()
