"""Training loop for one Fig. 5.2 cell.

Used by run_sweep.py. Reads everything from a config dict (see config.yml),
takes per-snapshot metrics, returns the snapshot list.
"""
import yaml
import torch

from metrics import c2st
from models.backbone import MLP
from models.drift import EuclideanDrift
from tasks import get_task


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, out_dim: int, device) -> EuclideanDrift:
    bb = cfg["backbone"]
    dr = cfg["drift"]
    backbone = MLP(
        in_dim=dr["dim_noise"],
        out_dim=out_dim,
        hidden_sizes=tuple(bb["hidden_sizes"]),
        activation=bb["activation"],
    )
    return EuclideanDrift(
        backbone=backbone,
        dim_noise=dr["dim_noise"],
        device=device,
        temp=dr["temp"],
        max_step=dr["max_step"],
        ignore_self_neg=dr["ignore_self_neg"],
        eps=dr["eps"],
        update_type=dr["update_type"],
        nu=dr.get("nu", 1.5),
    ).to(device)


def take_snapshot(model, task, step, n, ev, seed, device):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x_real = task.sample(n, seed=seed + step + 10**6).to(device)
        x_fake = model.sample(n, device=device)
    acc = c2st(
        x_real, x_fake,
        n_folds=int(ev["c2st_folds"]),
        max_iter=int(ev["c2st_max_iter"]),
        seed=seed,
    )
    if was_training:
        model.train()
    return {"step": int(step), "c2st": float(acc)}


def train_and_snapshot(cfg: dict, verbose: bool = True):
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)

    device = torch.device(cfg.get("device", "cpu"))
    task = get_task(cfg["task"]["name"])
    model = build_model(cfg, out_dim=task.dim, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=float(cfg["training"]["lr"]))

    n_steps = int(cfg["training"]["n_steps"])
    batch_size = int(cfg["training"]["batch_size"])
    log_every = int(cfg["training"]["log_every"])

    ev = cfg["eval"]
    n_track = int(ev.get("n_track_samples", 2048))
    snapshot_steps = sorted({int(s) for s in ev.get("snapshot_steps", [n_steps]) if int(s) <= n_steps})
    if n_steps not in snapshot_steps:
        snapshot_steps.append(n_steps)
    snapshot_set = set(snapshot_steps)

    snapshots = []
    if 0 in snapshot_set:
        snapshots.append(take_snapshot(model, task, 0, n_track, ev, seed, device))

    model.train()
    for step in range(1, n_steps + 1):
        X_pos = task.sample(batch_size, seed=seed + step).to(device)
        loss, stats = model.get_loss(X_pos)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if verbose and (step == 1 or step % log_every == 0):
            print(
                f"step {step:>5d}  loss={loss.item():.4f}  "
                f"V={stats['V_norm_mean'].item():.3f}  "
                f"|X|={stats['X_norm_mean'].item():.3f}"
            )
        if step in snapshot_set:
            snapshots.append(take_snapshot(model, task, step, n_track, ev, seed, device))
    return snapshots
