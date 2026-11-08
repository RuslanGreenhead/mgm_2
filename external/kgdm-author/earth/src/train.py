from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import copy
import time

import torch
from torch import nn

from task.earth_dataset import evaluate_earth_model


@dataclass
class TrainConfig:
    project: str = "kgd-earth"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    epochs: int = 20
    lr: float = 1e-4
    warmup_steps: int = 2500
    log_every: int = 50
    eval_every: int = 500
    device: str = "cpu"
    batch_size: int = 128
    grad_clip: Optional[float] = 1.0
    weight_decay: float = 1e-2
    ema_decay: float = 0.999
    mmd_gamma: float = 1.0

    # model selection
    selection_metric: str = "mmd"
    selection_mode: str = "min"  # "min" or "max"
    n_eval_repeats: int = 3

    # early stopping
    early_stopping_start_step: int = 100
    early_stopping_patience: int = 20

    # logging
    use_wandb: bool = False


def update_ema(model: nn.Module, ema_model: nn.Module, decay: float):
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)


def is_better(new_value: float, best_value: float | None, mode: str) -> bool:
    if best_value is None:
        return True
    if mode == "min":
        return new_value < best_value
    if mode == "max":
        return new_value > best_value
    raise ValueError(f"Unknown selection_mode: {mode}")


def averaged_eval(model, dataloader, cfg: "TrainConfig"):
    vals = []
    for _ in range(cfg.n_eval_repeats):
        metrics = evaluate_earth_model(
            model=model,
            dataloader=dataloader,
            device=cfg.device,
            gamma=cfg.mmd_gamma,
            return_samples=False,
        )
        vals.append(metrics)

    mean_metrics = {k: sum(v[k] for v in vals) / len(vals) for k in vals[0]}
    std_metrics = {
        k: (sum((v[k] - mean_metrics[k]) ** 2 for v in vals) / len(vals)) ** 0.5
        for k in vals[0]
    }
    return mean_metrics, std_metrics


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    cfg: TrainConfig,
):
    wandb = None
    if cfg.use_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(project=cfg.project, entity=cfg.entity,
                       name=cfg.run_name, config=vars(cfg))
        except Exception as e:  # noqa: BLE001
            print(f"[wandb disabled: {e}]")
            wandb = None

    def wlog(d):
        if wandb is not None:
            wandb.log(d)

    model.to(cfg.device)
    model.train()

    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad_(False)
    ema_model.eval()

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )

    total_steps = cfg.epochs * len(train_loader)

    def lr_lambda(step):
        if cfg.warmup_steps <= 0:
            return 1.0
        if step < cfg.warmup_steps:
            return float(step) / float(max(1, cfg.warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    use_amp = cfg.device.startswith("cuda")
    amp_dtype = torch.bfloat16

    steps_per_epoch = len(train_loader)
    print(f"Starting training: {steps_per_epoch} steps/epoch, {total_steps} total.")

    t0 = time.time()
    global_step = 0

    best_metric = None
    best_step = None
    best_epoch = None
    best_val_metrics = None
    best_ema_state = copy.deepcopy(ema_model.state_dict())
    evals_without_improvement = 0
    should_stop = False

    for epoch in range(1, cfg.epochs + 1):
        for batch in train_loader:
            global_step += 1
            X_pos = batch["x"].float().to(cfg.device)

            with torch.autocast(device_type=cfg.device, dtype=amp_dtype, enabled=use_amp):
                loss, stats = model.get_loss(X_pos=X_pos)

            if not torch.isfinite(loss):
                raise RuntimeError("Loss is not finite")

            optim.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.grad_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

            optim.step()
            scheduler.step()
            update_ema(model, ema_model, cfg.ema_decay)

            do_log = (global_step % cfg.log_every == 0) or (global_step == 1)
            do_eval = (global_step % cfg.eval_every == 0) or (
                epoch == cfg.epochs and global_step % steps_per_epoch == 0
            )

            if do_log or do_eval:
                log_dict = {"step": global_step, "epoch": epoch}

                if do_log:
                    dt = time.time() - t0
                    log_dict["train/loss"] = loss.item()
                    log_dict["train/lr"] = optim.param_groups[0]["lr"]
                    log_dict["train/grad_norm"] = float(grad_norm)
                    for k, v in stats.items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            log_dict[f"train/{k}"] = v.item()
                    if not do_eval:
                        print(
                            f"Step {global_step}/{total_steps} | loss {loss.item():.4f} | "
                            f"lr {log_dict['train/lr']:.2e} | {dt:.1f}s"
                        )

                if do_eval:
                    print(f"\n--- Validation at step {global_step}, epoch {epoch} ---")
                    val_mean, val_std = averaged_eval(ema_model, val_loader, cfg)
                    print("Val -> " + " | ".join(f"{k}: {v:.4f}" for k, v in val_mean.items()))

                    log_dict.update({f"val/{k}_mean": v for k, v in val_mean.items()})
                    log_dict.update({f"val/{k}_std": v for k, v in val_std.items()})

                    if cfg.selection_metric not in val_mean:
                        raise KeyError(
                            f"selection_metric='{cfg.selection_metric}' not in metrics: "
                            f"{list(val_mean.keys())}"
                        )

                    current_metric = val_mean[cfg.selection_metric]
                    if is_better(current_metric, best_metric, cfg.selection_mode):
                        best_metric = current_metric
                        best_step = global_step
                        best_epoch = epoch
                        best_val_metrics = copy.deepcopy(val_mean)
                        best_ema_state = copy.deepcopy(ema_model.state_dict())
                        evals_without_improvement = 0
                        print(f"New best: {cfg.selection_metric}={best_metric:.6f} "
                              f"(step {best_step}, epoch {best_epoch})")
                    else:
                        if global_step >= cfg.early_stopping_start_step:
                            evals_without_improvement += 1
                        print(f"No improvement. Patience: "
                              f"{evals_without_improvement}/{cfg.early_stopping_patience}")
                        if evals_without_improvement >= cfg.early_stopping_patience:
                            print(f"Early stopping at step {global_step}, epoch {epoch}.")
                            should_stop = True

                wlog(log_dict)
                t0 = time.time()
                if should_stop:
                    break
        if should_stop:
            break

    ema_model.load_state_dict(best_ema_state)
    ema_model.eval()

    test_mean, test_std = averaged_eval(ema_model, test_loader, cfg)

    print("\nFinal test evaluation using BEST validation checkpoint:")
    print(f"Best checkpoint: epoch={best_epoch}, step={best_step}")
    for k, v in test_mean.items():
        print(f"  {k}: {v:.4f} +/- {test_std[k]:.4f}")

    final_log = {"best/step": best_step, "best/epoch": best_epoch}
    if best_val_metrics is not None:
        final_log.update({f"best_val/{k}": v for k, v in best_val_metrics.items()})
    final_log.update({f"test/{k}_mean": v for k, v in test_mean.items()})
    final_log.update({f"test/{k}_std": v for k, v in test_std.items()})
    wlog(final_log)

    if wandb is not None:
        for k, v in final_log.items():
            wandb.summary[k] = v
        wandb.finish()

    return {
        "best_step": best_step,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "test_mean": test_mean,
        "test_std": test_std,
    }
