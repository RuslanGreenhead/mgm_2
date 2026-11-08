"""Training loop with snapshot capture + JSON metrics dump.

Runs one (manifold × dataset × geometry × method) configuration end-to-end.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics.sliced_w2 import sliced_w2
from metrics.tile_acc import tile_acc


@dataclass
class TrainConfig:
    steps: int = 2000
    lr: float = 1e-3
    log_every: int = 100
    capture_steps: List[int] = field(default_factory=lambda: [0, 10, 50, 200, 1000, 2000])
    n_eval_samples: int = 4096
    sliced_w2_n_proj: int = 128
    grad_clip: Optional[float] = 1.0
    device: str = "cpu"
    out_dir: str = "results/run"
    save_samples: bool = True


def _is_checkerboard(dataset_name: str) -> bool:
    return dataset_name.startswith("checkerboard_")


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    return x


def train(
    model: nn.Module,
    loader: DataLoader,
    target_pool: torch.Tensor,
    cfg: TrainConfig,
    *,
    dataset_name: str,
    manifold,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run training, capture samples + metrics at requested steps, dump JSON."""
    model.to(cfg.device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_pool = target_pool.to(cfg.device)

    snapshots: Dict[int, np.ndarray] = {}
    log_curve: List[Dict[str, float]] = []
    capture_set = set(cfg.capture_steps)

    # Step 0: pre-training snapshot.
    if 0 in capture_set:
        with torch.no_grad():
            s0 = model.sample(cfg.n_eval_samples, device=cfg.device)
        snapshots[0] = _to_numpy(s0)

    t0 = time.time()
    it = iter(loader)

    for step in range(1, cfg.steps + 1):
        try:
            y_pos = next(it)
        except StopIteration:
            it = iter(loader)
            y_pos = next(it)
        if isinstance(y_pos, (tuple, list)) and len(y_pos) == 1:
            y_pos = y_pos[0]
        y_pos = y_pos.to(cfg.device)

        loss, stats = model.get_loss(y_pos)

        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

        if step % cfg.log_every == 0 or step == 1 or step in capture_set:
            with torch.no_grad():
                s = model.sample(cfg.n_eval_samples, device=cfg.device)
            sw = sliced_w2(s, target_pool[: cfg.n_eval_samples], n_proj=cfg.sliced_w2_n_proj)
            row = {
                "step": int(step),
                "loss": float(loss.item()),
                "sliced_w2": sw,
                "V_norm_mean": float(stats["V_norm_mean"]),
                "dist_pos_mean": float(stats["dist_pos_mean"]),
                "dist_neg_mean": float(stats["dist_neg_mean"]),
                "wall": time.time() - t0,
            }
            if _is_checkerboard(dataset_name):
                row["tile_acc"] = tile_acc(s, manifold)
            log_curve.append(row)

            if step % cfg.log_every == 0 or step == 1:
                print(
                    f"step {step:5d}/{cfg.steps} | loss {row['loss']:.4f} "
                    f"| sw {row['sliced_w2']:.4f}"
                    + (f" | tile {row['tile_acc']:.3f}" if "tile_acc" in row else "")
                    + f" | V {row['V_norm_mean']:.3f}"
                    + f" | {row['wall']:.1f}s"
                )

            if step in capture_set:
                snapshots[step] = _to_numpy(s)

    # Final snapshot if not already captured.
    if cfg.steps not in snapshots:
        with torch.no_grad():
            s_final = model.sample(cfg.n_eval_samples, device=cfg.device)
        snapshots[cfg.steps] = _to_numpy(s_final)

    # Persist artifacts.
    out_metrics = {
        "config": {
            **{k: getattr(cfg, k) for k in ("steps", "lr", "log_every", "capture_steps",
                                            "n_eval_samples", "sliced_w2_n_proj", "device")},
            "dataset": dataset_name,
            **(extra_meta or {}),
        },
        "log": log_curve,
        "captured_steps": sorted(snapshots.keys()),
    }
    (out_dir / "metrics.json").write_text(json.dumps(out_metrics, indent=2))

    if cfg.save_samples:
        for step, arr in snapshots.items():
            np.save(out_dir / f"samples_step{step:05d}.npy", arr)
        np.save(out_dir / "target_pool.npy", _to_numpy(target_pool[: cfg.n_eval_samples]))

    print(f"\nwrote metrics -> {out_dir / 'metrics.json'}")
    return out_metrics
