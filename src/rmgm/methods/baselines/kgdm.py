"""KGDM with the pinned author's spherical field and a native-torus adaptation.

See KGDM_NOTES.md for provenance, finite-batch versus population semantics,
and controlled-suite changes. No teachers or scalar MMD optimization are used.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor, nn


AUTHOR_URL = "https://github.com/jcarrascopollo/kgd_thesis"
AUTHOR_COMMIT = "5860d212a1e53b53a499e251f2d087fcf0a5fc00"
AUTHOR_FIELD_SHA256 = "39008dd140a304851dc200216d035c714fe3264c9319ee28bb534caefbf0a5d0"
PAPER_URL = "https://arxiv.org/html/2605.10727"
PROVENANCE = {
    "paper": PAPER_URL, "author_url": AUTHOR_URL, "commit": AUTHOR_COMMIT,
    "field_sha256": AUTHOR_FIELD_SHA256, "hash_normalization": "LF",
    "sphere": "author_field_shared_backbone",
    "torus": "torus_adaptation_of_author_field",
}
DEFAULTS = {
    "width": 128,
    "depth": 3,
    "latent_dim": 16,
    "lr": 2e-4,
    "batch_size": 256,
    "kernel": "laplace",
    "temps": (0.02, 0.05, 0.2),
    "bandwidth_scale": 1.0,
    "dist_norm": True,
    "drop_T": True,
    "step_size": 1.0,
    "max_step": 1.0,
    "grad_clip": 1.0,
    "weight_decay": 0.01,
    "betas": (0.9, 0.95),
    "warmup_steps": 0,
    "ema_decay": 0.999,
    "sample_ema": True,
    "eps": 1e-8,
}


@lru_cache(maxsize=1)
def author_core():
    """Load only the unmodified torch-only field, without sys.path collisions."""
    path = Path(__file__).resolve().parents[4] / "external/kgdm-author/earth/drift_core.py"
    if not path.is_file():
        raise FileNotFoundError(f"Clone {AUTHOR_URL} at {AUTHOR_COMMIT} into {path.parents[1]}")
    digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if digest != AUTHOR_FIELD_SHA256:
        raise RuntimeError(f"KGDM author field checksum changed: {digest}")
    spec = importlib.util.spec_from_file_location("_kgdm_pinned_author_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _kernel_name(kernel):
    kernel = "gauss" if kernel == "gaussian" else kernel
    if kernel not in ("laplace", "gauss"):
        raise ValueError("KGDM kernel must be 'laplace', 'gauss', or 'gaussian'")
    return kernel


def _temperatures(temps):
    temps = tuple(float(t) for t in temps)
    if not temps or any(not math.isfinite(t) or t <= 0 for t in temps):
        raise ValueError("KGDM temperatures must be nonempty, finite, and positive")
    return temps


@torch.no_grad()
def intrinsic_field(ops, x: Tensor, real: Tensor, *, temps=(0.02, 0.05, 0.2),
                    dist_norm=True, kernel="laplace", drop_T=True, eps=1e-8) -> Tensor:
    """Detached finite-batch kernel-gradient field, negatives exactly x.

    Sphere: execute the pinned author's compute_V_sphere_multi unchanged.
    Torus: extend that same affinity algebra using wrapped log-map tangents.
    This is NOT the row-only population KDE score-ratio estimator.
    """
    kernel = _kernel_name(kernel)
    temps = _temperatures(temps)
    if x.ndim != 2 or real.ndim != 2 or x.shape[1] != real.shape[1]:
        raise ValueError("KGDM expects [N, q] generated and [M, q] real points")
    if x.shape[0] < 2 or real.shape[0] < 1:
        raise ValueError("KGDM needs >=2 generated points and >=1 real point")
    if ops.kind == "sphere":
        return author_core().compute_V_sphere_multi(
            x, real, x, temps, dist_norm=dist_norm, grad_form=True,
            kernel=kernel, drop_T=drop_T, eps=eps,
        )
    if ops.kind != "torus":
        raise ValueError(f"Unsupported KGDM geometry: {ops.kind}")

    # Keep the author's raw-distance direction, detached batch scale, diagonal
    # exclusion, double-softmax, cross-mass weights, and plain temperature sum.
    log_pos = ops.log(x[:, None, :], real[None, :, :])
    log_neg = ops.log(x[:, None, :], x[None, :, :])
    dist_pos = ops.inner(x[:, None, :], log_pos, log_pos).clamp_min(0).sqrt()
    dist_neg = ops.inner(x[:, None, :], log_neg, log_neg).clamp_min(0).sqrt()
    wd_pos, wd_neg = dist_pos, dist_neg
    if dist_norm:
        mean_d = torch.cat((dist_pos, dist_neg), dim=1).mean().clamp_min(1e-6)
        wd_pos, wd_neg = dist_pos / mean_d, dist_neg / mean_d
    if kernel == "gauss":
        wd_pos, wd_neg = wd_pos.square(), wd_neg.square()
        dir_pos, dir_neg = log_pos, log_neg
    else:
        dir_pos = log_pos / dist_pos.clamp_min(eps).unsqueeze(-1)
        dir_neg = log_neg / dist_neg.clamp_min(eps).unsqueeze(-1)
    eye = torch.eye(len(x), device=x.device, dtype=x.dtype)
    wd_neg = wd_neg + eye * 1e6
    dir_neg = dir_neg.masked_fill(eye.bool().unsqueeze(-1), 0.0)

    field = torch.zeros_like(x)
    for temperature in temps:
        logits = torch.cat((-wd_pos / temperature, -wd_neg / temperature), dim=1)
        affinity = (logits.softmax(-1) * logits.softmax(-2)).sqrt()
        a_pos, a_neg = affinity.split((len(real), len(x)), dim=1)
        w_pos = a_pos * a_neg.sum(1, keepdim=True)
        w_neg = a_neg * a_pos.sum(1, keepdim=True)
        drift = ((w_pos.unsqueeze(-1) * dir_pos).sum(1)
                 - (w_neg.unsqueeze(-1) * dir_neg).sum(1))
        if not drop_T:
            drift = drift * ((2.0 if kernel == "gauss" else 1.0) / temperature)
        field = field + drift
    return ops.tangent(x, field)


@torch.no_grad()
def drift_target(ops, x: Tensor, real: Tensor, *, step_size=1.0, max_step=1.0,
                 **field_options) -> tuple[Tensor, Tensor, Tensor]:
    """Return the detached exponential-map target, raw field, and capped step."""
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("KGDM step_size must be finite and positive")
    if max_step is not None and (not math.isfinite(max_step) or max_step <= 0):
        raise ValueError("KGDM max_step must be None or finite and positive")
    field = intrinsic_field(ops, x, real, **field_options)
    step = ops.tangent(x, field * step_size)
    if max_step is not None:
        norm = ops.inner(x, step, step).clamp_min(0).sqrt().unsqueeze(-1)
        step = step * (max_step / norm.clamp_min(max_step))
    target = ops.exp(x, step)
    return target.detach(), field.detach(), step.detach()


def _cpu_snapshot(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _require_finite(value: Tensor, message: str, error=FloatingPointError):
    finite = torch.isfinite(value).all()
    if value.device.type == "cuda":
        # Device assertion avoids copying a per-step boolean back to the host.
        torch._assert_async(finite, message)
    elif not finite:
        raise error(message)


class KGDM:
    """Controlled-suite generator adapter; update owns optimizer and optional EMA."""

    def __init__(self, ops, cfg: dict, device):
        from rmgm.geometry import PointGenerator

        if ops.kind not in ("sphere", "torus"):
            raise ValueError(f"Unsupported KGDM geometry: {ops.kind}")
        self.ops = ops
        self.cfg = {**DEFAULTS, **copy.deepcopy(cfg)}
        self.cfg["kernel"] = _kernel_name(self.cfg["kernel"])
        self.cfg["temps"] = _temperatures(self.cfg["temps"])
        c = self.cfg
        for key in ("lr", "bandwidth_scale", "step_size", "eps"):
            if not math.isfinite(c[key]) or c[key] <= 0:
                raise ValueError(f"KGDM {key} must be finite and positive")
        for key in ("max_step", "grad_clip"):
            if c[key] is not None and (not math.isfinite(c[key]) or c[key] <= 0):
                raise ValueError(f"KGDM {key} must be None or finite and positive")
        if not 0 <= c["ema_decay"] < 1:
            raise ValueError("KGDM ema_decay must be in [0, 1)")
        if int(c["warmup_steps"]) != c["warmup_steps"] or c["warmup_steps"] < 0:
            raise ValueError("KGDM warmup_steps must be a nonnegative integer")
        self.device = torch.device(device)
        self.model: nn.Module = PointGenerator(
            ops, latent_dim=c["latent_dim"], width=c["width"], depth=c["depth"],
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=c["lr"], betas=tuple(c["betas"]),
            weight_decay=c["weight_decay"],
        )
        self.ema_model = copy.deepcopy(self.model).requires_grad_(False).eval()
        self.steps = 0
        if ops.kind == "sphere":
            author_core()

    @property
    def fidelity(self):
        return ("author_field_shared_backbone" if self.ops.kind == "sphere"
                else "torus_adaptation_of_author_field")

    def update(self, real_batch: Tensor, rng: torch.Generator) -> dict:
        c = self.cfg
        dtype = next(self.model.parameters()).dtype
        real = real_batch.detach().to(device=self.device, dtype=dtype)
        if real.ndim != 2 or real.shape[1] != self.ops.dim or len(real) < 2:
            raise ValueError("KGDM update expects a real batch [N >= 2, ops.dim]")
        _require_finite(real, "KGDM real batch contains nonfinite values", ValueError)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        z = torch.randn(len(real), self.model.latent_dim, generator=rng,
                        device=self.device, dtype=dtype)
        x = self.model(z)
        target, field, step = drift_target(
            self.ops, x, real,
            temps=tuple(t * c["bandwidth_scale"] for t in c["temps"]),
            dist_norm=c["dist_norm"], kernel=c["kernel"], drop_T=c["drop_T"],
            eps=c["eps"], step_size=c["step_size"], max_step=c["max_step"],
        )
        residual = self.ops.log(x, target)
        loss = self.ops.inner(x, residual, residual).mean()
        _require_finite(loss, "Nonfinite KGDM detached-target loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), c["grad_clip"] or float("inf"),
        )
        _require_finite(grad_norm, "Nonfinite KGDM parameter gradient")
        # Match the author's LambdaLR step-0 convention when warmup is enabled.
        lr_scale = min(self.steps / max(1, c["warmup_steps"]), 1.0) if c["warmup_steps"] else 1.0
        for group in self.optimizer.param_groups:
            group["lr"] = c["lr"] * lr_scale
        self.optimizer.step()
        self.steps += 1
        with torch.no_grad():
            for average, parameter in zip(self.ema_model.parameters(), self.model.parameters()):
                average.mul_(c["ema_decay"]).add_(parameter, alpha=1 - c["ema_decay"])
            for average, buffer in zip(self.ema_model.buffers(), self.model.buffers()):
                average.copy_(buffer)
        return {
            "loss": loss.detach(), "grad_norm": grad_norm.detach(),
            "drift_norm": field.norm(dim=-1).mean(),
            "step_norm": step.norm(dim=-1).mean(),
            "lr": loss.detach().new_tensor(self.optimizer.param_groups[0]["lr"]),
            "step": loss.detach().new_tensor(self.steps),
        }

    @torch.no_grad()
    def sample(self, n: int, rng: torch.Generator, nfe: int = 1) -> Tensor:
        if nfe != 1:
            raise ValueError("KGDM is a one-generator-evaluation sampler: nfe must be 1")
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError("KGDM sample n must be a nonnegative integer")
        model = self.ema_model if self.cfg["sample_ema"] else self.model
        was_training = model.training
        model.eval()
        try:
            z = torch.randn(n, self.model.latent_dim, generator=rng,
                            device=self.device, dtype=next(model.parameters()).dtype)
            return model(z)
        finally:
            model.train(was_training)

    def checkpoint(self) -> dict:
        return _cpu_snapshot({
            "method": "kgdm", "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(), "ema_model": self.ema_model.state_dict(),
            "steps": self.steps, "config": self.cfg, "fidelity": self.fidelity,
            "provenance": PROVENANCE,
        })

    def load_checkpoint(self, state: dict):
        if state["config"] != self.cfg or state["fidelity"] != self.fidelity:
            raise ValueError("KGDM checkpoint config/geometry does not match this adapter")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.ema_model.load_state_dict(state["ema_model"])
        self.steps = int(state["steps"])


def create_method(ops, cfg: dict, device) -> KGDM:
    return KGDM(ops, cfg, device)
