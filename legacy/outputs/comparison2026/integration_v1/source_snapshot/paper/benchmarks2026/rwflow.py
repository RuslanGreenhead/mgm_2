"""RW-Flow-C: equation-faithful, shared-generator paper reimplementation.

Source: https://arxiv.org/html/2609.39271v2, Eq. (5), (7), (72), Alg. 1-2.
No verified public author implementation was found; see RWFlow_NOTES.md.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from torch import Tensor


PROVENANCE = {
    "classification": "equation-faithful paper reimplementation; shared-generator adaptation",
    "paper": "https://arxiv.org/html/2609.39271v2",
    "author_code_url": None,
    "author_code_commit": None,
    "cost": "full squared chordal distance in sphere XYZ / product-circle embedding",
    "teacher": "none",
}


@dataclass
class SinkhornResult:
    conditional: Tensor
    row_error: Tensor
    column_error: Tensor


def chordal_cost(ops, x: Tensor, y: Tensor, scale: float = 1.0) -> Tensor:
    """Full squared embedding distance, NOT half-squared or dimension averaged."""
    difference = ops.embed(x)[:, None, :] - ops.embed(y)[None, :, :]
    return scale * difference.square().sum(-1)


def chordal_cost_gradient(ops, x: Tensor, y: Tensor, scale: float = 1.0) -> Tensor:
    """Riemannian gradient in x, shape (source, target, native coordinate)."""
    difference = x[:, None, :] - y[None, :, :]
    if ops.kind == "sphere":
        return 2 * scale * ops.tangent(x[:, None, :], difference)
    if ops.kind == "torus":
        return 2 * scale * difference.sin()
    raise ValueError("RW-Flow-C supports only sphere and flat torus")


@torch.no_grad()
def sinkhorn_conditional(cost: Tensor, epsilon: float, iterations: int) -> SinkhornResult:
    """Algorithm 2 with uniform marginals and KL(pi || a tensor b).

    epsilon has the same units as cost; it is not a blur standard deviation.
    The last row update makes the returned rows conditional probabilities.
    No diagonal is removed, including for square matrices.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if cost.ndim != 2 or min(cost.shape) < 1:
        raise ValueError("cost must be a nonempty matrix")
    if not cost.is_floating_point() or not torch.isfinite(cost).all():
        raise ValueError("cost must contain finite floating-point values")
    if cost.dtype in (torch.float16, torch.bfloat16):
        cost = cost.float()
    n, m = cost.shape
    log_a, log_b = -math.log(n), -math.log(m)
    log_kernel = -cost.detach() / epsilon
    u = cost.new_zeros(n)
    v = cost.new_zeros(m)
    # Scaled duals u/epsilon, v/epsilon avoid repeated multiply/divide pairs.
    for _ in range(iterations):
        v = -torch.logsumexp(log_a + u[:, None] + log_kernel, dim=0)
        u = -torch.logsumexp(log_b + v[None, :] + log_kernel, dim=1)
    conditional = (log_b + u[:, None] + v[None, :] + log_kernel).exp()
    return SinkhornResult(
        conditional=conditional,
        row_error=(conditional.sum(1) - 1).abs().max(),
        column_error=(m * conditional.mean(0) - 1).abs().max(),
    )


@torch.no_grad()
def sinkhorn_field(
    ops, x: Tensor, real: Tensor, self_bank: Tensor, *, epsilon: float = 0.1,
    iterations: int = 256, cost_scale: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Eq. (72): negative target cost gradient PLUS positive self cost gradient.

    self_bank must be independently sampled by the caller. This routine never
    substitutes x for it, masks matching indices, or differentiates the solver.
    """
    if not math.isfinite(cost_scale) or cost_scale <= 0:
        raise ValueError("cost_scale must be finite and positive")
    x, real, self_bank = x.detach(), real.detach(), self_bank.detach()
    target = sinkhorn_conditional(chordal_cost(ops, x, real, cost_scale), epsilon, iterations)
    own = sinkhorn_conditional(chordal_cost(ops, x, self_bank, cost_scale), epsilon, iterations)
    attraction = -torch.einsum(
        "ij,ijd->id", target.conditional, chordal_cost_gradient(ops, x, real, cost_scale),
    )
    repulsion = torch.einsum(
        "ij,ijd->id", own.conditional, chordal_cost_gradient(ops, x, self_bank, cost_scale),
    )
    field = ops.tangent(x, attraction + repulsion)
    return field, {
        "target_row_error": target.row_error,
        "target_column_error": target.column_error,
        "self_row_error": own.row_error,
        "self_column_error": own.column_error,
        "attraction_norm": ops.inner(x, attraction, attraction).clamp_min(0).sqrt().mean(),
        "repulsion_norm": ops.inner(x, repulsion, repulsion).clamp_min(0).sqrt().mean(),
        "field_norm": ops.inner(x, field, field).clamp_min(0).sqrt().mean(),
    }


@torch.no_grad()
def geometric_targets(ops, x: Tensor, field: Tensor, flow_step: float) -> Tensor:
    """Algorithm 1's stopgrad(exp_x(eta V)); no teacher or unrolling."""
    return ops.exp(x.detach(), flow_step * field.detach()).detach()


class RWFlow:
    def __init__(self, ops, cfg: dict, device):
        from .common import PointGenerator

        if ops.kind not in ("sphere", "torus"):
            raise ValueError("RW-Flow-C supports only sphere and flat torus")
        defaults = dict(
            width=128, depth=3, latent_dim=16, lr=2e-4, batch_size=256,
            cost="chordal", cost_scale=1.0, epsilon=0.1, flow_step=0.1,
            sinkhorn_iters=256, self_batch_size=None, source_batch_size=None,
            weight_decay=0.01, grad_clip=1.0,
        )
        self.cfg = {**defaults, **cfg}
        if self.cfg["cost"] != "chordal":
            raise ValueError("Only the documented RW-Flow-C chordal cost is implemented")
        for key in ("epsilon", "flow_step", "cost_scale", "lr"):
            if not math.isfinite(float(self.cfg[key])) or float(self.cfg[key]) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("latent_dim", "sinkhorn_iters", "batch_size"):
            value = self.cfg[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("source_batch_size", "self_batch_size"):
            value = self.cfg[key]
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise ValueError(f"{key} must be None or a positive integer")
        for key in ("weight_decay", "grad_clip"):
            if not math.isfinite(float(self.cfg[key])) or float(self.cfg[key]) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        self.ops, self.device = ops, torch.device(device)
        self.model = PointGenerator(
            ops, latent_dim=self.cfg["latent_dim"], width=self.cfg["width"], depth=self.cfg["depth"],
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg["lr"], betas=(0.9, 0.999),
            eps=1e-8, weight_decay=self.cfg["weight_decay"],
        )
        self.steps = 0

    def _latent(self, n: int, rng: torch.Generator) -> Tensor:
        parameter = next(self.model.parameters())
        return torch.randn(
            n, self.model.latent_dim, generator=rng, device=rng.device, dtype=parameter.dtype,
        ).to(self.device)

    def update(self, real_batch: Tensor, rng: torch.Generator) -> dict:
        if real_batch.ndim != 2 or len(real_batch) == 0 or real_batch.shape[1] != self.ops.dim:
            raise ValueError("real_batch must be a nonempty matrix of native manifold coordinates")
        parameter = next(self.model.parameters())
        real = real_batch.detach().to(device=self.device, dtype=parameter.dtype)
        if not torch.isfinite(real).all():
            raise ValueError("real_batch must be finite")
        self.model.train()
        n = self.cfg["source_batch_size"] or len(real)
        n_self = self.cfg["self_batch_size"] or n
        self.optimizer.zero_grad(set_to_none=True)
        x = self.model(self._latent(n, rng))
        with torch.no_grad():
            self_bank = self.model(self._latent(n_self, rng))
            field, logs = sinkhorn_field(
                self.ops, x, real, self_bank, epsilon=self.cfg["epsilon"],
                iterations=self.cfg["sinkhorn_iters"], cost_scale=self.cfg["cost_scale"],
            )
            target = geometric_targets(self.ops, x, field, self.cfg["flow_step"])
        loss = self.ops.distance2(x, target).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite RW-Flow geometric regression loss")
        loss.backward()
        # A nonfinite gradient must never silently corrupt the optimizer state.
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg["grad_clip"] or float("inf"), error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.steps += 1
        return {
            "loss": loss.detach(), "grad_norm": norm.detach(),
            "step": loss.new_tensor(self.steps), **logs,
        }

    @torch.no_grad()
    def sample(self, n: int, rng: torch.Generator, nfe: int = 1) -> Tensor:
        if nfe != 1:
            raise ValueError("RW-Flow is a one-step generator; nfe must be 1")
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        was_training = self.model.training
        self.model.eval()
        try:
            return self.model(self._latent(n, rng))
        finally:
            self.model.train(was_training)

    def checkpoint(self) -> dict:
        # Own state is snapshotted; RNG, scheduling and evaluation belong to parent.
        return copy.deepcopy({
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "cfg": self.cfg, "steps": self.steps, "provenance": PROVENANCE,
        })

    def load_checkpoint(self, state: dict) -> None:
        if state["cfg"] != self.cfg:
            raise ValueError("Restore requires the same resolved RW-Flow config")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps = int(state["steps"])


def create_method(ops, cfg: dict, device) -> RWFlow:
    return RWFlow(ops, cfg, device)
