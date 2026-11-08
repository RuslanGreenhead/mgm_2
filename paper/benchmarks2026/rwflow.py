"""RW-Flow-C: equation-faithful, shared-generator paper reimplementation.

Source: https://arxiv.org/html/2609.39271v2, Eq. (5), (7), (72), Alg. 1-2.
No verified public author implementation was found; see RWFlow_NOTES.md.
"""

from __future__ import annotations

import copy
import math
import threading
from dataclasses import dataclass

import torch
from torch import Tensor


PROVENANCE = {
    "classification": "equation-faithful paper reimplementation; shared-generator adaptation",
    "paper": "https://arxiv.org/html/2609.39271v2",
    "author_code_url": None,
    "author_code_commit": None,
    "paper_sinkhorn_iterations": None,
    "paper_sinkhorn_tolerance": None,
    "paper_sinkhorn_stopping": "Algorithm 2 uses a symbolic fixed count L, not a tolerance",
    "cost": "full squared chordal distance in sphere XYZ / product-circle embedding",
    "teacher": "none",
}


@dataclass
class SinkhornResult:
    conditional: Tensor
    row_error: Tensor
    column_error: Tensor
    scaling_used: Tensor


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


def _sinkhorn_log(cost: Tensor, epsilon: float, iterations: int) -> Tensor:
    """Original tested log-domain reference, also accepting a leading batch axis."""
    n, m = cost.shape[-2:]
    log_a, log_b = -math.log(n), -math.log(m)
    log_kernel = -cost / epsilon
    u = cost.new_zeros(cost.shape[:-1])
    v = cost.new_zeros(cost.shape[:-2] + cost.shape[-1:])
    for _ in range(iterations):
        v = -torch.logsumexp(log_a + u.unsqueeze(-1) + log_kernel, dim=-2)
        u = -torch.logsumexp(log_b + v.unsqueeze(-2) + log_kernel, dim=-1)
    return (log_b + u.unsqueeze(-1) + v.unsqueeze(-2) + log_kernel).exp()


def _sinkhorn_scaling(cost: Tensor, epsilon: float, iterations: int) -> tuple[Tensor, bool]:
    """Equivalent finite-L scaling, or reference fallback outside a safe range."""
    n, m = cost.shape[-2:]
    shifted = (cost - cost.amin(dim=-1, keepdim=True)) / epsilon
    # With max(v)=1 each iteration, v>=exp(-R) and u<=exp(R).
    # R below this limit keeps all intermediates and final masses normal.
    limit = 0.5 * (-math.log(torch.finfo(cost.dtype).tiny) - math.log(m) - 2.0)
    if not (torch.isfinite(shifted).all() & (shifted.amax() <= limit)):
        return _sinkhorn_log(cost, epsilon, iterations), False
    kernel = (-shifted).exp()
    tiny = torch.finfo(cost.dtype).tiny
    # A row shift changes the initial source scaling. Compute the FIRST target
    # update from the unshifted log kernel to preserve the reference at every L.
    log_v = -math.log(m) - torch.logsumexp(-math.log(n) - cost / epsilon, dim=-2)
    v = (log_v - log_v.amax(dim=-1, keepdim=True)).exp()
    u = torch.matmul(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(tiny).reciprocal()
    transposed = kernel.transpose(-2, -1)
    for _ in range(iterations - 1):
        denominator = torch.matmul(transposed, u.unsqueeze(-1)).squeeze(-1)
        v = (n / m) / denominator.clamp_min(tiny)
        v = v / v.amax(dim=-1, keepdim=True).clamp_min(tiny)
        denominator = torch.matmul(kernel, v.unsqueeze(-1)).squeeze(-1)
        u = denominator.clamp_min(tiny).reciprocal()
    return u.unsqueeze(-1) * kernel * v.unsqueeze(-2), True


@torch.no_grad()
def _graph64_scaling_body(cost: Tensor, epsilon: float, iterations: int) -> Tensor:
    """Capture-safe copy of finite-L scaling algebra; caller guards its range."""
    n, m = cost.shape[-2:]
    shifted = (cost - cost.amin(dim=-1, keepdim=True)) / epsilon
    kernel = (-shifted).exp()
    tiny = torch.finfo(cost.dtype).tiny
    # Preserve the FIRST unshifted target update, including for L=1.
    log_v = -math.log(m) - torch.logsumexp(-math.log(n) - cost / epsilon, dim=-2)
    v = (log_v - log_v.amax(dim=-1, keepdim=True)).exp()
    u = torch.matmul(kernel, v.unsqueeze(-1)).squeeze(-1).clamp_min(tiny).reciprocal()
    transposed = kernel.transpose(-2, -1)
    for _ in range(iterations - 1):
        denominator = torch.matmul(transposed, u.unsqueeze(-1)).squeeze(-1)
        v = (n / m) / denominator.clamp_min(tiny)
        v = v / v.amax(dim=-1, keepdim=True).clamp_min(tiny)
        denominator = torch.matmul(kernel, v.unsqueeze(-1)).squeeze(-1)
        u = denominator.clamp_min(tiny).reciprocal()
    return u.unsqueeze(-1) * kernel * v.unsqueeze(-2)


class _Graph64Solve:
    """Private buffers, one capture, serialized replays, and owned results."""

    def __init__(self, cost: Tensor, epsilon: float, iterations: int):
        self.lock = threading.Lock()
        self.replays = 0
        self.device = cost.device
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)
            self.graph = torch.cuda.CUDAGraph()
            self.cost = torch.empty_like(cost)
            current = torch.cuda.current_stream(self.device)
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                self.cost.copy_(cost)
                for _ in range(3):
                    warmup = _graph64_scaling_body(self.cost, epsilon, iterations)
                del warmup
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.output = _graph64_scaling_body(self.cost, epsilon, iterations)
            current.wait_stream(self.stream)

    def replay(self, cost: Tensor) -> Tensor:
        with self.lock, torch.cuda.device(self.device):
            current = torch.cuda.current_stream(self.device)
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                cost.record_stream(self.stream)
                self.cost.copy_(cost)
                self.graph.replay()
                owned = self.output.clone()
            current.wait_stream(self.stream)
            owned.record_stream(current)
            self.replays += 1
            return owned


_GRAPH64_CACHE: dict[tuple, _Graph64Solve] = {}
_GRAPH64_CACHE_LOCK = threading.Lock()


def clear_graph64_cache() -> None:
    """Release opt-in graph buffers between probes, never during training."""
    with _GRAPH64_CACHE_LOCK:
        for entry in _GRAPH64_CACHE.values():
            with entry.lock:
                entry.stream.synchronize()
                entry.graph.reset()
        _GRAPH64_CACHE.clear()


@torch.no_grad()
def _sinkhorn_graph64(cost: Tensor, epsilon: float, iterations: int) -> tuple[Tensor, bool]:
    """Guard outside capture; graph only the detached, deterministic FP64 solve."""
    shifted = (cost - cost.amin(dim=-1, keepdim=True)) / epsilon
    m = cost.shape[-1]
    limit = 0.5 * (-math.log(torch.finfo(cost.dtype).tiny) - math.log(m) - 2.0)
    if not (torch.isfinite(shifted).all() & (shifted.amax() <= limit)):
        return _sinkhorn_log(cost, epsilon, iterations), False
    if cost.device.type != "cuda":
        return _graph64_scaling_body(cost, epsilon, iterations), True
    key = (cost.device, tuple(cost.shape), float(epsilon), iterations)
    with _GRAPH64_CACHE_LOCK:
        if key not in _GRAPH64_CACHE:
            _GRAPH64_CACHE[key] = _Graph64Solve(cost, epsilon, iterations)
        # Keep cache release from racing with an enqueued replay.
        return _GRAPH64_CACHE[key].replay(cost), True


@torch.no_grad()
def sinkhorn_conditional(
    cost: Tensor, epsilon: float, iterations: int, *, solver: str = "log", validate_finite: bool = True,
) -> SinkhornResult:
    """Algorithm 2 with uniform marginals and KL(pi || a tensor b).

    epsilon has the same units as cost; it is not a blur standard deviation.
    The last row update makes the returned rows conditional probabilities.
    No diagonal is removed. Leading batch axes represent independent solves.
    No tolerance or early stopping is applied. Scaling is opt-in and guarded;
    scaling64 promotes only the solve and casts weights back before the field.
    graph64 opts into the same FP64 algebra with cached CUDA-graph replay;
    CPU uses its eager body. Returned weights never alias private graph buffers.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if cost.ndim < 2 or min(cost.shape) < 1:
        raise ValueError("cost must be a nonempty matrix or batch of matrices")
    if not cost.is_floating_point() or (validate_finite and not torch.isfinite(cost).all()):
        raise ValueError("cost must contain finite floating-point values")
    if solver not in ("log", "scaling", "scaling64", "graph64"):
        raise ValueError("solver must be 'log', 'scaling', 'scaling64', or 'graph64'")
    cost = cost.detach()
    if cost.dtype in (torch.float16, torch.bfloat16):
        cost = cost.float()
    if solver == "graph64":
        conditional, scaling_used = _sinkhorn_graph64(cost.double(), epsilon, iterations)
        conditional = conditional.to(dtype=cost.dtype)
    elif solver in ("scaling", "scaling64"):
        solve_cost = cost.double() if solver == "scaling64" else cost
        conditional, scaling_used = _sinkhorn_scaling(solve_cost, epsilon, iterations)
        conditional = conditional.to(dtype=cost.dtype)
    else:
        conditional, scaling_used = _sinkhorn_log(cost, epsilon, iterations), False
    m = cost.shape[-1]
    return SinkhornResult(
        conditional=conditional,
        row_error=(conditional.sum(-1) - 1).abs().amax(-1),
        column_error=(m * conditional.mean(-2) - 1).abs().amax(-1),
        scaling_used=cost.new_full(cost.shape[:-2], int(scaling_used)),
    )


@torch.no_grad()
def sinkhorn_field(
    ops, x: Tensor, real: Tensor, self_bank: Tensor, *, epsilon: float = 0.1,
    iterations: int = 256, cost_scale: float = 1.0, solver: str = "log", validate_finite: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Eq. (72): negative target cost gradient PLUS positive self cost gradient.

    self_bank must be independently sampled by the caller. This routine never
    substitutes x for it, masks matching indices, or differentiates the solver.
    """
    if not math.isfinite(cost_scale) or cost_scale <= 0:
        raise ValueError("cost_scale must be finite and positive")
    x, real, self_bank = x.detach(), real.detach(), self_bank.detach()
    target_cost, self_cost = chordal_cost(ops, x, real, cost_scale), chordal_cost(ops, x, self_bank, cost_scale)
    if target_cost.shape == self_cost.shape:
        # Two independent couplings share kernel launches, not marginal mass.
        joint = sinkhorn_conditional(
            torch.stack((target_cost, self_cost)), epsilon, iterations,
            solver=solver, validate_finite=validate_finite,
        )
        target, own = [SinkhornResult(*(getattr(joint, key)[i] for key in (
            "conditional", "row_error", "column_error", "scaling_used",
        ))) for i in range(2)]
    else:
        target = sinkhorn_conditional(target_cost, epsilon, iterations, solver=solver, validate_finite=validate_finite)
        own = sinkhorn_conditional(self_cost, epsilon, iterations, solver=solver, validate_finite=validate_finite)
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
        "target_scaling_used": target.scaling_used,
        "self_scaling_used": own.scaling_used,
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
            sinkhorn_iters=256, sinkhorn_solver="log", sinkhorn_tol=None,
            self_batch_size=None, source_batch_size=None,
            weight_decay=0.01, grad_clip=1.0,
        )
        self.cfg = {**defaults, **cfg}
        if self.cfg["cost"] != "chordal":
            raise ValueError("Only the documented RW-Flow-C chordal cost is implemented")
        if self.cfg["sinkhorn_solver"] not in ("log", "scaling", "scaling64", "graph64"):
            raise ValueError("sinkhorn_solver must be 'log', 'scaling', 'scaling64', or 'graph64'")
        if self.cfg["sinkhorn_tol"] is not None:
            raise ValueError("Fixed-L RW-Flow has no early stopping; sinkhorn_tol must be None")
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
                solver=self.cfg["sinkhorn_solver"], validate_finite=False,
            )
            target = geometric_targets(self.ops, x, field, self.cfg["flow_step"])
        loss = self.ops.distance2(x, target).mean()
        loss.backward()
        # Consolidate device checks into ONE scalar synchronization before step.
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg["grad_clip"] or float("inf"), error_if_nonfinite=False,
        )
        if not (torch.isfinite(real).all() & torch.isfinite(loss) & torch.isfinite(norm)):
            raise FloatingPointError("Nonfinite RW-Flow data, loss, or gradient; optimizer was not stepped")
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
        saved_cfg = {"sinkhorn_solver": "log", "sinkhorn_tol": None, **state["cfg"]}
        if saved_cfg != self.cfg:
            raise ValueError("Restore requires the same resolved RW-Flow config")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        # worker's map_location=device also moves CPU Adam counters to CUDA.
        # Non-capturable, non-fused AdamW reads these scalars on the host.
        for group in self.optimizer.param_groups:
            if not group.get("capturable", False) and not group.get("fused", False):
                for parameter in group["params"]:
                    counter = self.optimizer.state.get(parameter, {}).get("step")
                    if torch.is_tensor(counter):
                        self.optimizer.state[parameter]["step"] = counter.cpu()
        self.steps = int(state["steps"])


def create_method(ops, cfg: dict, device) -> RWFlow:
    return RWFlow(ops, cfg, device)
