"""Source-derived Riemannian MeanFlow, with the benchmark method interface.

Loss and TMLP ported from haolsun/Riemannian_MeanFlow at AUTHOR_COMMIT.
The released architecture contains Meta copyright notices; see MEANFLOW_NOTES.md.
This is not a flow-matching surrogate or a generation-teacher method.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor, nn

AUTHOR_URL = "https://github.com/haolsun/Riemannian_MeanFlow"
AUTHOR_COMMIT = "45d96bcd09a271ea3a770b2014865dabbac9eeeb"
PAPER_URL = "https://arxiv.org/abs/2603.10718v3"


class _TimeLinear(nn.Module):
    """Author ConcatLinear_v2: independent additive biases for both times."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self._layer = nn.Linear(d_in, d_out)
        self._hyper_bias = nn.Linear(1, d_out, bias=False)
        nn.init.zeros_(self._hyper_bias.weight)
        self._hyper_bias_r = nn.Linear(1, d_out, bias=False)
        nn.init.zeros_(self._hyper_bias_r.weight)

    def forward(self, r: Tensor, t: Tensor, x: Tensor) -> Tensor:
        return (self._layer(x) + self._hyper_bias(t.reshape(-1, 1))
                + self._hyper_bias_r(r.reshape(-1, 1)))


class _TimeSwish(nn.Module):
    """Author TimeDependentSwish, including the beta(t)-gamma(r) gate."""

    def __init__(self, dim: int):
        super().__init__()
        hidden = min(64, dim * 4)
        self.beta = nn.Sequential(nn.Linear(1, hidden), nn.Softplus(),
                                  nn.Linear(hidden, dim), nn.Softplus())
        self.gamma = nn.Sequential(nn.Linear(1, hidden), nn.Softplus(),
                                   nn.Linear(hidden, dim), nn.Softplus())

    def forward(self, r: Tensor, t: Tensor, x: Tensor) -> Tensor:
        beta = self.beta(t.reshape(-1, 1))
        gamma = self.gamma(r.reshape(-1, 1))
        return x * torch.sigmoid(x * (beta - gamma))


class NativeTMLP(nn.Module):
    """Portable author TMLP; depth is the number of main affine layers.

    The optional author Fourier encoder uses sin features followed by cos
    features, unlike common.embed's cos/sin ordering. No time Fourier encoder
    or zero-final-layer initialization is introduced.
    """

    def __init__(self, dim: int, width: int = 128, depth: int = 3,
                 fourier: int = 0):
        super().__init__()
        if dim < 1 or width < 1 or depth < 2 or fourier < 0:
            raise ValueError("Require dim/width > 0, depth >= 2, fourier >= 0")
        self.fourier = int(fourier)
        input_dim = dim * self.fourier * 2 if self.fourier else dim
        layers = [_TimeLinear(input_dim, width)]
        for _ in range(depth - 2):
            layers.extend([_TimeSwish(width), _TimeLinear(width, width)])
        layers.extend([_TimeSwish(width), _TimeLinear(width, dim)])
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor, r: Tensor, t: Tensor) -> Tensor:
        if self.fourier:
            features = [torch.sin((i + 1) * x) for i in range(self.fourier)]
            features += [torch.cos((i + 1) * x) for i in range(self.fourier)]
            x = torch.cat(features, dim=-1)
        for layer in self.layers:
            x = layer(r, t, x)
        return x


class MeanFlowField(nn.Module):
    def __init__(self, ops: Any, width: int, depth: int, torus_fourier: int):
        super().__init__()
        self.ops = ops
        self.net = NativeTMLP(int(ops.dim), width, depth,
                             torus_fourier if ops.kind == "torus" else 0)

    def forward(self, x: Tensor, r: Tensor, t: Tensor) -> Tensor:
        return self.ops.tangent(x, self.net(x, r, t))


def geodesic_with_velocity(ops: Any, real: Tensor, noise: Tensor,
                           t: Tensor) -> tuple[Tensor, Tensor]:
    """Author orientation: data at t=0, uniform noise at t=1."""
    with torch.no_grad():
        direction = ops.log(real, noise)

        def path(time: Tensor) -> Tensor:
            return ops.proj(ops.exp(real, time * direction))

        return torch.func.jvp(path, (t,), (torch.ones_like(t),))


def rmf_loss_terms(model: nn.Module, ops: Any, real: Tensor, noise: Tensor,
                   r: Tensor, t: Tensor) -> dict[str, Tensor]:
    """Native get_loss/get_loss_pc_grad on explicitly supplied random inputs.

    Both the interpolation and its velocity are outside reverse-mode AD. The
    *entire* target is stopped; in particular no gradient flows through the
    JVP derivative, including for RMF-MT's second objective. As in the author
    implementation, the sphere derivative is the ambient JVP, not reprojected.
    """
    n = real.shape[0]
    r, t = r.detach().reshape(n, 1), t.detach().reshape(n, 1)
    xt, v = geodesic_with_velocity(ops, real.detach(), noise.detach(), t)
    u, dudt = torch.func.jvp(model, (xt, r, t),
                            (v, torch.zeros_like(r), torch.ones_like(t)))
    correction = ((t - r) * dudt).detach()
    target = (v - correction).detach()
    diff = u - target
    loss = ops.inner(xt, diff, diff).mean()
    loss1 = ops.inner(xt, u - v, u - v).mean()
    loss2 = ops.inner(xt, u, 2 * correction).mean()
    return {"loss": loss, "loss1": loss1, "loss2": loss2,
            "xt": xt, "velocity": v, "u": u, "dudt": dudt,
            "target": target, "correction": correction}


def pcgrad_pair(g1: Tensor, g2: Tensor, reduction: str = "mean") -> Tensor:
    """Two fully shared tasks, equivalent to the author's default PCGrad.

    The upstream default merges by mean (the paper's Eq. 16 uses sum).
    With only two tasks its random task shuffles do not change the result.
    """
    if reduction not in {"mean", "sum"}:
        raise ValueError("pcgrad_reduction must be mean or sum")
    dot = torch.dot(g1, g2)
    conflict = dot.clamp_max(0)
    tiny = torch.finfo(g1.dtype).tiny
    p1 = g1 - conflict / g2.square().sum().clamp_min(tiny) * g2
    p2 = g2 - conflict / g1.square().sum().clamp_min(tiny) * g1
    merged = p1 + p2
    return merged / 2 if reduction == "mean" else merged


def _require_finite(value: Tensor, message: str) -> None:
    if value.device.type == "cpu":
        if not bool(value):
            raise FloatingPointError(message)
    else:
        # Keep routine GPU updates free of tensor-to-host scalar conversions.
        torch._assert_async(value, message)


class RiemannianMeanFlow:
    """CPU/CUDA agnostic; construction never starts a training run."""

    def __init__(self, ops: Any, cfg: dict, device: str | torch.device):
        if ops.kind not in {"sphere", "torus"}:
            raise ValueError("RMF adapter supports sphere and torus only")
        self.ops = ops
        self.device = torch.device(device)
        self.cfg = {"width": 128, "depth": 3, "latent_dim": 16,
                    "lr": 2e-4, "batch_size": 256, "weight_decay": 0.01,
                    "prop_equal_times": 0.75, "time_distribution": "uniform",
                    "variant": "rmf", "torus_fourier": 1,
                    "pcgrad_reduction": "mean", "grad_clip": None, **cfg}
        variant = str(self.cfg["variant"]).lower().replace("-", "_")
        aliases = {"mf": "rmf", "core": "rmf", "pcgrad": "rmf_mt"}
        self.variant = aliases.get(variant, variant)
        if self.variant not in {"rmf", "rmf_mt"}:
            raise ValueError("variant must be rmf or rmf_mt")
        self.cfg["variant"] = self.variant
        if not 0 <= self.cfg["prop_equal_times"] <= 1:
            raise ValueError("prop_equal_times must be in [0, 1]")
        if self.cfg["time_distribution"] not in {"uniform", "lognormal"}:
            raise ValueError("time_distribution must be uniform or lognormal")
        if self.cfg["pcgrad_reduction"] not in {"mean", "sum"}:
            raise ValueError("pcgrad_reduction must be mean or sum")
        self.model = MeanFlowField(ops, int(self.cfg["width"]),
                                   int(self.cfg["depth"]),
                                   int(self.cfg["torus_fourier"])).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.cfg["lr"],
                                           weight_decay=self.cfg["weight_decay"])
        self.updates = 0
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.metadata = {
            "method": "RMF-MT" if self.variant == "rmf_mt" else "RMF",
            "author_url": AUTHOR_URL, "author_commit": AUTHOR_COMMIT,
            "paper_url": PAPER_URL,
            "fidelity": "source-derived native loss/TMLP; benchmark adaptation",
            "common_generator": False, "teacher": False,
            "parameter_count": self.parameter_count,
            "torus_periodic_features": ops.kind == "torus" and
                                       bool(self.cfg["torus_fourier"]),
            "derivative": "native ambient JVP; stopped target",
            "scheduler": "none; parent owns schedule/budget",
        }

    @torch.no_grad()
    def sample_times(self, n: int, rng: torch.Generator,
                     dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        options = {"device": self.device, "dtype": dtype, "generator": rng}
        if self.cfg["time_distribution"] == "uniform":
            t = torch.rand(n, **options)
            r = torch.rand(n, **options) * t
        else:
            # Match LogNormal(-0.4, 1).sample().sigmoid(), not logistic-normal.
            t = (-0.4 + torch.randn(n, **options)).exp()
            r = (-0.4 + torch.randn(n, **options)).exp()
            r, t = torch.minimum(r, t).sigmoid(), torch.maximum(r, t).sigmoid()
        n_equal = int(self.cfg["prop_equal_times"] * n)
        r[:n_equal] = t[:n_equal]
        return r[:, None], t[:, None]

    def update(self, real_batch: Tensor, rng: torch.Generator) -> dict:
        self.model.train()
        dtype = next(self.model.parameters()).dtype
        real = real_batch.detach().to(device=self.device, dtype=dtype)
        if real.ndim != 2 or real.shape[1] != self.ops.dim or not len(real):
            raise ValueError("real_batch must have shape (n>0, ops.dim)")
        self.optimizer.zero_grad(set_to_none=True)
        noise = self.ops.base(len(real), rng, self.device, dtype=dtype)
        r, t = self.sample_times(len(real), rng, dtype)
        terms = rmf_loss_terms(self.model, self.ops, real, noise, r, t)
        _require_finite(torch.isfinite(terms["loss"]), "Nonfinite RMF loss")
        if self.variant == "rmf":
            terms["loss"].backward()
        else:
            params = list(self.model.parameters())
            g1 = torch.autograd.grad(terms["loss1"], params, retain_graph=True)
            g2 = torch.autograd.grad(terms["loss2"], params)
            merged = pcgrad_pair(torch.cat([g.reshape(-1) for g in g1]),
                                 torch.cat([g.reshape(-1) for g in g2]),
                                 self.cfg["pcgrad_reduction"])
            start = 0
            for p in params:
                p.grad = merged[start:start + p.numel()].reshape_as(p).clone()
                start += p.numel()
        params = list(self.model.parameters())
        finite_grad = torch.stack([torch.isfinite(p.grad).all() for p in params
                                   if p.grad is not None]).all()
        _require_finite(finite_grad, "Nonfinite RMF gradient")
        if self.cfg["grad_clip"] is not None:
            nn.utils.clip_grad_norm_(params, self.cfg["grad_clip"])
        self.optimizer.step()
        self.updates += 1
        return {key: terms[key].detach() for key in ("loss", "loss1", "loss2")} | {
            "updates": self.updates, "parameter_count": self.parameter_count,
            "equal_time_fraction": (r == t).float().mean().detach()}

    @torch.no_grad()
    def sample(self, n: int, rng: torch.Generator, nfe: int = 1) -> Tensor:
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a nonnegative integer")
        if not isinstance(nfe, int) or nfe < 1:
            raise ValueError("nfe must be a positive integer")
        dtype = next(self.model.parameters()).dtype
        z = self.ops.base(n, rng, self.device, dtype=dtype)
        if n == 0:
            return z
        was_training = self.model.training
        self.model.eval()
        try:
            ts = torch.linspace(0, 1, nfe + 1, device=self.device,
                                dtype=dtype).flip(0)
            for r, t in zip(ts[1:], ts[:-1]):
                rr, tt = r.expand(n, 1), t.expand(n, 1)
                z = self.ops.exp(z, -(tt - rr) * self.model(z, rr, tt))
                z = self.ops.proj(z)
        finally:
            self.model.train(was_training)
        return z

    def checkpoint(self) -> dict:
        return copy.deepcopy({"model": self.model.state_dict(),
                              "optimizer": self.optimizer.state_dict(),
                              "updates": self.updates, "cfg": self.cfg,
                              "metadata": self.metadata})

    def load_checkpoint(self, state: dict) -> None:
        if state["cfg"] != self.cfg:
            raise ValueError("Construct RMF with the checkpoint cfg before loading")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])


def create_method(ops: Any, cfg: dict, device: str | torch.device) -> RiemannianMeanFlow:
    return RiemannianMeanFlow(ops, cfg, device)


def author_config(dataset: str, source: str = "repo") -> dict:
    """Explicit reference settings, never selected or applied automatically.

    Epoch budgets are metadata: the parent must translate to its own saved
    split and update schedule. See notes for paper-versus-repository conflicts.
    """
    names = ("volcano", "earthquake", "fire", "flood", "general", "glycine",
             "proline", "prepro", "rna")
    if dataset not in names or source not in {"repo", "paper"}:
        raise ValueError("Unknown author dataset or config source")
    i = names.index(dataset)
    layers = (12, 10, 10, 7, 4, 10, 4, 4, 6)
    repo_batch = (8192, 8192, 8192, 8192, 2048, 4096, 4096, 4096, 4096)
    paper_batch = (8192, 4096, 8192, 8192, 4096, 2048, 2048, 2048, 2048)
    repo_epochs = (2000, 700, 600, 900, 10000, 10000, 10000, 10000, 1450)
    paper_epochs = (2000, 700, 600, 700, 5000, 5000, 5000, 5000, 2000)
    repo_wd = (0.01, 0.001, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 0.01)
    return {"width": 2048 if i < 4 else 512, "depth": layers[i],
            "lr": 1e-4 if source == "repo" else 5e-4,
            "weight_decay": repo_wd[i] if source == "repo" else 0.01,
            "batch_size": (repo_batch if source == "repo" else paper_batch)[i],
            "prop_equal_times": 0.75, "time_distribution": "uniform",
            "torus_fourier": 0, "variant": "rmf_mt",
            "pcgrad_reduction": "mean" if source == "repo" else "sum",
            "author_min_epochs": (100 if i < 4 else 1000 if i == 8 else 8000)
                                  if source == "repo" else None,
            "author_max_epochs": (repo_epochs if source == "repo" else paper_epochs)[i],
            "author_scheduler": "CosineAnnealingLR per epoch; eta_min=1e-6",
            "author_config_source": source}
