"""Constrained one-step generators for Riemannian MGM experiments.

The modules in this file expose the simple API ``G(z) -> x in M`` and avoid
touching the original RFM training code.  They are intentionally conservative:
an unconstrained MLP predicts raw parameters and a small manifold-specific head
maps them into the representation used by ``manifm.manifolds``.
"""

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from manifm.manifolds.hyperbolic import PoincareBall
from manifm.manifolds.spd import SPD
from manifm.manifolds.sphere import Sphere
from manifm.manifolds.torus import FlatTorus


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class MLP(nn.Module):
    """Plain feed-forward network used by all generator heads."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        activation: str = "silu",
        zero_last: bool = False,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        act = _activation(activation)
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(d_in, d_out))
        else:
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(act())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act())
            layers.append(nn.Linear(hidden_dim, d_out))

        self.net = nn.Sequential(*layers)
        if zero_last:
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ProjectedGenerator(nn.Module):
    """Fallback generator that delegates constraints to ``manifold.projx``."""

    def __init__(self, latent_dim: int, dim: int, manifold: Any, **mlp_kwargs: Any):
        super().__init__()
        self.latent_dim = latent_dim
        self.dim = dim
        self.manifold = manifold
        self.raw = MLP(latent_dim, dim, **mlp_kwargs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.manifold.projx(self.raw(z))


class SphereGenerator(nn.Module):
    """Ambient generator normalized onto the unit sphere."""

    def __init__(
        self,
        latent_dim: int,
        dim: int,
        eps: float = 1e-8,
        **mlp_kwargs: Any,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dim = dim
        self.eps = eps
        self.raw = MLP(latent_dim, dim, **mlp_kwargs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.raw(z)
        norm = x.norm(dim=-1, keepdim=True)
        y = x / norm.clamp_min(self.eps)
        fallback = torch.zeros_like(x)
        fallback[..., 0] = 1.0
        return torch.where(norm > self.eps, y, fallback)


class FlatTorusGenerator(nn.Module):
    """Angle-coordinate generator for the flat torus in ``[0, 2 pi)``."""

    def __init__(
        self,
        latent_dim: int,
        dim: int,
        map_mode: str = "sigmoid",
        eps: float = 1e-6,
        **mlp_kwargs: Any,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dim = dim
        self.map_mode = map_mode
        self.eps = eps
        self.raw = MLP(latent_dim, dim, **mlp_kwargs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.raw(z)
        two_pi = 2.0 * math.pi
        if self.map_mode == "sigmoid":
            return torch.sigmoid(y) * (two_pi - self.eps)
        if self.map_mode == "mod":
            return y.remainder(two_pi)
        raise ValueError(f"Unknown FlatTorus map_mode: {self.map_mode}")


class FlatTorusMixtureGenerator(nn.Module):
    """Mixture-style torus generator with learnable chart centers.

    A plain MLP followed by modulo can represent wrapped distributions, but it
    often does so through long discontinuous stripes. This head keeps the same
    constrained ``G(z) -> M`` API while giving multimodal torus targets a more
    natural parameterization: a latent quantile picks a component, the component
    center lives on the torus, and an MLP predicts a bounded local tangent
    offset around that center.
    """

    def __init__(
        self,
        latent_dim: int,
        dim: int,
        components: int = 4,
        residual_scale: float = 0.35,
        init: str = "grid",
        **mlp_kwargs: Any,
    ):
        super().__init__()
        if components < 1:
            raise ValueError("components must be >= 1")
        self.latent_dim = latent_dim
        self.dim = dim
        self.components = components
        self.two_pi = 2.0 * math.pi
        self.raw = MLP(latent_dim + components, dim, **mlp_kwargs)
        self.centers = nn.Parameter(_init_torus_centers(components, dim, init))
        log_scale = _inverse_softplus(torch.full((components, dim), residual_scale))
        self.log_scale = nn.Parameter(log_scale)
        self.component_logits = nn.Parameter(torch.zeros(components))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        selector = 0.5 * (1.0 + torch.erf(z[..., :1] / math.sqrt(2.0)))
        boundaries = torch.cumsum(torch.softmax(self.component_logits, dim=0), dim=0)[:-1]
        index = torch.searchsorted(boundaries.to(selector), selector.squeeze(-1)).clamp(
            max=self.components - 1
        )
        one_hot = F.one_hot(index, num_classes=self.components).to(z)
        center = self.centers.remainder(self.two_pi)[index]
        scale = F.softplus(self.log_scale)[index]
        residual = torch.tanh(self.raw(torch.cat([z, one_hot], dim=-1))) * scale
        return (center + residual).remainder(self.two_pi)


class FlatTorusGaussianMixtureGenerator(nn.Module):
    """Gaussian-mixture generator in local torus charts.

    This is a more structured version of ``FlatTorusMixtureGenerator``. The
    latent selects one learnable chart center and uses the next ``dim`` latent
    coordinates as Gaussian tangent noise with learnable per-component scale.
    It matches the wrapped Gaussian targets used in the RFM paper-style toy
    experiments while keeping all samples exactly on the torus.
    """

    def __init__(
        self,
        latent_dim: int,
        dim: int,
        components: int = 4,
        residual_scale: float = 0.25,
        init: str = "grid",
        **mlp_kwargs: Any,
    ):
        del mlp_kwargs
        super().__init__()
        if latent_dim < dim + 1:
            raise ValueError(
                f"latent_dim={latent_dim} is too small for a {dim}D torus GMM; "
                f"need at least {dim + 1}."
            )
        if components < 1:
            raise ValueError("components must be >= 1")
        self.latent_dim = latent_dim
        self.dim = dim
        self.components = components
        self.two_pi = 2.0 * math.pi
        self.centers = nn.Parameter(_init_torus_centers(components, dim, init))
        log_scale = _inverse_softplus(torch.full((components, dim), residual_scale))
        self.log_scale = nn.Parameter(log_scale)
        self.component_logits = nn.Parameter(torch.zeros(components))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        selector = 0.5 * (1.0 + torch.erf(z[..., :1] / math.sqrt(2.0)))
        boundaries = torch.cumsum(torch.softmax(self.component_logits, dim=0), dim=0)[:-1]
        index = torch.searchsorted(boundaries.to(selector), selector.squeeze(-1)).clamp(
            max=self.components - 1
        )
        center = self.centers.remainder(self.two_pi)[index]
        scale = F.softplus(self.log_scale)[index]
        residual = z[..., 1 : 1 + self.dim] * scale
        return (center + residual).remainder(self.two_pi)


class SPDGenerator(nn.Module):
    """Cholesky-parameterized generator returning RFM vectorized SPD matrices."""

    def __init__(
        self,
        latent_dim: int,
        dim: int,
        manifold: SPD,
        min_diag: float = 1e-4,
        jitter: float = 1e-6,
        **mlp_kwargs: Any,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dim = dim
        self.manifold = manifold
        self.n = manifold.matdim(dim)
        if manifold.vecdim(self.n) != dim:
            raise ValueError(f"SPD vector dimension {dim} is not triangular")
        self.min_diag = min_diag
        self.jitter = jitter
        self.raw = MLP(latent_dim, dim, **mlp_kwargs)

        tril = torch.tril_indices(row=self.n, col=self.n, offset=0)
        diag_mask = tril[0] == tril[1]
        self.register_buffer("tril_rows", tril[0], persistent=False)
        self.register_buffer("tril_cols", tril[1], persistent=False)
        self.register_buffer("diag_mask", diag_mask, persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        params = self.raw(z)
        batch_shape = params.shape[:-1]
        flat = params.reshape(-1, self.dim)

        L = flat.new_zeros(flat.shape[0], self.n, self.n)
        values = flat.clone()
        values[:, self.diag_mask] = F.softplus(values[:, self.diag_mask]) + self.min_diag
        L[:, self.tril_rows, self.tril_cols] = values

        A = L @ L.transpose(-1, -2)
        eye = torch.eye(self.n, dtype=A.dtype, device=A.device)
        A = A + self.jitter * eye
        return self.manifold.vectorize(A).reshape(*batch_shape, self.dim)


class PoincareBallGenerator(nn.Module):
    """Generator constrained to the open Poincare ball."""

    def __init__(
        self,
        latent_dim: int,
        dim: int,
        manifold: PoincareBall,
        radius_scale: float = 0.995,
        eps: float = 1e-8,
        **mlp_kwargs: Any,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dim = dim
        self.manifold = manifold
        self.radius_scale = radius_scale
        self.eps = eps
        self.raw = MLP(latent_dim, dim, **mlp_kwargs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.raw(z)
        norm = y.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        radius = _poincare_radius(self.manifold, y).to(dtype=y.dtype, device=y.device)
        return y / norm * torch.tanh(norm) * radius * self.radius_scale


class LocalGaussianMixtureGenerator(nn.Module):
    """Local chart Gaussian-mixture generator for simple manifolds.

    The component centers are represented in the ambient coordinates used by the
    RFM manifolds. Forward projection keeps centers valid, tangent noise is
    projected with ``proju``, and samples are returned through ``expmap``.
    """

    def __init__(
        self,
        latent_dim: int,
        dim: int,
        manifold: Any,
        components: int = 4,
        residual_scale: float = 0.25,
        **mlp_kwargs: Any,
    ):
        del mlp_kwargs
        super().__init__()
        if latent_dim < dim + 1:
            raise ValueError(
                f"latent_dim={latent_dim} is too small for a {dim}D local GMM; "
                f"need at least {dim + 1}."
            )
        if components < 1:
            raise ValueError("components must be >= 1")
        self.latent_dim = latent_dim
        self.dim = dim
        self.manifold = manifold
        self.components = components
        self.centers = nn.Parameter(torch.randn(components, dim) * 0.1)
        log_scale = _inverse_softplus(torch.full((components, dim), residual_scale))
        self.log_scale = nn.Parameter(log_scale)
        self.component_logits = nn.Parameter(torch.zeros(components))

    def projected_centers(self) -> torch.Tensor:
        centers = self.centers
        if isinstance(self.manifold, Sphere):
            return centers / centers.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if isinstance(self.manifold, PoincareBall):
            norm = centers.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            radius = _poincare_radius(self.manifold, centers).to(centers)
            return centers / norm * torch.tanh(norm) * radius * 0.98
        return self.manifold.projx(centers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        selector = 0.5 * (1.0 + torch.erf(z[..., :1] / math.sqrt(2.0)))
        probs = torch.softmax(self.component_logits, dim=0)
        boundaries = torch.cumsum(probs, dim=0)[:-1].to(selector)
        index = torch.searchsorted(boundaries, selector.squeeze(-1)).clamp(
            max=self.components - 1
        )
        center = self.projected_centers()[index]
        scale = F.softplus(self.log_scale)[index]
        ambient_noise = z[..., 1 : 1 + self.dim] * scale
        tangent = self.manifold.proju(center, ambient_noise)
        return self.manifold.projx(self.manifold.expmap(center, tangent))


class ProjectedLocalGaussianMixtureGenerator(nn.Module):
    """Local Gaussian-mixture generator for manifolds without ``expmap``.

    Mesh in the RFM codebase exposes ``projx``/``proju`` but not ``expmap``.
    This head keeps centers on the manifold, samples tangent-like local noise,
    then projects the ambient update back to the manifold.
    """

    def __init__(
        self,
        latent_dim: int,
        dim: int,
        manifold: Any,
        components: int = 4,
        residual_scale: float = 0.25,
        **mlp_kwargs: Any,
    ):
        del mlp_kwargs
        super().__init__()
        if latent_dim < dim + 1:
            raise ValueError(
                f"latent_dim={latent_dim} is too small for a {dim}D projected local GMM; "
                f"need at least {dim + 1}."
            )
        if components < 1:
            raise ValueError("components must be >= 1")
        self.latent_dim = latent_dim
        self.dim = dim
        self.manifold = manifold
        self.components = components
        self.centers = nn.Parameter(torch.randn(components, dim) * 0.1)
        log_scale = _inverse_softplus(torch.full((components, dim), residual_scale))
        self.log_scale = nn.Parameter(log_scale)
        self.component_logits = nn.Parameter(torch.zeros(components))

    def projected_centers(self) -> torch.Tensor:
        return self.manifold.projx(self.centers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        selector = 0.5 * (1.0 + torch.erf(z[..., :1] / math.sqrt(2.0)))
        probs = torch.softmax(self.component_logits, dim=0)
        boundaries = torch.cumsum(probs, dim=0)[:-1].to(selector)
        index = torch.searchsorted(boundaries, selector.squeeze(-1)).clamp(
            max=self.components - 1
        )
        center = self.projected_centers()[index]
        scale = F.softplus(self.log_scale)[index]
        ambient_noise = z[..., 1 : 1 + self.dim] * scale
        tangent = self.manifold.proju(center, ambient_noise)
        return self.manifold.projx(center + tangent)


class SphericalMeshMapGenerator(nn.Module):
    """Generate on S^2 and map radial directions onto a closed mesh.

    This is a lightweight genus-0 mesh baseline.  The neural network produces a
    unit direction, a fixed soft nearest-direction radial map moves that point
    onto the mesh scale, and the final mesh projection guarantees feasibility.
    """

    def __init__(
        self,
        latent_dim: int,
        dim: int,
        manifold: Any,
        radial_temperature: float = 80.0,
        vertex_stride: int = 1,
        eps: float = 1e-8,
        **mlp_kwargs: Any,
    ):
        super().__init__()
        if dim != 3:
            raise ValueError("SphericalMeshMapGenerator expects 3D mesh coordinates.")
        if not hasattr(manifold, "v"):
            raise ValueError("SphericalMeshMapGenerator requires a mesh manifold with vertices.")
        self.latent_dim = latent_dim
        self.dim = dim
        self.manifold = manifold
        self.radial_temperature = radial_temperature
        self.eps = eps
        self.raw = MLP(latent_dim, dim, **mlp_kwargs)

        vertices = manifold.v.detach()
        if vertex_stride > 1:
            vertices = vertices[::vertex_stride]
        center = vertices.mean(dim=0)
        offsets = vertices - center
        radii = offsets.norm(dim=-1).clamp_min(eps)
        directions = offsets / radii[:, None]
        self.register_buffer("sphere_map_center", center.clone(), persistent=False)
        self.register_buffer("sphere_map_dirs", directions.clone(), persistent=False)
        self.register_buffer("sphere_map_radii", radii.clone(), persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        direction = self.raw(z)
        norm = direction.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(direction)
        fallback[..., 0] = 1.0
        direction = torch.where(
            norm > self.eps,
            direction / norm.clamp_min(self.eps),
            fallback,
        )
        dirs = self.sphere_map_dirs.to(direction)
        radii = self.sphere_map_radii.to(direction)
        center = self.sphere_map_center.to(direction)
        logits = self.radial_temperature * (direction @ dirs.T)
        radius = torch.softmax(logits, dim=-1) @ radii
        ambient = center + radius[:, None] * direction
        return self.manifold.projx(ambient)


class SphericalMeshLocalGaussianMixtureGenerator(nn.Module):
    """Local Gaussian mixture on S^2 followed by a radial map to the mesh."""

    def __init__(
        self,
        latent_dim: int,
        dim: int,
        manifold: Any,
        components: int = 16,
        residual_scale: float = 0.18,
        radial_temperature: float = 80.0,
        vertex_stride: int = 1,
        eps: float = 1e-8,
        **mlp_kwargs: Any,
    ):
        del mlp_kwargs
        super().__init__()
        if dim != 3:
            raise ValueError(
                "SphericalMeshLocalGaussianMixtureGenerator expects 3D mesh coordinates."
            )
        if latent_dim < 4:
            raise ValueError("sphere_map_gmm needs latent_dim >= 4.")
        if components < 1:
            raise ValueError("components must be >= 1")
        if not hasattr(manifold, "v"):
            raise ValueError("sphere_map_gmm requires a mesh manifold with vertices.")
        self.latent_dim = latent_dim
        self.dim = dim
        self.manifold = manifold
        self.components = components
        self.radial_temperature = radial_temperature
        self.eps = eps

        vertices = manifold.v.detach()
        if vertex_stride > 1:
            vertices = vertices[::vertex_stride]
        center = vertices.mean(dim=0)
        offsets = vertices - center
        radii = offsets.norm(dim=-1).clamp_min(eps)
        directions = offsets / radii[:, None]
        self.register_buffer("sphere_map_center", center.clone(), persistent=False)
        self.register_buffer("sphere_map_dirs", directions.clone(), persistent=False)
        self.register_buffer("sphere_map_radii", radii.clone(), persistent=False)

        self.centers = nn.Parameter(_fibonacci_sphere(components))
        log_scale = _inverse_softplus(torch.full((components, 3), residual_scale))
        self.log_scale = nn.Parameter(log_scale)
        self.component_logits = nn.Parameter(torch.zeros(components))

    def projected_centers(self) -> torch.Tensor:
        return _normalize(self.centers, eps=self.eps)

    @torch.no_grad()
    def initialize_from_mesh_data(self, data: torch.Tensor, iters: int = 40) -> None:
        directions = _normalize(data.to(self.centers) - self.sphere_map_center.to(data), self.eps)
        centers = _unit_kmeans(
            directions,
            self.components,
            iters=iters,
            init=self.projected_centers().detach(),
            eps=self.eps,
        )
        diff = directions[:, None, :] - centers[None, :, :]
        assignment = diff.pow(2).sum(dim=-1).argmin(dim=1)
        scales = torch.empty_like(self.log_scale)
        for k in range(self.components):
            points = directions[assignment == k]
            if points.shape[0] <= 1:
                scales[k].fill_(F.softplus(self.log_scale[k]).mean())
                continue
            center = centers[k].expand(points.shape[0], -1)
            tangent = points - (points * center).sum(dim=-1, keepdim=True) * center
            scales[k] = tangent.std(dim=0, unbiased=False).abs().clamp(0.02, 0.80)

        self.centers.copy_(centers.to(self.centers))
        self.log_scale.copy_(_inverse_softplus(scales.to(self.log_scale)))
        counts = torch.bincount(assignment, minlength=self.components).to(scales)
        probs = (counts + 1.0) / (counts.sum() + self.components)
        self.component_logits.copy_(probs.log().to(self.component_logits))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        selector = 0.5 * (1.0 + torch.erf(z[..., :1] / math.sqrt(2.0)))
        probs = torch.softmax(self.component_logits, dim=0)
        boundaries = torch.cumsum(probs, dim=0)[:-1].to(selector)
        index = torch.searchsorted(boundaries, selector.squeeze(-1)).clamp(
            max=self.components - 1
        )
        center = self.projected_centers()[index]
        scale = F.softplus(self.log_scale)[index]
        noise = z[..., 1:4] * scale
        tangent = noise - (noise * center).sum(dim=-1, keepdim=True) * center
        direction = _normalize(center + tangent, eps=self.eps)
        dirs = self.sphere_map_dirs.to(direction)
        radii = self.sphere_map_radii.to(direction)
        mesh_center = self.sphere_map_center.to(direction)
        logits = self.radial_temperature * (direction @ dirs.T)
        radius = torch.softmax(logits, dim=-1) @ radii
        ambient = mesh_center + radius[:, None] * direction
        return self.manifold.projx(ambient)


def build_generator(cfg: Optional[Any], manifold: Any, dim: int) -> nn.Module:
    """Build a constrained generator from a dict/OmegaConf/object config.

    Recognized fields, with conservative defaults:
    ``latent_dim``/``z_dim``: defaults to ``dim``;
    ``hidden_dim``/``d_model``: defaults to 256;
    ``num_layers``: defaults to 3;
    ``activation``/``actfn``: defaults to ``silu``;
    ``zero_last``: defaults to False.
    """

    cfg = _generator_cfg(cfg)
    latent_dim = int(_cfg_get(cfg, "latent_dim", _cfg_get(cfg, "z_dim", dim)))
    hidden_dim = int(_cfg_get(cfg, "hidden_dim", _cfg_get(cfg, "d_model", 256)))
    num_layers = int(_cfg_get(cfg, "num_layers", 3))
    activation = _cfg_get(cfg, "activation", _cfg_get(cfg, "actfn", "silu"))
    zero_last = bool(_cfg_get(cfg, "zero_last", False))

    mlp_kwargs = dict(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        activation=activation,
        zero_last=zero_last,
    )

    generator_type = str(_cfg_get(cfg, "generator_type", "mlp")).lower()
    if isinstance(manifold, Sphere):
        if generator_type in {"local_gmm", "gmm", "mixture"}:
            return LocalGaussianMixtureGenerator(
                latent_dim=latent_dim,
                dim=dim,
                manifold=manifold,
                components=int(_cfg_get(cfg, "generator_components", 4)),
                residual_scale=float(_cfg_get(cfg, "generator_residual_scale", 0.25)),
                **mlp_kwargs,
            )
        if generator_type != "mlp":
            raise ValueError(f"Unknown generator_type: {generator_type}")
        return SphereGenerator(
            latent_dim=latent_dim,
            dim=dim,
            eps=float(_cfg_get(cfg, "eps", 1e-8)),
            **mlp_kwargs,
        )
    if isinstance(manifold, FlatTorus):
        torus_generator = str(_cfg_get(cfg, "torus_generator", "mlp")).lower()
        if torus_generator in {"mixture", "mixture_head", "charts"}:
            return FlatTorusMixtureGenerator(
                latent_dim=latent_dim,
                dim=dim,
                components=int(_cfg_get(cfg, "torus_components", 4)),
                residual_scale=float(_cfg_get(cfg, "torus_residual_scale", 0.35)),
                init=str(_cfg_get(cfg, "torus_mixture_init", "grid")),
                **mlp_kwargs,
            )
        if torus_generator in {"gmm", "gaussian", "gaussian_mixture"}:
            return FlatTorusGaussianMixtureGenerator(
                latent_dim=latent_dim,
                dim=dim,
                components=int(_cfg_get(cfg, "torus_components", 4)),
                residual_scale=float(_cfg_get(cfg, "torus_residual_scale", 0.25)),
                init=str(_cfg_get(cfg, "torus_mixture_init", "grid")),
                **mlp_kwargs,
            )
        if torus_generator != "mlp":
            raise ValueError(f"Unknown torus_generator: {torus_generator}")
        return FlatTorusGenerator(
            latent_dim=latent_dim,
            dim=dim,
            map_mode=str(_cfg_get(cfg, "torus_map", "sigmoid")),
            eps=float(_cfg_get(cfg, "torus_eps", 1e-6)),
            **mlp_kwargs,
        )
    if isinstance(manifold, SPD):
        if generator_type in {"local_gmm", "gmm", "mixture"}:
            return LocalGaussianMixtureGenerator(
                latent_dim=latent_dim,
                dim=dim,
                manifold=manifold,
                components=int(_cfg_get(cfg, "generator_components", 4)),
                residual_scale=float(_cfg_get(cfg, "generator_residual_scale", 0.25)),
                **mlp_kwargs,
            )
        if generator_type != "mlp":
            raise ValueError(f"Unknown generator_type for SPD: {generator_type}")
        return SPDGenerator(
            latent_dim=latent_dim,
            dim=dim,
            manifold=manifold,
            min_diag=float(_cfg_get(cfg, "spd_min_diag", 1e-4)),
            jitter=float(_cfg_get(cfg, "spd_jitter", 1e-6)),
            **mlp_kwargs,
        )
    if isinstance(manifold, PoincareBall):
        if generator_type in {"local_gmm", "gmm", "mixture"}:
            return LocalGaussianMixtureGenerator(
                latent_dim=latent_dim,
                dim=dim,
                manifold=manifold,
                components=int(_cfg_get(cfg, "generator_components", 4)),
                residual_scale=float(_cfg_get(cfg, "generator_residual_scale", 0.20)),
                **mlp_kwargs,
            )
        if generator_type != "mlp":
            raise ValueError(f"Unknown generator_type: {generator_type}")
        return PoincareBallGenerator(
            latent_dim=latent_dim,
            dim=dim,
            manifold=manifold,
            radius_scale=float(_cfg_get(cfg, "ball_radius_scale", 0.995)),
            eps=float(_cfg_get(cfg, "eps", 1e-8)),
            **mlp_kwargs,
        )

    if generator_type == "sphere_map":
        return SphericalMeshMapGenerator(
            latent_dim=latent_dim,
            dim=dim,
            manifold=manifold,
            radial_temperature=float(_cfg_get(cfg, "sphere_map_temperature", 80.0)),
            vertex_stride=int(_cfg_get(cfg, "sphere_map_vertex_stride", 1)),
            eps=float(_cfg_get(cfg, "eps", 1e-8)),
            **mlp_kwargs,
        )

    if generator_type in {"sphere_map_gmm", "sphere_gmm", "radial_gmm"}:
        return SphericalMeshLocalGaussianMixtureGenerator(
            latent_dim=latent_dim,
            dim=dim,
            manifold=manifold,
            components=int(_cfg_get(cfg, "generator_components", 16)),
            residual_scale=float(_cfg_get(cfg, "generator_residual_scale", 0.18)),
            radial_temperature=float(_cfg_get(cfg, "sphere_map_temperature", 80.0)),
            vertex_stride=int(_cfg_get(cfg, "sphere_map_vertex_stride", 1)),
            eps=float(_cfg_get(cfg, "eps", 1e-8)),
            **mlp_kwargs,
        )

    if generator_type in {"local_gmm", "gmm", "mixture"}:
        return ProjectedLocalGaussianMixtureGenerator(
            latent_dim=latent_dim,
            dim=dim,
            manifold=manifold,
            components=int(_cfg_get(cfg, "generator_components", 4)),
            residual_scale=float(_cfg_get(cfg, "generator_residual_scale", 0.25)),
            **mlp_kwargs,
        )

    return ProjectedGenerator(
        latent_dim=latent_dim,
        dim=dim,
        manifold=manifold,
        **mlp_kwargs,
    )


@torch.no_grad()
def constraint_error(manifold: Any, x: torch.Tensor, reduction: str = "max") -> torch.Tensor:
    """Return a scalar tensor measuring violation of the manifold constraint."""

    if isinstance(manifold, Sphere):
        err = (x.norm(dim=-1) - 1.0).abs()
    elif isinstance(manifold, FlatTorus):
        err = torch.maximum((-x).clamp_min(0.0), (x - 2.0 * math.pi).clamp_min(0.0))
        err = err.amax(dim=-1)
    elif isinstance(manifold, SPD):
        A = manifold.devectorize(x)
        sym_err = (A - A.transpose(-1, -2)).abs().amax(dim=(-1, -2))
        min_eig = torch.linalg.eigvalsh(A).amin(dim=-1)
        eig_err = (-min_eig).clamp_min(0.0)
        err = torch.maximum(sym_err, eig_err)
    elif isinstance(manifold, PoincareBall):
        radius = _poincare_radius(manifold, x).to(dtype=x.dtype, device=x.device)
        err = (x.norm(dim=-1) - radius).clamp_min(0.0)
    else:
        projected = manifold.projx(x)
        err = (projected - x).reshape(x.shape[0], -1).norm(dim=-1)

    if reduction == "max":
        return err.max()
    if reduction == "mean":
        return err.mean()
    if reduction == "none":
        return err
    raise ValueError(f"Unknown reduction: {reduction}")


def _activation(name: str):
    normalized = str(name).lower()
    if normalized in {"silu", "swish"}:
        return nn.SiLU
    if normalized == "relu":
        return nn.ReLU
    if normalized == "gelu":
        return nn.GELU
    if normalized in {"srelu", "softplus"}:
        return nn.Softplus
    if normalized == "sine":
        return Sine
    if normalized in {"tanh", "tanhshrink"}:
        return nn.Tanh
    raise ValueError(f"Unknown activation: {name}")


def _init_torus_centers(components: int, dim: int, init: str) -> torch.Tensor:
    normalized = str(init).lower()
    if normalized == "random":
        return torch.rand(components, dim) * (2.0 * math.pi)
    if normalized != "grid":
        raise ValueError(f"Unknown torus mixture init: {init}")

    side = int(math.ceil(components ** (1.0 / dim)))
    axes = [
        torch.linspace(0.0, 2.0 * math.pi, side + 1, dtype=torch.float32)[:-1]
        + math.pi / side
        for _ in range(dim)
    ]
    mesh = torch.meshgrid(*axes, indexing="ij")
    centers = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
    return centers[:components].clone()


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _fibonacci_sphere(n: int) -> torch.Tensor:
    idx = torch.arange(n, dtype=torch.float32)
    phi = math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - 2.0 * (idx + 0.5) / n
    radius = (1.0 - z.pow(2)).clamp_min(0.0).sqrt()
    theta = idx * phi
    return torch.stack([radius * theta.cos(), radius * theta.sin(), z], dim=-1)


@torch.no_grad()
def _unit_kmeans(
    x: torch.Tensor,
    components: int,
    *,
    iters: int,
    init: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    centers = _normalize(init[:components].clone(), eps=eps)
    if x.shape[0] < components:
        repeats = math.ceil(components / max(x.shape[0], 1))
        centers = x.repeat(repeats, 1)[:components].clone()
    for _ in range(iters):
        dist2 = torch.cdist(x, centers).pow(2)
        assignment = dist2.argmin(dim=1)
        for k in range(components):
            points = x[assignment == k]
            if points.numel() == 0:
                continue
            centers[k] = _normalize(points.mean(dim=0, keepdim=True), eps=eps).squeeze(0)
    return centers


def _generator_cfg(cfg: Optional[Any]) -> Any:
    if cfg is None:
        return {}
    nested = _cfg_get(cfg, "generator", None)
    if nested is not None:
        return nested
    rmgm = _cfg_get(cfg, "rmgm", None)
    nested = _cfg_get(rmgm, "generator", None)
    return cfg if nested is None else nested


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(cfg, key, default)


def _poincare_radius(manifold: PoincareBall, like: torch.Tensor) -> torch.Tensor:
    c = getattr(manifold, "c", 1.0)
    if not torch.is_tensor(c):
        c = torch.as_tensor(c, dtype=like.dtype, device=like.device)
    return c.rsqrt()
