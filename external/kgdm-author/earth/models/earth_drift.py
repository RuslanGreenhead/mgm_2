from __future__ import annotations

import torch
import torch.nn as nn

import drift_core as dc
from models.kernels_spectral import TabulatedMaternS2


class EarthDrift(nn.Module):
    """
    Drift model for the Earth (S^2) experiment.

    drift_type:
      - "euclidean": double-softmax drift in ambient R^3. The generator output
        is a raw R^3 point during training and projected to S^2 only for
        sampling/evaluation.
      - "sphere": double-softmax drift with geodesic distance and log/exp maps.
      - "spectral": Nadaraya-Watson kernel score with the table-backed spectral
        Matern kernel, summed over the matern_kappas lengthscale ladder.
    """

    def __init__(
        self,
        backbone: nn.Module,
        dim_noise: int = 3,
        device: str | torch.device = "cpu",
        *,
        drift_type: str = "sphere",
        temps=(0.02, 0.05, 0.2),
        dist_norm: bool = True,
        grad_form: bool = True,
        kernel: str = "laplace",
        drop_T: bool = True,
        max_step: float | None = 1.0,
        eps: float = 1e-8,
        # spectral-only
        matern_nu: float = 2.5,
        matern_kappas=(1.0, 2.0, 4.0),
        matern_tables: str = "matern_tables.pt",
        spec_norm: bool = False,
    ):
        super().__init__()
        if drift_type not in ("euclidean", "sphere", "spectral"):
            raise ValueError(f"Unknown drift_type: {drift_type}")
        if kernel not in ("laplace", "gauss"):
            raise ValueError(f"Unknown kernel: {kernel}")

        self.net = backbone
        self.dim_noise = dim_noise
        self.device = device
        self.drift_type = drift_type
        self.temps = tuple(float(t) for t in temps)
        self.dist_norm = bool(dist_norm)
        self.grad_form = bool(grad_form)
        self.kernel = kernel
        self.drop_T = bool(drop_T)
        self.max_step = None if max_step is None else float(max_step)
        self.eps = eps
        self.spec_norm = bool(spec_norm)

        if drift_type == "spectral":
            self.spectral_kernels = nn.ModuleList(
                TabulatedMaternS2(
                    nu=float(matern_nu),
                    kappa=float(kappa),
                    table_path=matern_tables,
                )
                for kappa in matern_kappas
            )
        else:
            self.spectral_kernels = None

    # ------------------------------------------------------------------ sampling
    def sample(self, batch_size: int, mode: str = "train", device=None):
        device = device or self.device
        z = torch.randn(batch_size, self.dim_noise, device=device)
        x = self.net(z)
        if self.drift_type == "euclidean":
            # ambient during training; projected to the sphere only for eval
            if mode == "sample":
                return dc.sphere_normalize(x, self.eps)
            return x
        return dc.sphere_normalize(x, self.eps)

    # ------------------------------------------------------------------ drift
    def _clamp_step(self, V: torch.Tensor) -> torch.Tensor:
        """Per-sample cap on the step norm (caps large steps, leaves small
        ones); guards the sphere exp-map from wrapping."""
        if self.max_step is None:
            return V
        V_norm = V.norm(dim=-1, keepdim=True)
        return V * (self.max_step / V_norm.clamp_min(self.max_step))

    def get_loss(self, X_pos: torch.Tensor):
        X = self.sample(batch_size=X_pos.shape[0], mode="train", device=X_pos.device)
        X_neg = X

        with torch.no_grad():
            if self.drift_type == "euclidean":
                V = dc.compute_V_multi(
                    X, X_pos, X_neg, self.temps,
                    dist_norm=self.dist_norm, grad_form=self.grad_form,
                    kernel=self.kernel, drop_T=self.drop_T,
                )
                V = self._clamp_step(V)
                X_tgt = X + V
            elif self.drift_type == "sphere":
                V = dc.compute_V_sphere_multi(
                    X, X_pos, X_neg, self.temps,
                    dist_norm=self.dist_norm, grad_form=self.grad_form,
                    kernel=self.kernel, drop_T=self.drop_T, eps=self.eps,
                )
                V = self._clamp_step(V)
                X_tgt = dc.sphere_exp(X, V, self.eps)
            else:  # spectral
                V = dc.compute_V_spectral_multi(
                    X, X_pos, X_neg, self.spectral_kernels,
                    norm_scale=self.spec_norm, eps=self.eps,
                )
                V = self._clamp_step(V)
                X_tgt = dc.sphere_exp(X, V, self.eps)

        if self.drift_type == "euclidean":
            residual = X - X_tgt
        else:
            residual = dc.sphere_log(X, X_tgt, self.eps)

        loss = residual.pow(2).sum(dim=-1).mean()

        stats = {
            "V_norm_mean": V.norm(dim=-1).mean(),
            "X_norm_mean": X.norm(dim=-1).mean(),
            "X_tgt_norm_mean": X_tgt.norm(dim=-1).mean(),
        }
        return loss, stats
