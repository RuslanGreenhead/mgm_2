"""Geometry helpers for Riemannian midpoint generative models.

This module deliberately reuses the interpolation conventions from the RFM
codebase. It is a thin layer for MGM training code: sample the hidden midpoint
flip, evaluate the RFM geodesic interpolant, and differentiate that interpolant
with respect to its scalar time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
from torch.func import jvp

from manifm.manifolds.spd import SPD


ShapeLike = Union[int, Sequence[int], torch.Size]


@dataclass(frozen=True)
class MidpointFlipTime:
    """Container for the hidden midpoint flip used by MGM."""

    t: torch.Tensor
    s: torch.Tensor
    flip: torch.Tensor


def sample_midpoint_flip_time(
    shape: ShapeLike,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
    keepdim: bool = True,
) -> MidpointFlipTime:
    """Sample ``t in [0, 1/2]`` and the hidden flipped time ``s``.

    Args:
        shape: Batch size or explicit time tensor shape.
        device: Device for returned tensors.
        dtype: Floating dtype for returned ``t`` and ``s``.
        generator: Optional PyTorch RNG.
        keepdim: If ``shape`` is an int, return ``(N, 1)`` times when true and
            ``(N,)`` times otherwise.

    Returns:
        ``MidpointFlipTime(t, s, flip)`` where ``flip`` is true exactly when
        ``s = 1 - t``. Otherwise ``s = t``.
    """

    if isinstance(shape, int):
        time_shape: Tuple[int, ...] = (shape, 1) if keepdim else (shape,)
    else:
        time_shape = tuple(shape)

    t = 0.5 * torch.rand(
        time_shape, device=device, dtype=dtype, generator=generator
    )
    flip = torch.rand(time_shape, device=device, generator=generator) < 0.5
    s = torch.where(flip, 1.0 - t, t)
    return MidpointFlipTime(t=t, s=s, flip=flip)


def midpoint_path(
    manifold,
    x0: torch.Tensor,
    x1: torch.Tensor,
    s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the RFM interpolant and its time derivative at ``s``.

    The returned tensors are differentiable with respect to ``x0`` for generator
    updates. SPD follows the special RFM implementation based on
    ``manifold.geodesic``; other simple manifolds use
    ``manifm.manifolds.utils.geodesic`` with ``torch.func.jvp`` under ``vmap``.

    Mesh uses a differentiable local chord surrogate followed by projection.
    The official RFM mesh trainer still uses ``Mesh.solve_path`` with the YAML
    solver settings; this branch is for the implicit generator update, where
    the upstream mesh solver is decorated with ``torch.no_grad``.
    """

    if _is_mesh(manifold):
        return _mesh_local_midpoint_path(manifold, x0, x1, s)

    if isinstance(manifold, SPD):
        return _spd_midpoint_path(manifold, x0, x1, s)

    return _simple_midpoint_path(manifold, x0, x1, s)


def sample_midpoint_path(
    manifold,
    x0: torch.Tensor,
    x1: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, MidpointFlipTime]:
    """Sample midpoint-flip time and return ``(x_tilde, a_t, times)``."""

    times = sample_midpoint_flip_time(
        x0.shape[0], device=x0.device, dtype=x0.dtype, generator=generator
    )
    x_tilde, a_t = midpoint_path(manifold, x0, x1, times.s)
    return x_tilde, a_t, times


def reversal_residual(
    manifold,
    x0: torch.Tensor,
    x1: torch.Tensor,
    s: Optional[torch.Tensor] = None,
) -> dict:
    """Sanity check for geodesic reversal symmetry.

    Compares ``gamma(x0, x1, s)`` with ``gamma(x1, x0, 1 - s)`` and, where
    possible, compares the corresponding velocities after sign reversal.
    Returned values are per-sample squared Riemannian norms.
    """

    if _is_mesh(manifold):
        raise NotImplementedError(
            "reversal_residual for Mesh is not implemented because Mesh paths "
            "must be solved with explicit mesh solver settings."
        )

    if s is None:
        s = torch.rand(x0.shape[0], 1, device=x0.device, dtype=x0.dtype)

    x_fwd, u_fwd = midpoint_path(manifold, x0, x1, s)
    x_rev, u_rev = midpoint_path(manifold, x1, x0, 1.0 - s)

    if hasattr(manifold, "logmap"):
        point_delta = manifold.logmap(x_fwd, x_rev)
    else:
        point_delta = x_rev - x_fwd

    point_residual = _inner(manifold, x_fwd, point_delta, point_delta)
    velocity_residual = _inner(manifold, x_fwd, u_fwd + u_rev, u_fwd + u_rev)
    return {
        "point": point_residual,
        "velocity": velocity_residual,
        "point_mean": point_residual.mean(),
        "velocity_mean": velocity_residual.mean(),
    }


def _spd_midpoint_path(
    manifold: SPD,
    x0: torch.Tensor,
    x1: torch.Tensor,
    s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    time = _time_vector(s, x0)

    def path(t):
        return manifold.geodesic(x0, x1, t)

    x_tilde, a_t = jvp(path, (time,), (torch.ones_like(time),))
    return x_tilde.reshape_as(x0), a_t.reshape_as(x0)


def _simple_midpoint_path(
    manifold,
    x0: torch.Tensor,
    x1: torch.Tensor,
    s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    squeeze_batch = x0.ndim == 1
    if squeeze_batch:
        x0 = x0.unsqueeze(0)
        x1 = x1.unsqueeze(0)

    time = _time_column(s, x0)
    shooting_tangent = manifold.logmap(x0, x1)

    if _needs_finite_difference_velocity(manifold):
        return _finite_difference_midpoint_path(
            manifold, x0, shooting_tangent, time, squeeze_batch
        )

    def path(time_i):
        return manifold.expmap(x0, time_i * shooting_tangent)

    x_tilde, a_t = jvp(path, (time,), (torch.ones_like(time),))
    x_tilde = x_tilde.reshape_as(x0)
    a_t = a_t.reshape_as(x0)

    if squeeze_batch:
        x_tilde = x_tilde.squeeze(0)
        a_t = a_t.squeeze(0)

    return x_tilde, a_t


def _mesh_local_midpoint_path(
    manifold,
    x0: torch.Tensor,
    x1: torch.Tensor,
    s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    squeeze_batch = x0.ndim == 1
    if squeeze_batch:
        x0 = x0.unsqueeze(0)
        x1 = x1.unsqueeze(0)

    time = _time_column(s, x0)
    chord = x1 - x0
    x_tilde = manifold.projx(x0 + time * chord)
    a_t = manifold.proju(x_tilde, chord)

    if squeeze_batch:
        x_tilde = x_tilde.squeeze(0)
        a_t = a_t.squeeze(0)

    return x_tilde, a_t


def _finite_difference_midpoint_path(
    manifold,
    x0: torch.Tensor,
    shooting_tangent: torch.Tensor,
    time: torch.Tensor,
    squeeze_batch: bool,
    eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_tilde = manifold.expmap(x0, time * shooting_tangent)
    lo = (time - eps).clamp_min(0.0)
    hi = (time + eps).clamp_max(1.0)
    denom = (hi - lo).clamp_min(torch.finfo(time.dtype).eps)
    x_lo = manifold.expmap(x0, lo * shooting_tangent)
    x_hi = manifold.expmap(x0, hi * shooting_tangent)
    a_t = (x_hi - x_lo) / denom
    a_t = manifold.proju(x_tilde, a_t)

    if squeeze_batch:
        x_tilde = x_tilde.squeeze(0)
        a_t = a_t.squeeze(0)

    return x_tilde, a_t


def _time_column(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    batch = x.shape[0]
    t = torch.as_tensor(t, device=x.device, dtype=x.dtype)

    if t.ndim == 0:
        return t.expand(batch, 1)
    if t.shape == (batch,):
        return t.reshape(batch, 1)
    if t.shape == (batch, 1):
        return t

    raise ValueError(
        f"Expected scalar, ({batch},), or ({batch}, 1) time tensor; got {tuple(t.shape)}."
    )


def _time_vector(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    batch = x.shape[0]
    t = torch.as_tensor(t, device=x.device, dtype=x.dtype)

    if t.ndim == 0:
        return t.expand(batch)
    if t.shape == (batch,):
        return t
    if t.shape == (batch, 1):
        return t.reshape(batch)

    raise ValueError(
        f"Expected scalar, ({batch},), or ({batch}, 1) time tensor; got {tuple(t.shape)}."
    )


def _inner(manifold, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return manifold.inner(x, u, v)


def _is_mesh(manifold) -> bool:
    return manifold.__class__.__name__ == "Mesh"


def _needs_finite_difference_velocity(manifold) -> bool:
    return manifold.__class__.__name__ == "PoincareBall"
