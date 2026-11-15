"""Riemannian MGM extensions for the RFM codebase."""

from .generators import (
    FlatTorusGenerator,
    PoincareBallGenerator,
    ProjectedLocalGaussianMixtureGenerator,
    ProjectedGenerator,
    SPDGenerator,
    SphereGenerator,
    build_generator,
    constraint_error,
)
from .geometry import (
    MidpointFlipTime,
    midpoint_path,
    reversal_residual,
    sample_midpoint_flip_time,
    sample_midpoint_path,
)
from .critics import TangentCritic, build_critic
from .losses import critic_loss, generator_loss, metric_inner

__all__ = [
    "FlatTorusGenerator",
    "PoincareBallGenerator",
    "ProjectedLocalGaussianMixtureGenerator",
    "ProjectedGenerator",
    "SPDGenerator",
    "SphereGenerator",
    "build_generator",
    "constraint_error",
    "MidpointFlipTime",
    "midpoint_path",
    "reversal_residual",
    "sample_midpoint_flip_time",
    "sample_midpoint_path",
    "TangentCritic",
    "build_critic",
    "critic_loss",
    "generator_loss",
    "metric_inner",
]
