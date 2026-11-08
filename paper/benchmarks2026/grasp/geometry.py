"""SO(3) x R3 product geometry on SE(3), not a bi-invariant SE(3) metric.

Angular velocities are body-trivialized; translation velocities are in the
object frame. RoMa implements the rotation maps and PyTorch the linear algebra.
"""

import roma
import torch


def hat(v):
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)


def vee(m):
    return torch.stack((m[..., 2, 1] - m[..., 1, 2],
                        m[..., 0, 2] - m[..., 2, 0],
                        m[..., 1, 0] - m[..., 0, 1]), -1) * 0.5


def exp(v):
    return roma.rotvec_to_rotmat(v)


def log(r):
    return roma.rotmat_to_rotvec(r)


def quaternion_rotation(q):
    norm = q.norm(dim=-1, keepdim=True)
    identity = torch.zeros_like(q)
    identity[..., 3] = 1
    return roma.unitquat_to_rotmat(torch.where(norm > 1e-7, q / norm.clamp_min(1e-7), identity))


def prior(n, rng, device, dtype=torch.float32):
    # Same per-pose latent stream as the MGM generator enables paired draws.
    z = torch.randn(n, 16, generator=rng, device=device, dtype=dtype)
    return quaternion_rotation(z[:, :4]), z[:, 4:7]


def path(r0, p0, r1, p1, t):
    omega = log(r0.transpose(-1, -2) @ r1)
    velocity = torch.cat((omega, p1 - p0), -1)
    return r0 @ exp(t * omega), p0 + t * (p1 - p0), velocity


def step(r, p, v, dt):
    return r @ exp(dt * v[..., :3]), p + dt * v[..., 3:]


def inner(u, v, translation_weight=1.0):
    return (u[..., :3] * v[..., :3]).sum(-1) + translation_weight * (u[..., 3:] * v[..., 3:]).sum(-1)


def rotation_residual(r):
    eye = torch.eye(3, device=r.device, dtype=r.dtype)
    return {"orthogonality_max": float((r.transpose(-1, -2) @ r - eye).abs().max()),
            "determinant_error_max": float((torch.linalg.det(r) - 1).abs().max())}
