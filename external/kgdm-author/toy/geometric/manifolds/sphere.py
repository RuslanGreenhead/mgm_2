import torch
import math
from manifolds.base import BaseManifold


class SphereManifold(BaseManifold):
    """
    Class for operations on the unit sphere S^n.

    The unit sphere S^n is the set of points in R^(n+1) with unit norm:
    S^n = {x in R^(n+1) : ||x|| = 1}
    """

    def __init__(self, eps=1e-7):
        """Initialize the sphere manifold.

        Args:
            eps (float): Small constant for numerical stability
        """
        super().__init__(eps)
        self.north_pole = None  # Will be set dynamically based on input dimension

    def _get_north_pole(self, batch_size, dim, device):
        """Get the north pole (0,...,0,1) with proper batch size.

        Args:
            batch_size (int): Number of points
            dim (int): Dimension of the sphere (S^dim)
            device (torch.device): Device to place the tensor on

        Returns:
            torch.Tensor: North pole with shape (batch_size, dim+1)
        """
        if self.north_pole is None or self.north_pole.shape != (batch_size, dim + 1):
            zeros = torch.zeros(batch_size, dim, device=device)
            ones = torch.ones(batch_size, 1, device=device)
            self.north_pole = torch.cat([zeros, ones], dim=-1)
        return self.north_pole

    def project_to_tangent(self, x, v):
        """Project vector v onto the tangent space at x.

        Args:
            x (torch.Tensor): Point on the sphere
            v (torch.Tensor): Vector to project

        Returns:
            torch.Tensor: Projected vector in T_x S^n
        """
        dot = (x * v).sum(dim=-1, keepdim=True)
        return v - dot * x

    def exp_map(self, x0, v):
        """Compute the exponential map at x0 in direction v.

        Args:
            x0 (torch.Tensor): Base point on the sphere
            v (torch.Tensor): Tangent vector at x0

        Returns:
            torch.Tensor: Point on the sphere reached by following the geodesic
        """
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        alpha = v.norm(dim=-1, keepdim=True)
        # clamp to keep cos/sin numerically safe
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1e6, neginf=0.0).clamp_max(50.0)
        direction = v / (alpha + self.eps)
        out = torch.cos(alpha) * x0 + torch.sin(alpha) * direction
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def log_map(self, x0, x1):
        """Compute the logarithmic map from x0 to x1.

        Args:
            x0 (torch.Tensor): Base point on the sphere
            x1 (torch.Tensor): Target point on the sphere

        Returns:
            torch.Tensor: Tangent vector v at x0 such that exp_{x0}(v) = x1
        """
        dot = (x0 * x1).sum(dim=-1).clamp(min=-1 + self.eps, max=1 - self.eps)
        dist = torch.acos(dot)
        sin_dist = torch.sin(dist)

        dist = dist.unsqueeze(-1)
        sin_dist = sin_dist.unsqueeze(-1)
        dot = dot.unsqueeze(-1)

        v = (dist / (sin_dist + self.eps)) * (x1 - dot * x0)
        mask = dist.squeeze(-1) < self.eps
        v[mask] = 0.0
        return v

    def wrap(self, samples):
        """Map points from R^n to S^(n+1) using exponential map at north pole.

        Args:
            samples (torch.Tensor): Points in R^n

        Returns:
            torch.Tensor: Points on S^(n+1)
        """
        batch_size, dim = samples.shape
        north_pole = self._get_north_pole(batch_size, dim, samples.device)
        v = torch.cat([samples, torch.zeros_like(samples[..., :1])], dim=-1)
        return self.exp_map(north_pole, v)

    def unwrap(self, samples):
        """Map points from S^(n+1) to R^n using logarithmic map at north pole.

        Args:
            samples (torch.Tensor): Points on S^(n+1)

        Returns:
            torch.Tensor: Points in R^n
        """
        batch_size, dim = samples.shape
        north_pole = self._get_north_pole(batch_size, dim - 1, samples.device)
        v = self.log_map(north_pole, samples)
        return v[..., :-1]

    def geodesic(self, x0, x1, t):
        """Compute geodesic from x0 to x1 parametrized by t in [0,1].

        Args:
            x0 (torch.Tensor): Start point on the sphere
            x1 (torch.Tensor): End point on the sphere
            t (torch.Tensor): Parameter in [0,1]

        Returns:
            torch.Tensor: Point along the geodesic at parameter t
        """
        v = self.log_map(x0, x1)
        return self.exp_map(x0, t * v)

    def geodesic_velocity(self, x0, x1, t):
        """Compute velocity of geodesic at time t.

        Args:
            x0 (torch.Tensor): Start point on the sphere
            x1 (torch.Tensor): End point on the sphere
            t (torch.Tensor): Parameter in [0,1]

        Returns:
            torch.Tensor: Velocity vector at time t
        """
        v = self.log_map(x0, x1)
        alpha = v.norm(dim=-1, keepdim=True)
        direction = v / (alpha + self.eps)
        return alpha * (-torch.sin(alpha * t) * x0 + torch.cos(alpha * t) * direction)

    def sample(self, batch_size, dim=2, device="cpu"):
        """Sample points uniformly at random on S^dim.

        Args:
            batch_size (int): Number of points to sample
            dim (int): Dimension of the sphere (S^dim)
            device (torch.device): Device to place the tensor on

        Returns:
            torch.Tensor: Uniformly sampled points on S^dim
        """
        x = torch.randn(batch_size, dim + 1, device=device)
        return x / x.norm(dim=-1, keepdim=True)

    def distance(self, x, y, eps=1e-6):
        """
        Compute the spherical distance (great circle distance) between two points.

        Args:
            x (torch.Tensor): Point on the sphere
            y (torch.Tensor): Point on the sphere
            eps (float): Small constant for numerical stability

        Returns:
            torch.Tensor: Spherical distance between x and y
        """
        x = x / (x.norm(dim=-1, keepdim=True) + eps)
        y = y / (y.norm(dim=-1, keepdim=True) + eps)
        dot = (x * y).sum(dim=-1)
        dot = torch.clamp(dot, min=-1 + eps, max=1 - eps)
        return torch.acos(dot)

    def pairwise_distance(
        self, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        # x: [B,3], y: [B,3] -> [B,B]
        x = x / (x.norm(dim=-1, keepdim=True) + eps)
        y = y / (y.norm(dim=-1, keepdim=True) + eps)
        dot = x @ y.t()  # [B,B]
        dot = dot.clamp(-1 + eps, 1 - eps)
        return torch.acos(dot)


def from_2d_to_sphere(points_2d):
    """Wrap (B, 2) points in [-2, 2]^2 onto S^2 via (phi, theta) parametrization."""
    x1 = points_2d[:, 0]
    x2 = points_2d[:, 1]

    phi = math.pi * (x1 / 2.0)
    theta = (x2 + 2.0) / 4.0 * math.pi

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    x = sin_theta * cos_phi
    y = sin_theta * sin_phi
    z = cos_theta

    points_3d = torch.stack([x, y, z], dim=-1)
    return points_3d


def to_spherical_coords(x):
    """Cartesian (B, 3) on S^2 -> (theta in [0, pi], phi in [-pi, pi])."""
    eps = 1e-7
    z = x[:, 2].clamp(-1 + eps, 1 - eps)
    theta = torch.acos(z)
    phi = torch.atan2(x[:, 1], x[:, 0])
    return theta, phi


def grad_sphere(x, k=6, alpha=3.0):
    """Riemannian gradient of f(theta, phi) = alpha * sin(k theta) * sin(k phi)
    on S^2, expressed in ambient R^3."""
    sphere = SphereManifold()
    theta, phi = to_spherical_coords(x)
    df_dtheta = alpha * k * torch.cos(k * theta) * torch.sin(k * phi)
    df_dphi = alpha * k * torch.sin(k * theta) * torch.cos(k * phi)

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)

    e_theta = torch.stack(
        [cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta], dim=-1
    )
    e_phi = torch.stack(
        [-sin_theta * sin_phi, sin_theta * cos_phi, torch.zeros_like(theta)], dim=-1
    )
    grad = (df_dtheta.unsqueeze(-1) * e_theta) + (df_dphi.unsqueeze(-1) * e_phi)
    return sphere.project_to_tangent(x, grad)
