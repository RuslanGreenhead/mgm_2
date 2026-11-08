"""Matched conditional networks and local MGM/RFM/RMF/G-LSD objectives.

These are explicitly local product-manifold implementations, not executions of
the author's SDF network or checkpoints. Time runs data(0) -> Gaussian/Haar(1)
for flow training, and sampling integrates backwards from 1 to 0.
"""

import torch
from torch import nn
from torch.func import jvp

from .geometry import exp, hat, inner, path, prior, quaternion_rotation, step, vee


def mlp(inp, out, width):
    return nn.Sequential(nn.Linear(inp, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, out))


class ObjectEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(6, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.aggregate = mlp(66, 64, width)

    def forward(self, data, ids):
        unique, inverse = ids.unique(sorted=True, return_inverse=True)
        features = self.point(data.clouds[unique])
        pooled = torch.cat((features.mean(1), features.amax(1), data.scales[unique]), -1)
        return self.aggregate(pooled)[inverse]


class Generator(nn.Module):
    def __init__(self, width=128, latent=16):
        super().__init__()
        self.latent = latent
        self.encoder = ObjectEncoder(width)
        self.net = mlp(latent + 64, 6, width)
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, data, ids):
        update = self.net(torch.cat((z, self.encoder(data, ids)), -1))
        return quaternion_rotation(z[:, :4]) @ exp(update[:, :3]), z[:, 4:7] + update[:, 3:]


class Field(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.encoder = ObjectEncoder(width)
        self.net = mlp(14 + 64, 6, width)

    def encode(self, data, ids):
        return self.encoder(data, ids)

    def forward(self, r, p, s, t, condition):
        return self.net(torch.cat((r.flatten(-2), p, s, t, condition), -1))


def mgm_objective(generator, critic, data, rng, batch, critic_update, weight=1.0):
    ids, r1, p1 = data.sample(batch, rng)
    z = torch.randn(batch, generator.latent, generator=rng, device=data.device)
    folded = 0.5 * torch.rand(batch, 1, generator=rng, device=data.device)
    time = torch.where(torch.rand(batch, 1, generator=rng, device=data.device) < 0.5, folded, 1-folded)
    if critic_update:
        with torch.no_grad():
            r0, p0 = generator(z, data, ids)
            rt, pt, velocity = path(r0, p0, r1, p1, time)
    else:
        r0, p0 = generator(z, data, ids)
        rt, pt, velocity = path(r0, p0, r1, p1, time)
    pred = critic(rt, pt, folded, folded, critic.encode(data, ids))
    if critic_update:
        return inner(pred-velocity, pred-velocity, weight).mean() / 6
    return (2 * inner(pred, velocity, weight) - inner(pred, pred, weight)).mean() / 6


def flow_objective(method, field, data, rng, batch, weight=1.0):
    ids, rdata, pdata = data.sample(batch, rng)
    rnoise, pnoise = prior(batch, rng, data.device)
    t = torch.rand(batch, 1, generator=rng, device=data.device)
    s = torch.rand(batch, 1, generator=rng, device=data.device) * t
    with torch.no_grad():
        rt, pt, velocity = path(rdata, pdata, rnoise, pnoise, t)
    condition = field.encode(data, ids)
    if method == "rfm":
        diff = field(rt, pt, t, t, condition) - velocity
        return inner(diff, diff, weight).mean() / 6
    if method == "rmf_ccadapt":
        # Material derivative in a body frame includes the SO(3) connection.
        diagonal = torch.rand(batch, 1, generator=rng, device=data.device) < 0.25
        s = torch.where(diagonal, t, s)
        pred, material = jvp(lambda r, p, end: field(r, p, s, end, condition),
                             (rt, pt, t),
                             (rt @ hat(velocity[:, :3]), velocity[:, 3:], torch.ones_like(t)))
        connection = torch.cat((0.5 * torch.linalg.cross(velocity[:, :3], pred[:, :3]),
                                torch.zeros_like(pred[:, 3:])), -1)
        target = (velocity - (t-s) * (material + connection)).detach()
        diff = pred - target
        return inner(diff, diff, weight).mean() / 6
    if method == "glsd":
        # Backward flow map Exp_x[-(t-s) u(x,s,t)], differentiated at its endpoint.
        def endpoint(start):
            return step(rt, pt, field(rt, pt, start, t, condition), -(t-start))
        (rs, ps), (dr, dp) = jvp(endpoint, (s,), (torch.ones_like(s),))
        tangent = torch.cat((vee(rs.transpose(-1, -2) @ dr), dp), -1)
        with torch.no_grad():
            reference = field(rs.detach(), ps.detach(), s, s, condition.detach())
        sd = inner(tangent-reference, tangent-reference, weight).mean() / 6
        fm = field(rt, pt, t, t, condition) - velocity
        return sd + inner(fm, fm, weight).mean() / 6
    raise ValueError(method)


@torch.no_grad()
def generate(method, model, data, object_index, rng, count, steps=1):
    ids = torch.full((count,), object_index, device=data.device, dtype=torch.long)
    if method == "mgm":
        if steps != 1:
            raise ValueError("MGM is a direct 1-NFE generator; --steps must be 1")
        z = torch.randn(count, model.latent, generator=rng, device=data.device)
        return model(z, data, ids)
    r, p = prior(count, rng, data.device)
    condition = model.encode(data, ids)
    for i in range(steps):
        t = torch.full((count, 1), 1-i/steps, device=data.device)
        s = torch.full((count, 1), 1-(i+1)/steps, device=data.device)
        pred = model(r, p, t if method == "rfm" else s, t, condition)
        r, p = step(r, p, pred, s-t)
    return r, p
