"""Quick runtime sanity checks for R-MGM modules."""

import torch

from manifm.rmgm import build_critic, build_generator, critic_loss, generator_loss
from rmgm_train import make_experiment


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    for name in ["torus2d", "spd", "hyperbolic", "sphere"]:
        manifold, dim, sampler = make_experiment(name)
        cfg = {
            "generator": {"latent_dim": 4, "hidden_dim": 16, "num_layers": 2},
            "critic": {"hidden_dim": 16, "num_layers": 2, "fourier": 2},
        }
        generator = build_generator(cfg, manifold, dim).to(device)
        critic = build_critic(cfg, manifold, dim).to(device)
        z = torch.randn(8, 4, device=device)
        x1 = sampler.sample(8, device)
        loss_c, _ = critic_loss(manifold, generator, critic, z, x1)
        loss_g, _ = generator_loss(manifold, generator, critic, z, x1)
        loss_g.backward()
        print(
            name,
            f"critic={float(loss_c.detach().cpu()):.4f}",
            f"generator={float(loss_g.detach().cpu()):.4f}",
        )


if __name__ == "__main__":
    main()
