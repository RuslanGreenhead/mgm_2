"""Conditional extension of the maintained hidden-flip R-MGM objective."""

from __future__ import annotations

import copy
import torch

from .geometry import mean_inner, path, target_sphere
from .networks import Critic, Generator


class PromoterRMGM:
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        width = int(config.get("width", 256))
        blocks = int(config.get("blocks", 20))
        self.generator = Generator(width, blocks).to(device)
        self.critic = Critic(int(config.get("critic_width", width)), blocks).to(device)
        self.ema = copy.deepcopy(self.generator).eval()
        self.opt_g = torch.optim.AdamW(self.generator.parameters(), lr=float(config.get("lr_g", 2e-4)), weight_decay=0)
        self.opt_c = torch.optim.AdamW(self.critic.parameters(), lr=float(config.get("lr_c", 2e-4)), weight_decay=0)
        self.iteration = 0
        self.generator_updates = 0
        self.critic_updates = 0

    def observation(self, labels: torch.Tensor, signal: torch.Tensor, rng: torch.Generator):
        noise = torch.randn((*labels.shape, 4), device=self.device, generator=rng)
        x0 = self.generator(noise, signal)
        target = target_sphere(labels).to(self.device)
        t = torch.rand((len(labels), 1, 1), device=self.device, generator=rng) * 0.5
        hidden = torch.randint(0, 2, (len(labels), 1, 1), device=self.device, generator=rng)
        s = t + hidden * (1.0 - 2.0 * t)
        x, velocity = path(x0, target, s)
        return t, x, velocity

    def update(self, labels: torch.Tensor, signal: torch.Tensor, rng: torch.Generator) -> dict:
        labels, signal = labels.to(self.device), signal.to(self.device)
        self.generator.train()
        self.critic.train()
        warmup = self.iteration < int(self.config.get("critic_warmup", 500))
        n_critic = 1 if warmup else int(self.config.get("critic_steps", 5))
        for _ in range(n_critic):
            with torch.no_grad():
                t, x, velocity = self.observation(labels, signal, rng)
            self.opt_c.zero_grad(set_to_none=True)
            residual = self.critic(t, x, signal) - velocity
            critic_loss = mean_inner(residual, residual)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.config.get("grad_clip", 1.0)))
            self.opt_c.step()
            self.critic_updates += 1

        result = {"critic_loss": float(critic_loss.detach()), "generator_updated": False}
        if not warmup:
            self.critic.requires_grad_(False)
            try:
                self.opt_g.zero_grad(set_to_none=True)
                t, x, velocity = self.observation(labels, signal, rng)
                field = self.critic(t, x, signal)
                # Frozen critic weights; gradients through its input reach the generator.
                generator_loss = 2 * mean_inner(field, velocity) - mean_inner(field, field)
                generator_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), float(self.config.get("grad_clip", 1.0)))
                self.opt_g.step()
            finally:
                self.critic.requires_grad_(True)
            decay = float(self.config.get("ema_decay", 0.999))
            with torch.no_grad():
                for avg, live in zip(self.ema.parameters(), self.generator.parameters()):
                    avg.lerp_(live, 1.0 - decay)
            self.generator_updates += 1
            result.update(generator_loss=float(generator_loss.detach()), generator_updated=True)
        self.iteration += 1
        return result

    @torch.no_grad()
    def sample(self, signal: torch.Tensor, rng: torch.Generator, use_ema: bool = True) -> torch.Tensor:
        model = self.ema if use_ema else self.generator
        model.eval()
        signal = signal.to(self.device)
        noise = torch.randn((*signal.shape[:2], 4), device=self.device, generator=rng)
        return model(noise, signal)

    def state_dict(self) -> dict:
        return {"generator": self.generator.state_dict(), "critic": self.critic.state_dict(),
                "ema": self.ema.state_dict(), "opt_g": self.opt_g.state_dict(),
                "opt_c": self.opt_c.state_dict(), "iteration": self.iteration,
                "generator_updates": self.generator_updates, "critic_updates": self.critic_updates,
                "config": self.config}

    def load_state_dict(self, state: dict) -> None:
        self.generator.load_state_dict(state["generator"])
        self.critic.load_state_dict(state["critic"])
        self.ema.load_state_dict(state["ema"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_c.load_state_dict(state["opt_c"])
        self.iteration = state["iteration"]
        self.generator_updates = state["generator_updates"]
        self.critic_updates = state["critic_updates"]
