"""Regression checks for the maintained method package."""

import unittest

import torch

from rmgm.geometry import ManifoldOps
from rmgm.methods.rmgm import create_method
from paper.benchmarks2026.mgm import create_method as archived_create_method
from paper.benchmarks2026.common import ManifoldOps as ArchivedManifoldOps


class CanonicalMethodsTest(unittest.TestCase):
    def test_default_rmgm_matches_2026_reference(self):
        config = {"critic_warmup": 0, "critic_steps": 1, "width": 16, "depth": 2, "latent_dim": 4}
        torch.manual_seed(17)
        current = create_method(ManifoldOps("sphere", 3), config, "cpu")
        torch.manual_seed(17)
        reference = archived_create_method(ArchivedManifoldOps("sphere", 3), config, "cpu")
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(current.model.parameters(), reference.model.parameters())))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(current.critic.parameters(), reference.critic.parameters())))
        z = torch.randn(8, 4)
        self.assertTrue(torch.equal(current.model(z), reference.model(z)))

    def test_gmm_generator_trains_and_obeys_geometry(self):
        for kind, dim in (("sphere", 3), ("torus", 2)):
            with self.subTest(kind=kind):
                ops = ManifoldOps(kind, dim)
                config = {"generator": "gmm", "gmm_components": 3, "latent_dim": 6,
                          "critic_warmup": 0, "critic_steps": 1, "width": 16, "depth": 2}
                method = create_method(ops, config, "cpu")
                rng = torch.Generator().manual_seed(19)
                real = ops.base(8, rng, "cpu")
                result = method.update(real, rng)
                samples = method.sample(16, rng)
                self.assertTrue(torch.isfinite(result["loss"]))
                self.assertTrue(torch.isfinite(samples).all())
                if kind == "sphere":
                    self.assertTrue(torch.allclose(samples.norm(dim=-1), torch.ones(16), atol=1e-5))
                else:
                    self.assertTrue(((samples >= 0) & (samples < 2 * torch.pi)).all())
                self.assertIsNotNone(method.model.centers.grad)
                self.assertIsNotNone(method.model.log_scale.grad)


if __name__ == "__main__":
    unittest.main()
