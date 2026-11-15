"""CPU checks for the supplement's metrics and fixed split contract."""
import unittest
from unittest.mock import patch

import torch

import controlled_suite as suite


class ProtocolTests(unittest.TestCase):
    def test_split_matches_author(self):
        manifold, dim, data, indices = suite.load_data("fire")
        expected = torch.utils.data.random_split(range(12809), [10249, 1280, 1280],
                                                 generator=torch.Generator().manual_seed(0))
        for key, subset in zip(("train", "val", "test"), expected):
            self.assertTrue(torch.equal(indices[key], torch.tensor(subset.indices)))
        self.assertEqual(len(set(indices["train"].tolist()) & set(indices["test"].tolist())), 0)

    def test_fixed_kernel_and_identity(self):
        m = suite.FlatTorus()
        x = torch.tensor([[0.1, 0.2], [0.9, 1.1], [3.0, 4.0]])
        ref, _ = suite.reference(m, x)
        before = dict(ref)
        metrics = suite.metrics(m, x, x, ref)
        self.assertAlmostEqual(metrics["mmd2_biased"], 0.0, places=12)
        self.assertAlmostEqual(metrics["symmetric_nn"], 0.0, places=6)
        suite.metrics(m, x + 0.5, x, ref)
        self.assertEqual(ref, before)

    def test_torus_wrap(self):
        x = torch.tensor([[0.01]], dtype=torch.float64)
        y = torch.tensor([[2 * torch.pi - 0.01]], dtype=torch.float64)
        self.assertAlmostEqual(float(suite.pairwise(suite.FlatTorus(), x, y)), 0.02)

    def test_mixture_logits_have_no_pathwise_gradient(self):
        cfg = suite.config("general")
        g = suite.build_generator({"generator": suite.asdict(cfg)}, suite.FlatTorus(), 2)
        g(torch.randn(32, cfg.latent_dim)).sum().backward()
        self.assertIsNone(g.component_logits.grad)
        self.assertIsNotNone(g.centers.grad)

    def test_no_manifold_attribute_required_for_sphere_mlp(self):
        cfg = suite.asdict(suite.config("fire"))
        cfg["generator_type"] = "mlp"
        model = suite.build_generator({"generator": cfg}, suite.Sphere(), 3)
        self.assertFalse(hasattr(model, "manifold"))
        copied_manifold = getattr(model, "manifold", suite.Sphere())
        self.assertIsInstance(copied_manifold, suite.Sphere)

    def test_cpu_draw_does_not_reseed_cuda(self):
        cfg = suite.config("fire")
        settings = suite.asdict(cfg)
        settings["generator_type"] = "mlp"
        model = suite.build_generator({"generator": settings}, suite.Sphere(), 3)
        with patch("torch.cuda.manual_seed_all") as cuda_seed:
            suite.draw(model, suite.Sphere(), 3, cfg, "mgm_mlp", 16, 42, torch.device("cpu"), 1)
            cuda_seed.assert_not_called()

    def test_stabilization_field_is_periodic(self):
        from stabilization import PeriodicTangentCritic
        model = PeriodicTangentCritic(suite.FlatTorus())
        t, x = torch.rand(32, 1), torch.rand(32, 2) * 2 * torch.pi
        self.assertTrue(torch.allclose(model(t, x), model(t, x + 2 * torch.pi), atol=1e-6, rtol=1e-5))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
