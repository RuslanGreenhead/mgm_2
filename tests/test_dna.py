"""Geometry, data-alignment, metric, and update checks for promoter R-MGM."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rmgm.dna.data import PromoterArrays
from rmgm.dna.evaluation import kmer_counts, kmer_pearson
from rmgm.dna.geometry import fisher_rao_distance, from_logits, path, sphere_exp, sphere_log, target_sphere
from rmgm.dna.method import PromoterRMGM


class GeometryTests(unittest.TestCase):
    def test_geodesic_endpoints_velocity_and_fisher_scale(self):
        torch.manual_seed(3)
        x = from_logits(torch.randn(2, 7, 4, requires_grad=True))
        y = target_sphere(torch.randint(0, 4, (2, 7)))
        delta = sphere_log(x, y)
        torch.testing.assert_close(sphere_exp(x, delta), y, atol=2e-5, rtol=2e-5)
        start, _ = path(x, y, torch.zeros(2, 1, 1))
        end, _ = path(x, y, torch.ones(2, 1, 1))
        torch.testing.assert_close(start, x, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(end, y, atol=2e-5, rtol=2e-5)
        t = torch.full((2, 1, 1), 0.37)
        point, velocity = path(x, y, t)
        plus, _ = path(x, y, t + 1e-4)
        minus, _ = path(x, y, t - 1e-4)
        torch.testing.assert_close((plus - minus) / 2e-4, velocity, atol=1e-3, rtol=1e-3)
        self.assertLess(float((point * velocity).sum(-1).abs().max()), 1e-5)
        self.assertTrue(torch.all(point >= -1e-6))
        expected = 2 * torch.linalg.vector_norm(torch.linalg.vector_norm(delta, dim=-1), dim=-1)
        torch.testing.assert_close(fisher_rao_distance(x, y), expected, atol=2e-5, rtol=2e-5)

    def test_boundary_and_generator_gradient(self):
        labels = torch.tensor([[0, 1, 2, 3, 4]])
        y = target_sphere(labels)
        self.assertTrue(torch.equal(y[0, 4], torch.full((4,), 0.5)))
        logits = torch.randn(1, 5, 4, requires_grad=True)
        x = from_logits(logits)
        midpoint, velocity = path(x, y, torch.full((1, 1, 1), 0.5))
        (midpoint.square().sum() + velocity.square().sum()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


class MetricTests(unittest.TestCase):
    def test_pooled_kmers_and_unknowns(self):
        labels = np.array([[0, 1, 2, 3, 0, 1, 4], [0, 1, 2, 3, 0, 1, 2]], dtype=np.uint8)
        counts = kmer_counts(labels)
        self.assertEqual(int(counts.sum()), 3)
        self.assertAlmostEqual(kmer_pearson(counts, counts), 1.0)


class MethodTests(unittest.TestCase):
    def test_conditional_update_and_checkpoint(self):
        config = {"width": 8, "critic_width": 8, "blocks": 1, "critic_warmup": 1,
                  "critic_steps": 1, "ema_decay": 0.9}
        method = PromoterRMGM(config, torch.device("cpu"))
        rng = torch.Generator().manual_seed(4)
        labels = torch.randint(0, 4, (2, 32))
        signal = torch.rand(2, 32, 1)
        self.assertFalse(method.update(labels, signal, rng)["generator_updated"])
        self.assertTrue(method.update(labels, signal, rng)["generator_updated"])
        self.assertEqual((method.critic_updates, method.generator_updates), (2, 1))
        sample = method.sample(signal, rng)
        self.assertEqual(sample.shape, (2, 32, 4))
        torch.testing.assert_close(sample.square().sum(-1), torch.ones(2, 32), atol=1e-5, rtol=1e-5)
        restored = PromoterRMGM(config, torch.device("cpu"))
        restored.load_state_dict(method.state_dict())
        self.assertEqual(restored.iteration, 2)


class PreparedDataTests(unittest.TestCase):
    def test_joint_shift(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            length = 1044
            np.save(path / "train_labels.npy", np.tile(np.arange(length, dtype=np.uint8) % 4, (2, 1)))
            np.save(path / "train_signal.npy", np.tile(np.arange(length, dtype=np.float32), (2, 1)))
            (path / "manifest.json").write_text(json.dumps({"flank": 10}))
            dataset = PromoterArrays(path, "train", shift=10, seed=7)
            labels, signal = dataset[0]
            self.assertEqual(labels.shape, (1024,))
            self.assertTrue(torch.equal(labels, signal[:, 0].long() % 4))


if __name__ == "__main__":
    unittest.main()
