"""CPU-only algebra, finite-difference, and adapter-contract tests.

Run: .conda/rmgm/python.exe -B -m unittest paper.benchmarks2026.test_rwflow -v
"""

from __future__ import annotations

import math
import unittest

import torch

from .rwflow import (
    chordal_cost, chordal_cost_gradient, create_method, geometric_targets,
    sinkhorn_conditional, sinkhorn_field,
)


class ReferenceOps:
    """Independent native-coordinate geometry for mathematical unit tests."""

    def __init__(self, kind, dim):
        self.kind, self.dim, self.coord_dim = kind, dim, dim

    def embed(self, x):
        return x if self.kind == "sphere" else torch.cat((x.cos(), x.sin()), -1)

    def proj(self, x):
        return (x / x.norm(dim=-1, keepdim=True)) if self.kind == "sphere" else x.remainder(2 * math.pi)

    def tangent(self, x, v):
        return v - (x * v).sum(-1, keepdim=True) * x if self.kind == "sphere" else v

    def inner(self, x, u, v):
        return (u * v).sum(-1)

    def exp(self, x, v):
        if self.kind == "torus":
            return (x + v).remainder(2 * math.pi)
        r = v.norm(dim=-1, keepdim=True)
        return self.proj(r.cos() * x + torch.sinc(r / math.pi) * v)

    def distance2(self, x, y):
        if self.kind == "torus":
            d = y - x
            return torch.atan2(d.sin(), d.cos()).square().sum(-1)
        x, y = self.proj(x), self.proj(y)
        angle = 2 * torch.atan2((x - y).norm(dim=-1), (x + y).norm(dim=-1))
        return angle.square()

    def base(self, n, rng, device, dtype=torch.float32):
        if self.kind == "sphere":
            return self.proj(torch.randn(n, self.dim, generator=rng, device=device, dtype=dtype))
        return 2 * math.pi * torch.rand(n, self.dim, generator=rng, device=device, dtype=dtype)


def ot_value(ops, x, y, epsilon):
    cost = chordal_cost(ops, x, y)
    conditional = sinkhorn_conditional(cost, epsilon, 1200).conditional
    pi = conditional / len(x)
    log_density = (conditional * len(y)).log()
    return (pi * (cost + epsilon * log_density)).sum()


class MathematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def points(self, ops, n, seed=1):
        return ops.base(n, torch.Generator().manual_seed(seed), "cpu", torch.float64)

    def test_cost_gradient_directional_finite_difference(self):
        for kind, dim in (("sphere", 3), ("torus", 2), ("torus", 7)):
            with self.subTest(kind=kind, dim=dim):
                ops = ReferenceOps(kind, dim)
                x, y = self.points(ops, 4), self.points(ops, 5, 2)
                direction = ops.tangent(x, torch.randn(x.shape, generator=torch.Generator().manual_seed(3), dtype=x.dtype))
                h = 1e-6
                numeric = (chordal_cost(ops, ops.exp(x, h * direction), y)
                           - chordal_cost(ops, ops.exp(x, -h * direction), y)) / (2 * h)
                analytic = (chordal_cost_gradient(ops, x, y) * direction[:, None]).sum(-1)
                torch.testing.assert_close(numeric, analytic, atol=3e-9, rtol=3e-7)

    def test_periodic_embedding_and_gradient(self):
        ops = ReferenceOps("torus", 2)
        x = torch.tensor([[0.01, 2 * math.pi - 0.02]], dtype=torch.float64)
        y = torch.tensor([[2 * math.pi - 0.01, 0.02]], dtype=torch.float64)
        expected = (2 - 2 * torch.cos(x - y)).sum(-1)
        torch.testing.assert_close(chordal_cost(ops, x, y)[0], expected)
        torch.testing.assert_close(chordal_cost(ops, x + 2 * math.pi, y), chordal_cost(ops, x, y))
        torch.testing.assert_close(chordal_cost_gradient(ops, x, y)[0, 0], 2 * (x - y).sin()[0])

    def test_sinkhorn_analytic_diagonal_and_entropy_units(self):
        cost = torch.tensor([[0., 2.], [2., 0.]], dtype=torch.float64, requires_grad=True)
        result = sinkhorn_conditional(cost, epsilon=0.4, iterations=20)
        diagonal = 1 / (1 + math.exp(-5))
        expected = torch.tensor([[diagonal, 1 - diagonal], [1 - diagonal, diagonal]], dtype=cost.dtype)
        torch.testing.assert_close(result.conditional, expected)
        self.assertFalse(result.conditional.requires_grad)
        self.assertGreater(result.conditional[0, 0].item(), 0.99)

    def test_sinkhorn_rectangular_uniform_marginals(self):
        cost = torch.tensor([[0., 1., 3.], [2., 0., 1.]], dtype=torch.float64)
        result = sinkhorn_conditional(cost, 0.3, 600)
        torch.testing.assert_close(result.conditional.sum(1), torch.ones(2, dtype=cost.dtype))
        torch.testing.assert_close(result.conditional.mean(0), torch.full((3,), 1 / 3, dtype=cost.dtype))
        self.assertLess(result.column_error.item(), 1e-10)

    def test_singleton_field_has_both_terms_and_correct_sign(self):
        for kind, dim in (("sphere", 3), ("torus", 2)):
            ops = ReferenceOps(kind, dim)
            x, y, bank = self.points(ops, 1), self.points(ops, 1, 5), self.points(ops, 1, 7)
            field, _ = sinkhorn_field(ops, x, y, bank)
            expected = -chordal_cost_gradient(ops, x, y)[0] + chordal_cost_gradient(ops, x, bank)[0]
            torch.testing.assert_close(field, expected)
            scaled, _ = sinkhorn_field(ops, x, y, bank, epsilon=0.2, cost_scale=2.)
            torch.testing.assert_close(scaled, 2 * field)
            if kind == "sphere":
                self.assertLess((field * x).sum(-1).abs().max().item(), 1e-12)

    def test_equal_real_and_self_bank_cancel_exactly(self):
        for kind, dim in (("sphere", 3), ("torus", 7)):
            ops = ReferenceOps(kind, dim)
            x, bank = self.points(ops, 5), self.points(ops, 7, 8)
            field, _ = sinkhorn_field(ops, x, bank, bank)
            torch.testing.assert_close(field, torch.zeros_like(x), atol=0., rtol=0.)

    def test_field_is_negative_full_sinkhorn_divergence_gradient(self):
        # Moving x in BOTH arguments of OT(x,x) checks the 1/2 self factor.
        for kind, dim in (("sphere", 3), ("torus", 2)):
            with self.subTest(kind=kind):
                ops = ReferenceOps(kind, dim)
                x, y = self.points(ops, 3), self.points(ops, 4, 20)
                direction = ops.tangent(x, torch.randn(x.shape, generator=torch.Generator().manual_seed(11), dtype=x.dtype))
                epsilon, h = 0.7, 2e-5
                field, _ = sinkhorn_field(ops, x, y, x, epsilon=epsilon, iterations=1200)

                def energy(source):
                    return ot_value(ops, source, y, epsilon) - 0.5 * ot_value(ops, source, source, epsilon)

                numeric = (energy(ops.exp(x, h * direction)) - energy(ops.exp(x, -h * direction))) / (2 * h)
                analytic = -(field * direction).sum() / len(x)
                torch.testing.assert_close(numeric, analytic, atol=2e-8, rtol=2e-6)

    def test_independent_bank_field_finite_difference(self):
        for kind, dim in (("sphere", 3), ("torus", 7)):
            ops = ReferenceOps(kind, dim)
            x, real, bank = self.points(ops, 3), self.points(ops, 4, 12), self.points(ops, 5, 33)
            direction = ops.tangent(x, torch.ones_like(x))
            epsilon, h = 0.9, 2e-5
            field, _ = sinkhorn_field(ops, x, real, bank, epsilon=epsilon, iterations=1200)

            def energy(source):
                return ot_value(ops, source, real, epsilon) - ot_value(ops, source, bank, epsilon)

            numeric = (energy(ops.exp(x, h * direction)) - energy(ops.exp(x, -h * direction))) / (2 * h)
            torch.testing.assert_close(numeric, -(field * direction).sum() / len(x), atol=2e-8, rtol=2e-6)

    def test_scale_equivalence(self):
        ops = ReferenceOps("torus", 7)
        x, y, bank = self.points(ops, 5), self.points(ops, 6, 3), self.points(ops, 4, 8)
        v, _ = sinkhorn_field(ops, x, y, bank, epsilon=0.4)
        v2, _ = sinkhorn_field(ops, x, y, bank, epsilon=0.8, cost_scale=2.)
        torch.testing.assert_close(v2, 2 * v)
        torch.testing.assert_close(geometric_targets(ops, x, v, 0.1), geometric_targets(ops, x, v2, 0.05))

    def test_targets_and_field_are_fully_detached(self):
        for kind, dim in (("sphere", 3), ("torus", 2)):
            ops = ReferenceOps(kind, dim)
            x = self.points(ops, 4).requires_grad_()
            real = self.points(ops, 5, 7).requires_grad_()
            bank = self.points(ops, 4, 8).requires_grad_()
            field, _ = sinkhorn_field(ops, x, real, bank)
            target = geometric_targets(ops, x, field, 0.1)
            self.assertFalse(field.requires_grad)
            self.assertFalse(target.requires_grad)
            loss = ops.distance2(x, target).mean()
            loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertIsNone(real.grad)
            self.assertIsNone(bank.grad)

    def test_invalid_solver_inputs(self):
        cost = torch.zeros(2, 2)
        for epsilon in (0., -1., float("nan")):
            with self.assertRaises(ValueError):
                sinkhorn_conditional(cost, epsilon, 10)
        for iterations in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                sinkhorn_conditional(cost, 0.1, iterations)


class AdapterTests(unittest.TestCase):
    def test_shared_generator_update_checkpoint_sample_and_rng(self):
        from .common import ManifoldOps

        for kind, dim in (("sphere", 3), ("torus", 2), ("torus", 7)):
            with self.subTest(kind=kind, dim=dim):
                ops = ManifoldOps(kind, dim)
                cfg = dict(width=16, depth=2, latent_dim=16, batch_size=8, sinkhorn_iters=25)
                method = create_method(ops, cfg, "cpu")
                before = method.checkpoint()
                real = ops.base(8, torch.Generator().manual_seed(17), "cpu")
                global_rng = torch.random.get_rng_state().clone()
                recorded = []
                hook = method.model.register_forward_hook(
                    lambda module, args, output: recorded.append((args[0].detach().clone(), output.requires_grad))
                )
                logs = method.update(real, torch.Generator().manual_seed(22))
                hook.remove()
                torch.testing.assert_close(torch.random.get_rng_state(), global_rng)
                self.assertEqual(len(recorded), 2)
                self.assertTrue(recorded[0][1])
                self.assertFalse(recorded[1][1])
                self.assertFalse(torch.equal(recorded[0][0], recorded[1][0]))
                self.assertTrue(all(torch.is_tensor(value) and not value.requires_grad
                                    and value.device.type == "cpu" for value in logs.values()))
                self.assertTrue(all(math.isfinite(float(value)) for value in logs.values()))
                self.assertTrue(any(not torch.equal(before["model"][key], value)
                                    for key, value in method.model.state_dict().items()))
                state = method.checkpoint()
                restored = create_method(ops, cfg, "cpu")
                restored.load_checkpoint(state)
                a = method.sample(11, torch.Generator().manual_seed(9))
                b = restored.sample(11, torch.Generator().manual_seed(9))
                torch.testing.assert_close(a, b, atol=0., rtol=0.)
                self.assertEqual(tuple(a.shape), (11, dim))
                self.assertTrue(torch.isfinite(a).all())
                if kind == "sphere":
                    torch.testing.assert_close(a.norm(dim=-1), torch.ones(11), atol=2e-6, rtol=2e-6)
                else:
                    self.assertTrue(((a >= 0) & (a < 2 * math.pi)).all())
                with self.assertRaises(ValueError):
                    method.sample(2, torch.Generator(), nfe=2)
                # A returned checkpoint must remain immutable after another update.
                snapshot = {key: value.clone() for key, value in state["model"].items()}
                rng_a, rng_b = torch.Generator().manual_seed(47), torch.Generator().manual_seed(47)
                method.update(real, rng_a)
                restored.update(real, rng_b)
                for key, value in method.model.state_dict().items():
                    torch.testing.assert_close(value, restored.model.state_dict()[key], atol=0., rtol=0.)
                    torch.testing.assert_close(state["model"][key], snapshot[key], atol=0., rtol=0.)


if __name__ == "__main__":
    unittest.main()
