"""CPU-only algebra, finite-difference, and adapter-contract tests.

Run: .conda/rmgm/python.exe -B -m unittest paper.benchmarks2026.test_rwflow -v
"""

from __future__ import annotations

import math
import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode

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


def original_log_conditional(cost, epsilon, iterations):
    """Unchanged pre-optimization reference, with no batched paths."""
    n, m = cost.shape
    kernel = -cost / epsilon
    u, v = cost.new_zeros(n), cost.new_zeros(m)
    for _ in range(iterations):
        v = -torch.logsumexp(-math.log(n) + u[:, None] + kernel, dim=0)
        u = -torch.logsumexp(-math.log(m) + v[None, :] + kernel, dim=1)
    return (-math.log(m) + u[:, None] + v[None, :] + kernel).exp()


class ScalarCounter(TorchDispatchMode):
    def __init__(self):
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.count += int(func == torch.ops.aten._local_scalar_dense.default)
        return func(*args, **(kwargs or {}))


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

    def test_batched_and_scaling_match_original_at_every_fixed_count(self):
        for dtype in (torch.float32, torch.float64):
            cost = torch.rand(2, 5, 7, generator=torch.Generator().manual_seed(70), dtype=dtype) * 8
            # Row shifts must not change the reference solver's first iterate.
            cost[:, 0] += 100
            tolerance = dict(atol=3e-6, rtol=2e-5) if dtype == torch.float32 else dict(atol=2e-14, rtol=2e-12)
            for iterations in (1, 2, 32, 64, 256):
                expected = torch.stack([original_log_conditional(c, 1., iterations) for c in cost])
                for solver in ("log", "scaling"):
                    result = sinkhorn_conditional(cost, 1., iterations, solver=solver)
                    torch.testing.assert_close(result.conditional, expected, **tolerance)
                    self.assertTrue((result.scaling_used == int(solver == "scaling")).all())

    def test_scaling_large_representable_range_and_underflow_fallback(self):
        cost = torch.rand(4, 6, generator=torch.Generator().manual_seed(71), dtype=torch.float64) * 250
        result = sinkhorn_conditional(cost, 1., 256, solver="scaling")
        self.assertEqual(result.scaling_used.item(), 1)
        torch.testing.assert_close(result.conditional, original_log_conditional(cost, 1., 256), atol=1e-12, rtol=1e-10)
        # Float32 kernels in this range are rejected rather than clamped into a
        # different OT problem. Even the conservative fallback is transparent.
        result = sinkhorn_conditional(cost.float(), 1., 64, solver="scaling")
        self.assertEqual(result.scaling_used.item(), 0)
        torch.testing.assert_close(result.conditional, original_log_conditional(cost.float(), 1., 64), atol=0., rtol=0.)

    def test_scaling_field_matches_reference_for_both_bank_layouts(self):
        for kind, dim in (("sphere", 3), ("torus", 2), ("torus", 7)):
            ops = ReferenceOps(kind, dim)
            x, real = self.points(ops, 5), self.points(ops, 7, 3)
            for bank_size in (7, 6):
                bank = self.points(ops, bank_size, 4)
                for iterations in (1, 32, 64, 256):
                    expected = (-torch.einsum("ij,ijd->id", original_log_conditional(chordal_cost(ops, x, real), 1., iterations), chordal_cost_gradient(ops, x, real))
                                + torch.einsum("ij,ijd->id", original_log_conditional(chordal_cost(ops, x, bank), 1., iterations), chordal_cost_gradient(ops, x, bank)))
                    field, logs = sinkhorn_field(ops, x, real, bank, epsilon=1., iterations=iterations, solver="scaling")
                    torch.testing.assert_close(field, ops.tangent(x, expected), atol=3e-14, rtol=3e-12)
                    self.assertEqual(logs["target_scaling_used"].item(), 1)

    def test_no_convergence_scalar_reads_inside_either_solver(self):
        cost = torch.rand(2, 4, 5, generator=torch.Generator().manual_seed(6))
        for solver, expected in (("log", 0), ("scaling", 1), ("scaling64", 1)):
            for iterations in (1, 64):
                counter = ScalarCounter()
                with counter:
                    sinkhorn_conditional(cost, 1., iterations, solver=solver, validate_finite=False)
                self.assertEqual(counter.count, expected)

    def test_scaling64_matches_same_cost_float64_reference_then_cast(self):
        for dtype in (torch.float32, torch.float64):
            cost = 28 * torch.rand(2, 5, 7, generator=torch.Generator().manual_seed(81), dtype=dtype)
            for iterations in (1, 2, 32, 64, 256):
                expected = torch.stack([original_log_conditional(c.double(), 0.3, iterations) for c in cost]).to(dtype)
                result = sinkhorn_conditional(cost, 0.3, iterations, solver="scaling64")
                self.assertEqual(result.conditional.dtype, dtype)
                self.assertTrue((result.scaling_used == 1).all())
                tolerance = dict(atol=1e-7, rtol=1e-6) if dtype == torch.float32 else dict(atol=3e-13, rtol=1e-10)
                torch.testing.assert_close(result.conditional, expected, **tolerance)
                if dtype == torch.float32:
                    original32 = torch.stack([original_log_conditional(c, 0.3, iterations) for c in cost])
                    torch.testing.assert_close(result.conditional, original32, atol=6e-6, rtol=3e-5)
                # Residuals reflect the returned weights, not uncast FP64 mass.
                torch.testing.assert_close(result.row_error, (result.conditional.sum(-1) - 1).abs().amax(-1))

    def test_scaling64_expands_safe_range_without_removing_fallback(self):
        cost = torch.tensor([[0., 25., 12.], [24., 0., 15.]], dtype=torch.float32)
        ordinary = sinkhorn_conditional(cost, 0.3, 64, solver="scaling")
        promoted = sinkhorn_conditional(cost, 0.3, 64, solver="scaling64")
        self.assertEqual(ordinary.scaling_used.item(), 0)
        self.assertEqual(promoted.scaling_used.item(), 1)
        unsafe = torch.tensor([[0., 1000.], [1000., 0.]], dtype=torch.float32)
        fallback = sinkhorn_conditional(unsafe, 1., 64, solver="scaling64")
        self.assertEqual(fallback.scaling_used.item(), 0)
        torch.testing.assert_close(fallback.conditional, original_log_conditional(unsafe.double(), 1., 64).float(), atol=0., rtol=0.)

    def test_scaling64_field_preserves_float32_contract(self):
        for kind, dim in (("sphere", 3), ("torus", 2), ("torus", 7)):
            ops = ReferenceOps(kind, dim)
            x, real = self.points(ops, 5).float(), self.points(ops, 7, 23).float()
            for bank_size in (7, 6):
                bank = self.points(ops, bank_size, 24).float()
                target_weights = original_log_conditional(chordal_cost(ops, x, real).double(), 0.3, 64).float()
                self_weights = original_log_conditional(chordal_cost(ops, x, bank).double(), 0.3, 64).float()
                expected = (-torch.einsum("ij,ijd->id", target_weights, chordal_cost_gradient(ops, x, real))
                            + torch.einsum("ij,ijd->id", self_weights, chordal_cost_gradient(ops, x, bank)))
                field, logs = sinkhorn_field(ops, x, real, bank, epsilon=0.3, iterations=64, solver="scaling64")
                self.assertEqual(field.dtype, torch.float32)
                self.assertTrue(all(value.dtype == torch.float32 for value in logs.values()))
                torch.testing.assert_close(field, ops.tangent(x, expected), atol=5e-7, rtol=3e-6)

    def test_log_reference_is_scriptable_and_unchanged(self):
        from .rwflow import _sinkhorn_log

        scripted = torch.jit.script(_sinkhorn_log)
        cost = torch.rand(2, 5, 7, generator=torch.Generator().manual_seed(90), dtype=torch.float64)
        expected = torch.stack([original_log_conditional(c, 1., 32) for c in cost])
        torch.testing.assert_close(scripted(cost, 1., 32), expected, atol=1e-14, rtol=1e-12)


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_fixed_count_config_and_nonfinite_step_do_not_mutate_optimizer(self):
        from .common import ManifoldOps

        ops = ManifoldOps("torus", 2)
        cfg = dict(width=16, depth=2, batch_size=8, sinkhorn_iters=4)
        with self.assertRaises(ValueError):
            create_method(ops, {**cfg, "sinkhorn_tol": 1e-3}, "cpu")
        method = create_method(ops, cfg, "cpu")
        before = method.checkpoint()
        with self.assertRaises(FloatingPointError):
            method.update(torch.full((8, 2), float("nan")), torch.Generator().manual_seed(3))
        self.assertEqual(method.steps, 0)
        self.assertEqual(len(method.optimizer.state), 0)
        for key, value in method.model.state_dict().items():
            torch.testing.assert_close(value, before["model"][key], atol=0., rtol=0.)

    def test_scaling_checkpoint_continuation_and_cpu_adam_counters(self):
        from .common import ManifoldOps

        ops = ManifoldOps("torus", 7)
        real = ops.base(8, torch.Generator().manual_seed(3), "cpu")
        for solver in ("scaling", "scaling64"):
            cfg = dict(width=16, depth=2, batch_size=8, epsilon=0.3, sinkhorn_iters=32, sinkhorn_solver=solver)
            method = create_method(ops, cfg, "cpu")
            logs = method.update(real, torch.Generator().manual_seed(4))
            self.assertTrue(all(value.dtype == torch.float32 for value in logs.values()))
            if solver == "scaling64":
                self.assertEqual(logs["self_scaling_used"].item(), 1)
            restored = create_method(ops, cfg, "cpu")
            restored.load_checkpoint(method.checkpoint())
            self.assertTrue(all(state["step"].device.type == "cpu" for state in restored.optimizer.state.values()))
            method.update(real, torch.Generator().manual_seed(5))
            restored.update(real, torch.Generator().manual_seed(5))
            for key, value in method.model.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[key], atol=0., rtol=0.)

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
