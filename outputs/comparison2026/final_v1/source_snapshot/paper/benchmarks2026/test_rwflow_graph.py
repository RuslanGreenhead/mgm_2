"""Finite-L graph64 checks; CUDA tests require explicit parent-release opt-in.

CPU: unittest discovery. After GPU release only, set RWFLOW_GRAPH_CUDA_TESTS=1
and run this module's Graph64CUDATests. No experimental data are opened.
"""

import copy
import math
import os
import time
import unittest
from unittest.mock import patch

import torch

from .common import ManifoldOps, make_rng
from .rwflow import (
    _GRAPH64_CACHE, _graph64_scaling_body, _sinkhorn_log, clear_graph64_cache,
    chordal_cost, create_method, sinkhorn_conditional, sinkhorn_field,
)


def banks(kind, dim, n=9, m=11, *, concentrated=False, seed=970, device="cpu"):
    ops = ManifoldOps(kind, dim)
    rng = make_rng(seed, device)

    def draw(count):
        if not concentrated:
            return ops.base(count, rng, device)
        noise = torch.randn(count, dim, generator=rng, device=device) * 0.08
        center = torch.ones_like(noise)
        return ops.proj(center + noise)

    return ops, draw(n), draw(m), draw(m)


class Graph64CPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.guard = patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden"))
        self.guard.start()
        self.addCleanup(self.guard.stop)

    def test_finite_L_body_matches_unchanged_scaling64_and_log_reference(self):
        for dtype in (torch.float32, torch.float64):
            cost = 28 * torch.rand(2, 5, 7, generator=make_rng(971), dtype=dtype)
            cost[:, 0] += 100  # First iterate must use the unshifted kernel.
            for iterations in (1, 2, 128, 256, 512, 1024, 2048):
                with self.subTest(dtype=dtype, iterations=iterations):
                    result = sinkhorn_conditional(cost, 0.3, iterations, solver="graph64")
                    eager = sinkhorn_conditional(cost, 0.3, iterations, solver="scaling64")
                    self.assertTrue(torch.equal(result.conditional, eager.conditional))
                    expected = _sinkhorn_log(cost.double(), 0.3, iterations).to(dtype)
                    torch.testing.assert_close(result.conditional, expected, atol=3e-12, rtol=1e-9)
                    self.assertTrue((result.scaling_used == 1).all())

    def test_chordal_global_bounds_fit_fp64_guard(self):
        limit = 0.5 * (-math.log(torch.finfo(torch.float64).tiny) - math.log(256) - 2)
        for kind, dim in (("sphere", 3), ("torus", 2), ("torus", 7)):
            bound = 4 if kind == "sphere" else 4 * dim
            self.assertLess(bound / 0.3, limit)
            ops, x, real, _ = banks(kind, dim)
            self.assertLessEqual(float(chordal_cost(ops, x, real).max()), bound + 1e-5)

    def test_batch_independence_and_owned_results_after_changed_cost(self):
        first = torch.rand(2, 7, 9, generator=make_rng(972), dtype=torch.float64)
        second = first.flip(-1).clone() * 4
        result = sinkhorn_conditional(first, 0.3, 128, solver="graph64")
        saved = result.conditional.clone()
        changed = sinkhorn_conditional(second, 0.3, 128, solver="graph64")
        self.assertFalse(torch.allclose(saved, changed.conditional))
        self.assertTrue(torch.equal(result.conditional, saved))
        self.assertNotEqual(result.conditional.data_ptr(), changed.conditional.data_ptr())
        for i in range(2):
            separate = sinkhorn_conditional(first[i], 0.3, 128, solver="scaling64")
            torch.testing.assert_close(saved[i], separate.conditional, atol=1e-14, rtol=1e-12)

    def test_uniform_and_concentrated_fields_match_for_equal_and_unequal_banks(self):
        for kind, dim in (("sphere", 3), ("torus", 2), ("torus", 7)):
            for concentrated in (False, True):
                ops, x, real, own = banks(kind, dim, concentrated=concentrated)
                for own_bank in (own, own[:-1]):
                    for iterations in (1, 128, 512):
                        actual, _ = sinkhorn_field(ops, x, real, own_bank, epsilon=0.3,
                                                  iterations=iterations, solver="graph64")
                        expected, _ = sinkhorn_field(ops, x, real, own_bank, epsilon=0.3,
                                                    iterations=iterations, solver="scaling64")
                        self.assertTrue(torch.equal(actual, expected))

    def test_guarded_fallback_and_no_rng_or_autograd_or_cuda_cache(self):
        unsafe = torch.tensor([[0., 1000.], [1000., 0.]], requires_grad=True)
        before = torch.random.get_rng_state()
        result = sinkhorn_conditional(unsafe, 1.0, 128, solver="graph64")
        self.assertEqual(result.scaling_used.item(), 0)
        self.assertFalse(result.conditional.requires_grad)
        self.assertTrue(torch.equal(result.conditional, _sinkhorn_log(unsafe.detach().double(), 1., 128).float()))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertFalse(_GRAPH64_CACHE)
        direct = _graph64_scaling_body(torch.zeros(2, 3, 4, requires_grad=True).double(), 0.3, 1)
        self.assertFalse(direct.requires_grad)

    def test_adapter_default_unchanged_and_cpu_graph_continuation(self):
        ops = ManifoldOps("torus", 7)
        cfg = {"width": 8, "depth": 2, "sinkhorn_iters": 128, "epsilon": 0.3}
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(973)
            graph = create_method(ops, {**cfg, "sinkhorn_solver": "graph64"}, "cpu")
            torch.random.default_generator.manual_seed(973)
            eager = create_method(ops, {**cfg, "sinkhorn_solver": "scaling64"}, "cpu")
            self.assertEqual(create_method(ops, {}, "cpu").cfg["sinkhorn_solver"], "log")
        real = ops.base(8, make_rng(974), "cpu")
        graph.update(real, make_rng(975))
        eager.update(real, make_rng(975))
        for a, b in zip(graph.model.parameters(), eager.model.parameters()):
            self.assertTrue(torch.equal(a, b))
        state = copy.deepcopy(graph.checkpoint())
        restored = create_method(ops, {**cfg, "sinkhorn_solver": "graph64"}, "cpu")
        restored.load_checkpoint(state)
        graph.update(real, make_rng(976))
        restored.update(real, make_rng(976))
        for a, b in zip(graph.model.parameters(), restored.model.parameters()):
            self.assertTrue(torch.equal(a, b))


@unittest.skipUnless(os.environ.get("RWFLOW_GRAPH_CUDA_TESTS") == "1",
                     "CUDA reserved; requires explicit parent GPU release")
class Graph64CUDATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        cls.device = torch.device("cuda")

    @classmethod
    def tearDownClass(cls):
        clear_graph64_cache()

    def test_uniform_concentrated_field_matching_and_changing_cost_ownership(self):
        for kind, dim in (("sphere", 3), ("torus", 2), ("torus", 7)):
            for concentrated in (False, True):
                ops, x, real, own = banks(kind, dim, n=32, m=32, device=self.device,
                                          concentrated=concentrated)
                for iterations in (128, 512, 1024):
                    actual, _ = sinkhorn_field(ops, x, real, own, epsilon=0.3,
                                              iterations=iterations, solver="graph64")
                    for solver in ("scaling64", "log"):
                        expected, _ = sinkhorn_field(ops, x, real, own, epsilon=0.3,
                                                    iterations=iterations, solver=solver)
                        self.assertLess(float((actual - expected).abs().max()), 1e-5)
        cost = chordal_cost(ops, x, real).double()
        first = sinkhorn_conditional(cost, 0.3, 128, solver="graph64").conditional
        saved = first.clone()
        changed = cost.flip(-1).clone() + torch.linspace(0, 1, 32, device=self.device)
        second = sinkhorn_conditional(changed, 0.3, 128, solver="graph64").conditional
        expected = sinkhorn_conditional(changed, 0.3, 128, solver="scaling64").conditional
        torch.testing.assert_close(second, expected, atol=1e-12, rtol=1e-9)
        self.assertFalse(torch.allclose(first, second))
        self.assertTrue(torch.equal(first, saved))
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertTrue(_GRAPH64_CACHE)

    def test_warm_replay_speed_probe_at_training_batch_shape(self):
        ops, x, real, own = banks("torus", 7, n=256, m=256, device=self.device)
        cost = torch.stack((chordal_cost(ops, x, real), chordal_cost(ops, x, own)))
        for iterations in (128, 512, 1024):
            times = {}
            for solver in ("scaling64", "graph64"):
                sinkhorn_conditional(cost, 0.3, iterations, solver=solver)
                durations = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    result = sinkhorn_conditional(cost, 0.3, iterations, solver=solver)
                    torch.cuda.synchronize()
                    durations.append(time.perf_counter() - started)
                times[solver] = sorted(durations)[1]
            print({"batch": [2, 256, 256], "epsilon": 0.3, "L": iterations,
                   "median_seconds": times, "speedup": times["scaling64"] / times["graph64"],
                   "max_column_error": float(result.column_error.max())}, flush=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
