"""CPU-only KGDM numerical and adapter-contract tests.

Run: .conda/rmgm/python.exe -m unittest paper.benchmarks2026.test_kgdm -v
"""

import itertools
import math
import unittest

import torch

from .common import ManifoldOps, PointGenerator
from .kgdm import DEFAULTS, author_core, create_method, drift_target, intrinsic_field


def fixture(kind, n=5, m=7, dtype=torch.float64, dim=None):
    ops = ManifoldOps(kind, dim or (3 if kind == "sphere" else 2))
    rng = torch.Generator(device="cpu").manual_seed(1925)
    x = ops.base(n, rng, "cpu", dtype=dtype)
    y = ops.base(m, rng, "cpu", dtype=dtype)
    return ops, x, y


def kernel(ops, x, y, temperature, family, scale=1.0):
    # Clamp only singular coincidences, never the separated pairs under test.
    d2 = ops.pairwise_distance2(x, y).clamp_min(1e-24)
    exponent = d2 / scale**2 if family == "gauss" else d2.sqrt() / scale
    return (-exponent / temperature).exp()


def pairwise_gradient(ops, x, y, temperature, family):
    logs = ops.log(x[:, None], y[None])
    kval = kernel(ops, x, y, temperature, family)
    if family == "gauss":
        return (2 / temperature) * kval[..., None] * logs
    distance = ops.pairwise_distance2(x, y).sqrt()
    return kval[..., None] * logs / (temperature * distance[..., None])


def frozen_weights(ops, x, y, temperature, family, dist_norm):
    if ops.kind == "sphere":
        dc = author_core()
        dp, dn = dc.pairwise_geodesic(x, y), dc.pairwise_geodesic(x, x)
    else:
        dp = ops.pairwise_distance2(x, y).sqrt()
        dn = ops.pairwise_distance2(x, x).sqrt()
    scale = torch.cat((dp, dn), 1).mean().clamp_min(1e-6) if dist_norm else x.new_tensor(1.0)
    p, n = dp / scale, dn / scale
    if family == "gauss":
        p, n = p.square(), n.square()
    n = n + torch.eye(len(x), dtype=x.dtype) * 1e6
    logits = torch.cat((-p / temperature, -n / temperature), 1)
    a = (logits.softmax(1) * logits.softmax(0)).sqrt()
    ap, an = a.split((len(y), len(x)), 1)
    return (ap * an.sum(1, keepdim=True)).detach(), (an * ap.sum(1, keepdim=True)).detach(), scale.detach()


def autodiff_frozen_field(ops, x, y, temps, family, dist_norm, drop_T):
    """Differentiate pair kernels, NOT affinity normalizers or batch scale.

    W and K(x0,y) are fixed coefficients. The released field deliberately drops
    the normalized-distance chain factor; undo it here before differentiating.
    """
    delta = torch.zeros_like(x, requires_grad=True)
    query = ops.exp(x, delta)
    potential = delta.sum() * 0
    for temperature in temps:
        wp, wn, scale = frozen_weights(ops, x, y, temperature, family, dist_norm)
        kp0 = kernel(ops, x, y, temperature, family, scale).detach()
        kn0 = kernel(ops, x, x, temperature, family, scale).detach()
        kp = kernel(ops, query, y, temperature, family, scale)
        kn = kernel(ops, query, x, temperature, family, scale)
        factor = scale**2 if family == "gauss" else scale
        if drop_T:
            factor = factor * temperature / (2 if family == "gauss" else 1)
        potential = potential + factor * ((wp * kp / kp0).sum() - (wn * kn / kn0).sum())
    return ops.tangent(x, torch.autograd.grad(potential, delta)[0])


class IntrinsicGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_pair_gradients_against_autodiff(self):
        for kind, family in itertools.product(("sphere", "torus"), ("laplace", "gauss")):
            with self.subTest(kind=kind, kernel=family):
                ops, x, y = fixture(kind, n=3, m=4)
                analytic = pairwise_gradient(ops, x, y, 0.7, family)
                for j in range(len(y)):
                    delta = torch.zeros_like(x, requires_grad=True)
                    values = kernel(ops, ops.exp(x, delta), y[j:j + 1], 0.7, family)
                    gradient = torch.autograd.grad(values.sum(), delta)[0]
                    torch.testing.assert_close(gradient, analytic[:, j], rtol=1e-10, atol=1e-11)

    def test_pair_gradients_against_finite_difference(self):
        for kind, family in itertools.product(("sphere", "torus"), ("laplace", "gauss")):
            with self.subTest(kind=kind, kernel=family):
                ops, x, y = fixture(kind, n=3, m=4)
                direction = ops.tangent(x, torch.tensor([[0.2] * ops.dim], dtype=x.dtype).expand_as(x))
                analytic = pairwise_gradient(ops, x, y, 0.7, family)
                eps = 1e-6
                numeric = (kernel(ops, ops.exp(x, eps * direction), y, 0.7, family)
                           - kernel(ops, ops.exp(x, -eps * direction), y, 0.7, family)) / (2 * eps)
                expected = (analytic * direction[:, None]).sum(-1)
                torch.testing.assert_close(numeric, expected, rtol=3e-6, atol=1e-10)

    def test_full_field_against_frozen_kernel_autodiff(self):
        for kind, family, dist_norm, drop_t in itertools.product(
            ("sphere", "torus"), ("laplace", "gauss"), (False, True), (False, True),
        ):
            with self.subTest(kind=kind, kernel=family, dist_norm=dist_norm, drop_T=drop_t):
                ops, x, y = fixture(kind)
                actual = intrinsic_field(ops, x, y, temps=(0.4, 0.9), kernel=family,
                                         dist_norm=dist_norm, drop_T=drop_t)
                reference = autodiff_frozen_field(ops, x, y, (0.4, 0.9), family, dist_norm, drop_t)
                torch.testing.assert_close(actual, reference, rtol=2e-9, atol=1e-10)

    def test_sphere_executes_exact_author_field(self):
        ops, x, y = fixture("sphere")
        for family, dist_norm, drop_t in itertools.product(("laplace", "gauss"), (False, True), (False, True)):
            with self.subTest(kernel=family, dist_norm=dist_norm, drop_T=drop_t):
                actual = intrinsic_field(ops, x, y, temps=(0.2, 0.4), kernel=family,
                                         dist_norm=dist_norm, drop_T=drop_t)
                reference = author_core().compute_V_sphere_multi(
                    x, y, x, (0.2, 0.4), dist_norm=dist_norm, grad_form=True,
                    kernel=family, drop_T=drop_t,
                )
                self.assertTrue(torch.equal(actual, reference))

    def test_rna7_full_field_against_frozen_autodiff(self):
        ops, x, y = fixture("torus", dim=7)
        for family, dist_norm, drop_t in itertools.product(("laplace", "gauss"), (False, True), (False, True)):
            with self.subTest(kernel=family, dist_norm=dist_norm, drop_T=drop_t):
                actual = intrinsic_field(ops, x, y, temps=(0.4, 0.9), kernel=family,
                                         dist_norm=dist_norm, drop_T=drop_t)
                reference = autodiff_frozen_field(ops, x, y, (0.4, 0.9), family, dist_norm, drop_t)
                torch.testing.assert_close(actual, reference, rtol=2e-9, atol=1e-10)

    def test_temperature_ladder_is_sum_not_mean(self):
        for kind in ("sphere", "torus"):
            ops, x, y = fixture(kind)
            actual = intrinsic_field(ops, x, y, temps=(0.2, 0.4))
            expected = sum(intrinsic_field(ops, x, y, temps=(t,)) for t in (0.2, 0.4))
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)

    def test_permutation_equivariance(self):
        for kind in ("sphere", "torus"):
            ops, x, y = fixture(kind)
            perm = torch.tensor([3, 1, 4, 0, 2])
            field = intrinsic_field(ops, x, y)
            reordered = intrinsic_field(ops, x[perm], y.flip(0))
            torch.testing.assert_close(reordered, field[perm], rtol=1e-10, atol=1e-12)

    def test_field_target_and_all_normalizers_are_detached(self):
        for kind in ("sphere", "torus"):
            ops, x, y = fixture(kind)
            x.requires_grad_()
            y.requires_grad_()
            target, field, step = drift_target(ops, x, y)
            for tensor in (target, field, step):
                self.assertFalse(tensor.requires_grad)
                self.assertIsNone(tensor.grad_fn)

    def test_cap_target_tangency_and_constraints(self):
        for kind in ("sphere", "torus"):
            ops, x, y = fixture(kind)
            target, field, step = drift_target(ops, x, y, step_size=100, max_step=0.3)
            self.assertLessEqual(float(step.norm(dim=-1).max()), 0.3 + 1e-12)
            torch.testing.assert_close(target, ops.exp(x, step))
            torch.testing.assert_close(ops.distance2(x, target), step.square().sum(-1), rtol=1e-10, atol=1e-12)
            if kind == "sphere":
                torch.testing.assert_close((x * field).sum(-1), torch.zeros(len(x), dtype=x.dtype), atol=1e-12, rtol=0)
                torch.testing.assert_close(target.norm(dim=-1), torch.ones(len(x), dtype=x.dtype))
            else:
                self.assertTrue(((target >= 0) & (target < 2 * math.pi)).all())

    def test_drop_temperature_keeps_author_constants(self):
        for kind, family in itertools.product(("sphere", "torus"), ("laplace", "gauss")):
            ops, x, y = fixture(kind)
            dropped = intrinsic_field(ops, x, y, temps=(0.4,), kernel=family, drop_T=True)
            kept = intrinsic_field(ops, x, y, temps=(0.4,), kernel=family, drop_T=False)
            torch.testing.assert_close(kept, dropped * ((2 if family == "gauss" else 1) / 0.4))

    def test_single_positive_attraction_has_correct_sign(self):
        for kind, family in itertools.product(("sphere", "torus"), ("laplace", "gauss")):
            ops, x, _ = fixture(kind)
            x = x[:1].expand(3, -1).clone()
            move = ops.tangent(x[:1], torch.full_like(x[:1], 0.2))
            y = ops.exp(x[:1], move)
            field = intrinsic_field(ops, x, y, temps=(0.4,), kernel=family)
            self.assertTrue((ops.inner(x, field, move.expand_as(x)) > 0).all())

    def test_identical_points_remain_finite_and_zero(self):
        for kind, family, dtype in itertools.product(
            ("sphere", "torus"), ("laplace", "gauss"), (torch.float32, torch.float64),
        ):
            ops, x, _ = fixture(kind, dtype=dtype)
            x = x[:1].expand(4, -1).clone()
            target, field, _ = drift_target(ops, x, x, kernel=family)
            self.assertTrue(torch.isfinite(target).all())
            torch.testing.assert_close(field, torch.zeros_like(field), atol=1e-6, rtol=0)

    def test_self_diagonal_is_removed_from_affinity(self):
        for kind in ("sphere", "torus"):
            ops, x, y = fixture(kind)
            wp, wn, _ = frozen_weights(ops, x, y, 0.4, "laplace", True)
            self.assertTrue((wn.diagonal() == 0).all())
            torch.testing.assert_close(wp.sum(1), wn.sum(1), atol=1e-15, rtol=1e-12)

    def test_bad_field_inputs_fail(self):
        ops, x, y = fixture("sphere")
        for options in ({"temps": ()}, {"temps": (0,)}, {"temps": (float("nan"),)}, {"kernel": "mmd"}):
            with self.assertRaises(ValueError):
                intrinsic_field(ops, x, y, **options)
        with self.assertRaises(ValueError):
            intrinsic_field(ops, x[:1], y)
        with self.assertRaises(ValueError):
            intrinsic_field(ops, x, y[:0])
        for options in ({"step_size": 0}, {"max_step": -1}):
            with self.assertRaises(ValueError):
                drift_target(ops, x, y, **options)


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def method(self, ops, **cfg):
        return create_method(ops, {"width": 16, "depth": 2, **cfg}, "cpu")

    def test_common_generator_exact_parameterization(self):
        for kind in ("sphere", "torus"):
            ops, _, _ = fixture(kind)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(17)
                method = self.method(ops)
                torch.random.default_generator.manual_seed(17)
                common = PointGenerator(ops, latent_dim=16, width=16, depth=2)
            self.assertIsInstance(method.model, PointGenerator)
            for key, value in common.state_dict().items():
                self.assertTrue(torch.equal(method.model.state_dict()[key], value))

    def test_update_is_finite_changes_weights_and_not_real_batch(self):
        for kind, family in itertools.product(("sphere", "torus"), ("laplace", "gauss")):
            ops, _, y = fixture(kind, m=12, dtype=torch.float32)
            y.requires_grad_()
            method = self.method(ops, kernel=family)
            before = {key: value.clone() for key, value in method.model.state_dict().items()}
            stats = method.update(y, torch.Generator().manual_seed(48))
            self.assertTrue(all(isinstance(value, torch.Tensor) and not value.requires_grad
                                and value.device.type == "cpu" for value in stats.values()))
            self.assertTrue(all(torch.isfinite(torch.as_tensor(value)) for value in stats.values()))
            self.assertEqual(method.steps, 1)
            self.assertIsNone(y.grad)
            self.assertTrue(any(not torch.equal(value, before[key]) for key, value in method.model.state_dict().items()))

    def test_sample_one_forward_and_explicit_rng(self):
        for kind in ("sphere", "torus"):
            ops, _, _ = fixture(kind)
            method = self.method(ops)
            count = [0]
            def record(*_):
                count[0] += 1
            hook = method.ema_model.register_forward_hook(record)
            global_rng = torch.random.get_rng_state().clone()
            a = method.sample(11, torch.Generator().manual_seed(81))
            hook.remove()
            self.assertEqual(count[0], 1)
            b = method.sample(11, torch.Generator().manual_seed(81))
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))
            self.assertEqual(a.shape, (11, ops.dim))
            self.assertFalse(a.requires_grad)
            torch.testing.assert_close(a, ops.proj(a))
            self.assertEqual(method.sample(0, torch.Generator().manual_seed(0)).shape, (0, ops.dim))
            for nfe in (0, 2):
                with self.assertRaises(ValueError):
                    method.sample(2, torch.Generator().manual_seed(0), nfe=nfe)

    def test_update_does_not_use_global_rng(self):
        ops, _, y = fixture("sphere", dtype=torch.float32)
        method = self.method(ops)
        state = torch.random.get_rng_state().clone()
        rng = torch.Generator().manual_seed(12)
        rng_before = rng.get_state().clone()
        method.update(y, rng)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertFalse(torch.equal(rng_before, rng.get_state()))

    def test_checkpoint_is_frozen_and_resumes_optimizer_ema_rng(self):
        for kind in ("sphere", "torus"):
            ops, _, y = fixture(kind, dtype=torch.float32)
            method = self.method(ops)
            rng = torch.Generator().manual_seed(48)
            method.update(y, rng)
            state = method.checkpoint()
            rng_state = rng.get_state().clone()
            frozen_model = {key: value.clone() for key, value in state["model"].items()}
            frozen_ema = {key: value.clone() for key, value in state["ema_model"].items()}
            stats = method.update(y, rng)
            restored = self.method(ops)
            restored.load_checkpoint(state)
            restored_rng = torch.Generator().set_state(rng_state)
            restored_stats = restored.update(y, restored_rng)
            self.assertEqual(stats["step"], restored_stats["step"])
            torch.testing.assert_close(stats["loss"], restored_stats["loss"], atol=0, rtol=0)
            for a, b in zip(method.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            for key in frozen_model:
                self.assertTrue(torch.equal(frozen_model[key], state["model"][key]))
                self.assertTrue(torch.equal(frozen_ema[key], state["ema_model"][key]))
            a = method.sample(10, torch.Generator().manual_seed(5))
            b = restored.sample(10, torch.Generator().manual_seed(5))
            self.assertTrue(torch.equal(a, b))

    def test_optional_author_warmup(self):
        ops, _, y = fixture("sphere", dtype=torch.float32)
        method = self.method(ops, warmup_steps=2)
        rng = torch.Generator().manual_seed(9)
        rates = [method.update(y, rng)["lr"] for _ in range(3)]
        torch.testing.assert_close(torch.stack(rates), torch.tensor([0, DEFAULTS["lr"] / 2, DEFAULTS["lr"]]))

    def test_rna7_resume_and_sampling_contract(self):
        ops, _, y = fixture("torus", dim=7, dtype=torch.float32)
        method = self.method(ops)
        rng = torch.Generator().manual_seed(82)
        method.update(y, rng)
        state, rng_state = method.checkpoint(), rng.get_state().clone()
        expected = method.update(y, rng)
        restored = self.method(ops)
        restored.load_checkpoint(state)
        actual = restored.update(y, torch.Generator().set_state(rng_state))
        torch.testing.assert_close(actual["loss"], expected["loss"], atol=0, rtol=0)
        a = method.sample(16, torch.Generator().manual_seed(39))
        b = restored.sample(16, torch.Generator().manual_seed(39))
        self.assertEqual(a.shape, (16, 7))
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(((a >= 0) & (a < 2 * math.pi)).all())

    def test_configs_and_invalid_batches(self):
        ops, _, y = fixture("sphere", dtype=torch.float32)
        self.assertEqual(self.method(ops, kernel="gaussian").cfg["kernel"], "gauss")
        for cfg in ({"lr": 0}, {"bandwidth_scale": 0}, {"ema_decay": 1},
                    {"warmup_steps": -1}, {"max_step": 0}, {"grad_clip": -1}):
            with self.assertRaises(ValueError):
                self.method(ops, **cfg)
        method = self.method(ops)
        for real in (y[:1], y[:0], y[:, :2], torch.full_like(y, float("nan"))):
            with self.assertRaises(ValueError):
                method.update(real, torch.Generator().manual_seed(1))


if __name__ == "__main__":
    unittest.main()
