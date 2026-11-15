"""CPU-only RMF analytic, finite-difference, and unchanged-author-source tests.

Run: .conda/rmgm/python.exe -m unittest paper.benchmarks2026.test_meanflow -v
No dataset access, CUDA initialization, compilation, or subprocesses are used.
"""

from __future__ import annotations

import ast
import copy
import math
import random
import unittest
from abc import ABC, abstractmethod
from enum import Enum
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import geoopt
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from paper.benchmarks2026.common import ManifoldOps
from paper.benchmarks2026.meanflow import (
    AUTHOR_COMMIT, NativeTMLP, author_config, create_method,
    geodesic_with_velocity, pcgrad_pair, rmf_loss_terms,
)

AUTHOR = Path(__file__).resolve().parents[2] / "external" / "riemannian-meanflow-author"
SRC = AUTHOR / "rmf" / "rmf"


def rng(seed=123):
    return torch.Generator(device="cpu").manual_seed(seed)


def _execute_definitions(path, names, env):
    """Execute original AST nodes, without importing the broken gfm stack."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"Missing author definitions: {names}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)


def author_definitions():
    env = {"torch": torch, "Tensor": Tensor, "nn": nn, "F": F,
           "geoopt": geoopt, "math": math, "Enum": Enum, "ABC": ABC,
           "abstractmethod": abstractmethod, "signature": signature,
           "copy": copy, "random": random, "_eps": 1e-8,
           "RiemannianGenerativeModule": nn.Module,
           "match_dims": lambda t, shape: t.reshape(t.shape[0], *([1] * (len(shape) - 1)))}
    _execute_definitions(SRC / "models/arch/rfm/diffeq_layers/basic.py",
                         {"ConcatLinear_v2", "TimeDependentSwish"}, env)
    _execute_definitions(SRC / "models/arch/rfm/diffeq_layers/wrappers.py",
                         {"DiffEqWrapper", "diffeq_wrapper"}, env)
    _execute_definitions(SRC / "models/arch/rfm/diffeq_layers/container.py",
                         {"SequentialDiffEq"}, env)
    env["diffeq_layers"] = SimpleNamespace(**{
        key: env[key] for key in ("ConcatLinear_v2", "TimeDependentSwish",
                                  "diffeq_wrapper", "SequentialDiffEq")})
    env["ACTFNS"] = {"swish": env["TimeDependentSwish"]}
    _execute_definitions(SRC / "models/arch/rfm/arch.py",
                         {"TMLP", "PositionalEncoding"}, env)
    _execute_definitions(SRC / "manifold/sphere.py", {"FixedGeooptSphere"}, env)
    _execute_definitions(SRC / "manifold/manifold.py",
                         {"ManifoldType", "Manifold", "Sphere", "FlatTorus"}, env)
    _execute_definitions(SRC / "models/generative_module.py", {"_ManifoldProjected"}, env)
    _execute_definitions(SRC / "models/meanflow_module.py", {"MeanFlowModule"}, env)
    # Disable only the optimization decorator, never alter PCGrad's method body.
    with patch.object(torch, "compile", lambda fn: fn):
        _execute_definitions(SRC / "models/pcgrad.py", {"PCGrad"}, env)
    return env


def copy_parameters(source, destination):
    src, dst = list(source.parameters()), list(destination.parameters())
    assert len(src) == len(dst)
    with torch.no_grad():
        for left, right in zip(src, dst):
            assert left.shape == right.shape
            right.copy_(left)


def native_module(env, ops, model, noise, r, t):
    cls = env["MeanFlowModule"]
    native = cls.__new__(cls)
    nn.Module.__init__(native)
    native.net = model
    native.manifold = env["Sphere" if ops.kind == "sphere" else "FlatTorus"]()
    native.manifold.prior = lambda shape, device: noise.clone().to(device)
    native.sample_r_t = lambda n, device: (r.flatten().clone(), t.flatten().clone())
    native.hparams = SimpleNamespace(prop_equal_times=0.75, in_shape=(ops.dim,))
    native.device = torch.device("cpu")
    return native


class AnalyticTorusField(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(-0.4, dtype=torch.float64))
        self.c = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))

    def forward(self, x, r, t):
        return self.a * x + self.b * r + self.c * t


class ConstantField(nn.Module):
    def __init__(self, ops):
        super().__init__()
        self.ops = ops
        self.value = nn.Parameter(torch.arange(1, ops.dim + 1, dtype=torch.float64) / 7)
        self.calls = 0

    def forward(self, x, r, t):
        self.calls += 1
        return self.ops.tangent(x, self.value.expand_as(x))


class MeanFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.author = author_definitions()

    def assertClose(self, a, b, atol=2e-10, rtol=2e-10):
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)

    def test_author_checkout_commit_is_pinned_without_processes(self):
        head = (AUTHOR / ".git/HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            loose_ref = AUTHOR / ".git" / ref
            if loose_ref.exists():
                head = loose_ref.read_text().strip()
            else:
                packed = (AUTHOR / ".git/packed-refs").read_text().splitlines()
                head = next(line.split()[0] for line in packed if line.endswith(" " + ref))
        self.assertEqual(head, AUTHOR_COMMIT)

    def test_native_architecture_output_jvp_and_reverse_gradient(self):
        for dim, fourier in ((3, 0), (2, 0), (2, 1), (7, 1)):
            with self.subTest(dim=dim, fourier=fourier):
                torch.manual_seed(51)
                native = self.author["TMLP"](dim, d_model=16, num_layers=4,
                                              fourier=fourier or None).double()
                adapter = NativeTMLP(dim, 16, 4, fourier).double()
                copy_parameters(native, adapter)
                x = torch.rand(8, dim, generator=rng(), dtype=torch.float64)
                r = torch.full((8, 1), 0.2, dtype=torch.float64)
                t = torch.full((8, 1), 0.7, dtype=torch.float64)
                direction = (torch.ones_like(x) / 5, torch.zeros_like(r), torch.ones_like(t))
                a, da = torch.func.jvp(native, (x, r, t), direction)
                b, db = torch.func.jvp(adapter, (x, r, t), direction)
                self.assertClose(a, b)
                self.assertClose(da, db)
                ga = torch.autograd.grad(a.square().sum(), list(native.parameters()))
                gb = torch.autograd.grad(b.square().sum(), list(adapter.parameters()))
                for left, right in zip(ga, gb):
                    self.assertClose(left, right)

    def test_native_loss_terms_gradients_and_sampling_equivalence(self):
        for kind, dim, fourier in (("sphere", 3, 0), ("torus", 2, 0),
                                    ("torus", 2, 1), ("torus", 7, 1)):
            with self.subTest(kind=kind, dim=dim, fourier=fourier):
                ops = ManifoldOps(kind, dim)
                method = create_method(ops, {"width": 16, "depth": 3,
                                             "torus_fourier": fourier}, "cpu")
                method.model.double()
                native_net = self.author["TMLP"](dim, d_model=16, num_layers=3,
                                                  fourier=fourier or None).double()
                copy_parameters(method.model.net, native_net)
                manifold = self.author["Sphere" if kind == "sphere" else "FlatTorus"]()
                native_net = self.author["_ManifoldProjected"](native_net, manifold, False)
                real = ops.base(8, rng(10), "cpu", dtype=torch.float64)
                noise = ops.base(8, rng(13), "cpu", dtype=torch.float64)
                t = torch.linspace(0.2, 0.8, 8, dtype=torch.float64)[:, None]
                r = 0.35 * t
                r[:6] = t[:6]
                native = native_module(self.author, ops, native_net, noise, r, t)
                terms = rmf_loss_terms(method.model, ops, real, noise, r, t)
                loss = native.get_loss(real)
                self.assertClose(loss, terms["loss"])
                gradients = torch.autograd.grad(loss, list(native_net.parameters()))
                ours = torch.autograd.grad(terms["loss"], list(method.model.parameters()),
                                           retain_graph=True)
                for left, right in zip(gradients, ours):
                    self.assertClose(left, right)
                n1, n2 = native.get_loss_pc_grad(real)
                for key, value in (("loss1", n1), ("loss2", n2)):
                    self.assertClose(value, terms[key])
                    ga = torch.autograd.grad(value, list(native_net.parameters()), retain_graph=True)
                    gb = torch.autograd.grad(terms[key], list(method.model.parameters()), retain_graph=True)
                    for left, right in zip(ga, gb):
                        self.assertClose(left, right)
                for steps in (1, 4):
                    fixed_noise = ops.base(8, rng(90), "cpu", dtype=torch.float64)
                    native.manifold.prior = lambda shape, device: fixed_noise.clone()
                    old_dtype = torch.get_default_dtype()
                    try:
                        # Native sample_batch's linspace has no dtype argument.
                        torch.set_default_dtype(torch.float64)
                        expected = native.sample_batch(steps, 8)
                    finally:
                        torch.set_default_dtype(old_dtype)
                    self.assertClose(expected, method.sample(8, rng(90), nfe=steps))

    def test_analytic_geodesic_velocity_sphere_and_torus_seam(self):
        ops = ManifoldOps("sphere", 3)
        theta = 1.1
        real = torch.tensor([[1., 0., 0.]], dtype=torch.float64).expand(2, 3)
        noise = torch.tensor([[math.cos(theta), math.sin(theta), 0.]],
                              dtype=torch.float64).expand(2, 3)
        t = torch.tensor([[0.2], [0.7]], dtype=torch.float64)
        x, v = geodesic_with_velocity(ops, real, noise, t)
        expected = torch.cat([(theta*t).cos(), (theta*t).sin(), torch.zeros_like(t)], -1)
        velocity = theta * torch.cat([-(theta*t).sin(), (theta*t).cos(), torch.zeros_like(t)], -1)
        self.assertClose(x, expected)
        self.assertClose(v, velocity)
        self.assertClose((x * v).sum(-1), torch.zeros(2, dtype=torch.float64))
        h = 1e-5
        xp, _ = geodesic_with_velocity(ops, real, noise, t+h)
        xm, _ = geodesic_with_velocity(ops, real, noise, t-h)
        self.assertClose((xp-xm)/(2*h), v, atol=1e-9, rtol=1e-9)
        ops = ManifoldOps("torus", 2)
        real = torch.tensor([[2*math.pi-0.1, 0.1]], dtype=torch.float64)
        noise = torch.tensor([[0.1, 2*math.pi-0.1]], dtype=torch.float64)
        x, v = geodesic_with_velocity(ops, real, noise, t[:1])
        self.assertClose(v, torch.tensor([[0.2, -0.2]], dtype=torch.float64))
        self.assertClose(x, ops.proj(real + 0.2*v))

    def test_analytic_loss_stops_entire_target_and_holds_r_fixed(self):
        ops = ManifoldOps("torus", 2)
        model = AnalyticTorusField()
        real = torch.tensor([[0.2, 0.4], [0.3, 0.5]], dtype=torch.float64, requires_grad=True)
        noise = real.detach() + 0.8
        noise.requires_grad_()
        r = torch.tensor([[0.1], [0.2]], dtype=torch.float64)
        t = torch.tensor([[0.7], [0.8]], dtype=torch.float64)
        terms = rmf_loss_terms(model, ops, real, noise, r, t)
        v = noise.detach() - real.detach()
        xt = real.detach() + t*v
        expected_jvp = model.a.detach()*v + model.c.detach()
        expected_target = v - (t-r)*expected_jvp
        self.assertClose(terms["dudt"], expected_jvp)
        self.assertClose(terms["target"], expected_target)
        self.assertFalse(terms["target"].requires_grad)
        self.assertFalse(terms["velocity"].requires_grad)
        self.assertFalse(terms["xt"].requires_grad)
        terms["loss"].backward()
        residual = (terms["u"] - expected_target).detach()
        self.assertClose(model.a.grad, (2*residual*xt).sum(-1).mean())
        self.assertClose(model.b.grad, (2*residual*r).sum(-1).mean())
        self.assertClose(model.c.grad, (2*residual*t).sum(-1).mean())
        self.assertIsNone(real.grad)
        self.assertIsNone(noise.grad)
        # Parameter finite differences must freeze the target, just as the loss
        # does in reverse AD; recomputing its JVP would test a different gradient.
        with torch.no_grad():
            for p in model.parameters():
                original = p.clone()
                h = 1e-5
                p.copy_(original+h)
                plus = (model(xt, r, t)-expected_target).square().sum(-1).mean()
                p.copy_(original-h)
                minus = (model(xt, r, t)-expected_target).square().sum(-1).mean()
                p.copy_(original)
                self.assertClose((plus-minus)/(2*h), p.grad, atol=2e-9, rtol=2e-9)

    def test_jvp_finite_difference_and_sphere_normal_component_retained(self):
        for kind, dim in (("sphere", 3), ("torus", 7)):
            ops = ManifoldOps(kind, dim)
            method = create_method(ops, {"width": 16}, "cpu")
            method.model.double()
            real = ops.base(8, rng(1), "cpu", dtype=torch.float64)
            noise = ops.base(8, rng(2), "cpu", dtype=torch.float64)
            r = torch.full((8, 1), 0.2, dtype=torch.float64)
            t = torch.full((8, 1), 0.7, dtype=torch.float64)
            terms = rmf_loss_terms(method.model, ops, real, noise, r, t)
            x, v = terms["xt"], terms["velocity"]
            h = 1e-5
            plus = method.model(x+h*v, r, t+h)
            minus = method.model(x-h*v, r, t-h)
            self.assertClose((plus-minus)/(2*h), terms["dudt"], atol=2e-8, rtol=2e-8)
        ops = ManifoldOps("sphere", 3)
        field = ConstantField(ops)
        real = ops.proj(torch.tensor([[1., 0.2, 0.1]], dtype=torch.float64))
        noise = ops.proj(torch.tensor([[0.2, 1., 0.3]], dtype=torch.float64))
        t, r = torch.tensor([[0.7]], dtype=torch.float64), torch.tensor([[0.1]], dtype=torch.float64)
        terms = rmf_loss_terms(field, ops, real, noise, r, t)
        x, v, c = terms["xt"], terms["velocity"], field.value
        derivative = -v*(x*c).sum(-1, keepdim=True)-x*(v*c).sum(-1, keepdim=True)
        self.assertClose(terms["dudt"], derivative)
        self.assertGreater(abs((x*derivative).sum().item()), 1e-3)

    def test_periodic_torus_field_and_jvp(self):
        ops = ManifoldOps("torus", 7)
        method = create_method(ops, {"width": 16}, "cpu")
        method.model.double()
        x = ops.base(8, rng(), "cpu", dtype=torch.float64)
        r = torch.full((8, 1), 0.2, dtype=torch.float64)
        t = torch.full((8, 1), 0.7, dtype=torch.float64)
        shift = 2*math.pi*torch.arange(7, dtype=torch.float64)
        direction = (torch.ones_like(x), torch.zeros_like(r), torch.ones_like(t))
        a, da = torch.func.jvp(method.model, (x, r, t), direction)
        b, db = torch.func.jvp(method.model, (x+shift, r, t), direction)
        self.assertClose(a, b)
        self.assertClose(da, db)

    def test_sampling_sign_nfe_and_training_state(self):
        ops = ManifoldOps("torus", 2)
        method = create_method(ops, {}, "cpu")
        method.model = ConstantField(ops)
        method.model.train()
        for steps in (1, 5):
            method.model.calls = 0
            sample = method.sample(8, rng(22), steps)
            expected = ops.proj(ops.base(8, rng(22), "cpu", dtype=torch.float64)
                                - method.model.value)
            self.assertClose(sample, expected)
            self.assertEqual(method.model.calls, steps)
            self.assertTrue(method.model.training)
        method.model.eval()
        method.sample(8, rng(22))
        self.assertFalse(method.model.training)
        self.assertEqual(method.sample(0, rng()).shape, (0, 2))
        with self.assertRaises(ValueError):
            method.sample(2, rng(), nfe=0)

    def test_time_sampling_native_law_equal_fraction_and_isolated_rng(self):
        method = create_method(ManifoldOps("torus", 2), {}, "cpu")
        global_before = torch.random.get_rng_state().clone()
        r, t = method.sample_times(16, rng(32), torch.float64)
        g = rng(32)
        expected_t = torch.rand(16, generator=g, dtype=torch.float64)
        expected_r = torch.rand(16, generator=g, dtype=torch.float64)*expected_t
        expected_r[:12] = expected_t[:12]
        self.assertClose(t.flatten(), expected_t)
        self.assertClose(r.flatten(), expected_r)
        self.assertEqual((r == t).sum().item(), 12)
        self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))
        method = create_method(ManifoldOps("torus", 2),
                               {"time_distribution": "lognormal", "prop_equal_times": 0.}, "cpu")
        r, t = method.sample_times(16, rng(32), torch.float64)
        g = rng(32)
        a = (-0.4 + torch.randn(16, generator=g, dtype=torch.float64)).exp()
        b = (-0.4 + torch.randn(16, generator=g, dtype=torch.float64)).exp()
        self.assertClose(r.flatten(), torch.minimum(a, b).sigmoid())
        self.assertClose(t.flatten(), torch.maximum(a, b).sigmoid())

    def test_native_pcgrad_default_equivalence_and_zero_gradient(self):
        optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(3, dtype=torch.float64))], lr=0.1)
        native = self.author["PCGrad"](optimizer)
        for g1, g2 in (([1., 2., 3.], [-3., -1., 1.]),
                       ([1., 2., 3.], [3., 2., 1.]),
                       ([0., 0., 0.], [1., 2., 3.])):
            a, b = torch.tensor(g1, dtype=torch.float64), torch.tensor(g2, dtype=torch.float64)
            expected = native._project_conflicting([a.clone(), b.clone()],
                                                    [torch.ones_like(a), torch.ones_like(b)])
            self.assertClose(pcgrad_pair(a, b), expected)
        self.assertClose(pcgrad_pair(torch.zeros(3), torch.zeros(3)), torch.zeros(3))

    def test_update_checkpoint_exact_continuation_both_variants(self):
        for kind, dim in (("sphere", 3), ("torus", 2), ("torus", 7)):
            for variant in ("rmf", "rmf_mt"):
                with self.subTest(kind=kind, dim=dim, variant=variant):
                    ops = ManifoldOps(kind, dim)
                    cfg = {"width": 16, "depth": 3, "variant": variant}
                    method = create_method(ops, cfg, "cpu")
                    real = ops.base(16, rng(6), "cpu")
                    train_rng = rng(4)
                    global_before = torch.random.get_rng_state().clone()
                    initial = [p.detach().clone() for p in method.model.parameters()]
                    logs = method.update(real, train_rng)
                    self.assertTrue(torch.equal(global_before, torch.random.get_rng_state()))
                    for key in ("loss", "loss1", "loss2", "equal_time_fraction"):
                        self.assertFalse(logs[key].requires_grad)
                        self.assertTrue(torch.isfinite(logs[key]))
                    self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial, method.model.parameters())))
                    state = method.checkpoint()
                    other = create_method(ops, cfg, "cpu")
                    other.load_checkpoint(state)
                    cloned_rng = rng()
                    cloned_rng.set_state(train_rng.get_state())
                    a = method.update(real, train_rng)
                    b = other.update(real, cloned_rng)
                    self.assertClose(a["loss"], b["loss"], atol=0, rtol=0)
                    for left, right in zip(method.model.parameters(), other.model.parameters()):
                        self.assertClose(left, right, atol=0, rtol=0)
                    self.assertEqual(method.updates, other.updates)
                    self.assertClose(method.sample(16, rng(9)), other.sample(16, rng(9)), atol=0, rtol=0)
                    self.assertEqual(state["updates"], 1)
                    self.assertTrue(any(not torch.equal(state["model"][key], value)
                                        for key, value in method.model.state_dict().items()))

    def test_nonfinite_input_fails_before_optimizer_step(self):
        method = create_method(ManifoldOps("torus", 2), {"width": 16}, "cpu")
        initial = copy.deepcopy(method.model.state_dict())
        with self.assertRaises(FloatingPointError):
            method.update(torch.full((8, 2), float("nan")), rng())
        self.assertEqual(method.updates, 0)
        for key, value in method.model.state_dict().items():
            self.assertClose(value, initial[key], atol=0, rtol=0)

    def test_author_reference_configs_and_parameter_counts(self):
        self.assertEqual(author_config("volcano")["author_max_epochs"], 2000)
        self.assertEqual(author_config("flood")["author_max_epochs"], 900)
        self.assertEqual(author_config("flood", "paper")["author_max_epochs"], 700)
        self.assertEqual(author_config("rna")["depth"], 6)
        for kind, dim, expected in (("sphere", 3, 51721), ("torus", 2, 51718),
                                     ("torus", 7, 53653)):
            method = create_method(ManifoldOps(kind, dim), {}, "cpu")
            self.assertEqual(method.parameter_count, expected)
            self.assertFalse(method.metadata["common_generator"])
            self.assertFalse(method.metadata["teacher"])


if __name__ == "__main__":
    unittest.main()
