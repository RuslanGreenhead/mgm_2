"""Synthetic rotations are unit-test inputs ONLY, never experiment targets."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .data import make_pose_split, make_split, safe_mesh_path, validate_pose_split, validate_transforms
from .geometry import exp, hat, inner, log, path, prior, rotation_residual
from .models import Field, Generator, flow_objective, mgm_objective
from .runner import check_device, make_rng

torch.set_num_threads(2)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_exp_log_scipy_near_zero_and_pi(self):
        v = torch.tensor([[0., 0., 0.], [1e-10, -2e-10, 3e-10], [.2, -.7, .9],
                          [np.pi-1e-7, 0., 0.], [0., -np.pi+1e-7, 0.]], dtype=torch.float64)
        r = exp(v)
        np.testing.assert_allclose(r.numpy(), Rotation.from_rotvec(v.numpy()).as_matrix(), atol=1e-12)
        torch.testing.assert_close(exp(log(r)), r, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(log(exp(v)), v, atol=1e-10, rtol=1e-10)
        self.assertLess(max(rotation_residual(r).values()), 1e-12)

    def test_reversal_points_and_tangents(self):
        rng = make_rng(9, "cpu")
        a, pa = prior(32, rng, "cpu", torch.float64)
        b, pb = prior(32, rng, "cpu", torch.float64)
        t = torch.rand(32, 1, generator=rng, dtype=torch.float64)
        f, pf, vf = path(a, pa, b, pb, t)
        r, pr, vr = path(b, pb, a, pa, 1-t)
        torch.testing.assert_close(f, r, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(pf, pr)
        torch.testing.assert_close(f @ hat(vf[:, :3]), -(r @ hat(vr[:, :3])), atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(vf[:, 3:], -vr[:, 3:])

    def test_path_velocity_finite_difference_and_generator_gradient(self):
        a, pa = prior(8, make_rng(3, "cpu"), "cpu", torch.float64)
        b, pb = prior(8, make_rng(4, "cpu"), "cpu", torch.float64)
        a.requires_grad_(True)
        time = torch.full((8, 1), .27, dtype=torch.float64)
        r, p, v = path(a, pa, b, pb, time)
        lo, plo, _ = path(a, pa, b, pb, time-1e-5)
        hi, phi, _ = path(a, pa, b, pb, time+1e-5)
        torch.testing.assert_close((hi-lo)/2e-5, r @ hat(v[:, :3]), atol=1e-8, rtol=1e-8)
        torch.testing.assert_close((phi-plo)/2e-5, v[:, 3:])
        gradient = torch.autograd.grad(inner(v, v).mean()+r.mean(), a)[0]
        self.assertTrue(torch.isfinite(gradient).all())

    def test_object_split_no_scale_leak_and_small_dataset_refusal(self):
        split = make_split(["a", "a", "b", "c", "d"], 7)
        self.assertEqual(set.union(*map(set, split.values())), {"a", "b", "c", "d"})
        self.assertTrue(all(split.values()))
        self.assertEqual(sum(len(v) for v in split.values()), 4)
        with self.assertRaises(ValueError):
            make_split(["mug", "table"], 7)
        smoke = make_split(["mug", "table"], 7, True)
        self.assertFalse(smoke["val"])

    def test_bad_transform_and_traversal_rejected(self):
        transforms = np.eye(4)[None]
        validate_transforms(transforms)
        transforms[0, 0, 0] = -1
        with self.assertRaises(ValueError):
            validate_transforms(transforms)
        with self.assertRaises(ValueError):
            safe_mesh_path(".", "../escape.obj")

    def test_within_object_pose_split_seeded_and_disjoint(self):
        source = np.arange(20, 2020, 2)
        split = make_pose_split(source, 2026, "real_object_fixture")
        self.assertEqual({k: len(v) for k, v in split.items()}, {"train": 800, "val": 100, "test": 100})
        self.assertEqual(split, make_pose_split(source, 2026, "real_object_fixture"))
        validate_pose_split(split, source)
        split["test"][0] = split["train"][0]
        with self.assertRaises(ValueError):
            validate_pose_split(split, source)

    def test_duplicate_source_row_within_partition_rejected(self):
        split = make_pose_split(np.arange(100), 2026, "fixture")
        split["train"].append(split["train"][0])
        with self.assertRaises(ValueError):
            validate_pose_split(split, np.arange(100))

    def test_gpu_guard_before_device_allocation(self):
        with self.assertRaises(ValueError):
            check_device("cuda", False)
        check_device("cpu", False)


class GuardAndProtocolTests(unittest.TestCase):
    @staticmethod
    def gpu_query(process_csv):
        def query(argv, **kwargs):
            stdout = "NVIDIA test GPU, 560.94, 1500, 24564, 3\n" if "--query-gpu=" in " ".join(argv) else process_csv
            return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")
        return query

    def test_locked_gpu_never_queries_or_allocates(self):
        from .runner import evaluate, train
        with patch("subprocess.run") as query, patch("torch.cuda.is_available") as cuda:
            for function in (train, evaluate):
                with self.assertRaises(ValueError):
                    function(SimpleNamespace(device="cuda", gpu_release=False))
            query.assert_not_called()
            cuda.assert_not_called()

    def test_gpu_foreign_pid_rejected_before_train_and_evaluate(self):
        from .runner import evaluate, train
        for name in ("C:/other/python.exe", "E:/Codex Projects/Riemmanian Midpoint Generative Models/.conda/rmgm/python.exe", "C:/other/pythonw.exe"):
            with patch("subprocess.run", side_effect=self.gpu_query(f"{os.getpid()+100000}, {name}\n")):
                for function in (train, evaluate):
                    with self.assertRaisesRegex(RuntimeError, "Foreign GPU"):
                        function(SimpleNamespace(device="cuda", gpu_release=True))

    def test_gpu_empty_or_self_pid_allowed(self):
        for stdout in ("", f"{os.getpid()}, C:/env/python.exe\n"):
            with patch("subprocess.run", side_effect=self.gpu_query(stdout)):
                self.assertTrue(check_device("cuda", True)["gpu_inspected"])

    def test_wddm_desktop_unknown_and_quoted_nvidia_csv_are_logged(self):
        stdout = ' 1928, C:/Windows/explorer.exe\n 8844, [Insufficient Permissions]\n 3136, "C:/Program Files/Some, App/Video.UI.exe"\n'
        with patch("subprocess.run", side_effect=self.gpu_query(stdout)):
            snapshot = check_device("cuda", True)
        self.assertEqual([p["classification"] for p in snapshot["processes"]], ["desktop_gui", "unknown_nonpython", "desktop_gui"])
        self.assertEqual(snapshot["unknown_entries"][0]["pid"], 8844)
        self.assertEqual(snapshot["gpus"][0]["memory_used_mib"], "1500")
        self.assertEqual(snapshot["raw_compute_apps_csv"], stdout)

    def test_gpu_unknown_or_failed_query_fails_closed(self):
        for stdout in ("[N/A]\n", "No running processes found\n"):
            with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, stdout=stdout)):
                with self.assertRaisesRegex(RuntimeError, "could not be verified"):
                    check_device("cuda", True)
        for error in (FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 15), subprocess.CalledProcessError(1, "nvidia-smi")):
            with patch("subprocess.run", side_effect=error):
                with self.assertRaises(RuntimeError):
                    check_device("cuda", True)

    def test_cpu_thread_cap(self):
        from .runner import check_cpu_threads
        for value in (0, 3, 4):
            with self.assertRaises(ValueError):
                check_cpu_threads(value)
        check_cpu_threads(2)

    def test_pilot_run_locked_before_plan_read(self):
        from .pilot import run_pilot
        with patch("rmgm.grasp.pilot.load_plan") as load:
            with self.assertRaises(ValueError):
                run_pilot("does-not-exist.json")
            load.assert_not_called()

    def test_detail_frame_counts_are_explicit(self):
        from .visualization import frame_counts, mesh_poses
        vertices = np.array([[-.1, 0, 0], [.1, 0, 0]])
        transformed = mesh_poses(vertices, np.broadcast_to(np.eye(3), (3, 3, 3)), np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]]))
        counts = frame_counts(transformed, [(-1, 1)]*3)
        self.assertEqual(counts, {"total": 3, "fully_inside": 1, "partly_outside": 1, "fully_outside": 1, "not_fully_inside": 2})

    def test_frozen_plan_rejects_budget_or_evaluation_changes(self):
        from . import OUTPUT
        from .data import write_json
        from .pilot import load_plan
        from .pilot_plan import build_plan
        destination = OUTPUT/"verification_ready"
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination) as tmp:
            path = Path(build_plan(Path(tmp)/"plan.json"))
            plan, _ = load_plan(path)
            self.assertEqual(sum(len(j["evaluate_argvs"]) for j in plan["jobs"]), 24)
            original = path.read_text()
            plan["jobs"][0]["train_argv"][plan["jobs"][0]["train_argv"].index("--iterations")+1] = "9999"
            write_json(path, plan)
            with self.assertRaisesRegex(ValueError, "budget"):
                load_plan(path)
            plan = json.loads(original)
            plan["jobs"][-1]["evaluate_argvs"].pop()
            write_json(path, plan)
            with self.assertRaisesRegex(ValueError, "50/100"):
                load_plan(path)

    def test_sequential_pilot_orchestration_and_completed_eval_resume_without_cuda(self):
        from . import OUTPUT
        from .pilot_plan import build_plan
        from .pilot import load_plan, run_pilot
        destination = OUTPUT/"verification_ready"
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination) as tmp:
            plan_path = build_plan(Path(tmp)/"plan.json")
            plan, data = load_plan(plan_path)
            steps, calls = {}, []
            def fake_train(args):
                calls.append((args.method, args.seed, args.iterations))
                steps[str(Path(args.out)/"last.pt")] = args.iterations
            with patch("rmgm.grasp.pilot.check_device"), patch("rmgm.grasp.pilot.load_plan", return_value=(plan, data)), \
                 patch("rmgm.grasp.pilot.train", side_effect=fake_train), \
                 patch("rmgm.grasp.pilot.checkpoint_step", side_effect=lambda p: steps.get(str(p), 0)), \
                 patch("rmgm.grasp.pilot.runtime_estimate"), patch("rmgm.grasp.pilot.write_json"), \
                 patch("rmgm.grasp.pilot.evaluation_complete", return_value=False) as complete, \
                 patch("rmgm.grasp.pilot.evaluate") as evaluate, \
                 patch("rmgm.grasp.pilot.aggregate", return_value="summary.json") as aggregate:
                self.assertEqual(run_pilot(plan_path, True), "summary.json")
                self.assertEqual(len(calls), 12)
                self.assertEqual([c for c in calls if c[2] == 200], [(m, 0, 200) for m in ("mgm", "rmf_mt", "rfm")])
                self.assertEqual(evaluate.call_count, 24)
                self.assertEqual(len(aggregate.call_args.args[1]), 24)
                complete.return_value = True
                evaluate.reset_mock()
                calls.clear()
                run_pilot(plan_path, True)
                self.assertEqual(len(calls), 9)
                self.assertTrue(all(c[2] == 10000 for c in calls))
                evaluate.assert_not_called()
                calls.clear()
                steps.clear()
                aggregate.reset_mock()
                with patch("rmgm.grasp.pilot.shutil.copyfile"):
                    run_pilot(plan_path, True, True)
                self.assertEqual(calls, [(m, 0, n) for m in ("mgm", "rmf_mt", "rfm") for n in (100, 200)])
                evaluate.assert_not_called()
                aggregate.assert_not_called()


class NativeMTTests(unittest.TestCase):
    def test_native_loss_terms_match_independent_body_frame_algebra(self):
        from .source_mt import load_source
        native, _, _ = load_source()
        rng = make_rng(932, "cpu")
        r, _ = prior(8, rng, "cpu", torch.float64)
        u, v, du = [torch.randn(8, 3, generator=rng, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        up, vp, dup = [torch.randn(8, 3, generator=rng, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        t, s = torch.full((8, 1), .8), torch.full((8, 1), .2)
        context = SimpleNamespace(config=SimpleNamespace(training=SimpleNamespace(so3_loss_weight=1., r3_loss_weight=3.)))
        l1, l2, _ = native(context, r, r@hat(u), r@hat(v), r@hat(du), up, vp, dup, t, s)
        expected1 = ((u-v)**2).sum(-1).mean()+3*((up-vp)**2).mean()
        expected2 = (2*(t-s)*(u*du.detach()).sum(-1, keepdim=True)).mean()+6*((t-s)*up*dup.detach()).sum(-1).mean()
        torch.testing.assert_close(l1, expected1)
        torch.testing.assert_close(l2, expected2)
        self.assertIsNone(torch.autograd.grad(l2, du, allow_unused=True, retain_graph=True)[0])
        self.assertIsNone(torch.autograd.grad(l2, dup, allow_unused=True)[0])

    def test_native_pcgrad_conflicting_gradient_projection(self):
        from .source_mt import load_source
        _, pcgrad, _ = load_source()
        p = torch.nn.Parameter(torch.tensor([1., 2.]))
        optimizer = torch.optim.SGD([p], lr=.1)
        pcgrad(optimizer).pc_backward([p[0], -p[0]+p[1]])
        torch.testing.assert_close(p.grad, torch.tensor([.25, .75]))

    def test_native_pcgrad_source_hash_is_pinned(self):
        from .source_mt import load_source
        from .data import sha256
        load_source.cache_clear()
        try:
            with patch("rmgm.grasp.source_mt.sha256", side_effect=lambda p: "changed" if p.name == "pcgrad.py" else sha256(p)):
                with self.assertRaisesRegex(ValueError, "PCGrad"):
                    load_source()
        finally:
            load_source.cache_clear()


class RealDataSmoke(unittest.TestCase):
    def test_actual_matched_mesh_normalization_and_all_source_rows(self):
        from . import OUTPUT
        from .data import PreparedData, verify_record
        data = PreparedData(OUTPUT/"data/public_within_object/manifest.json", "cpu", True)
        self.assertEqual(len(data.source_verification), 2)
        self.assertEqual([{s: len(v) for s, v in r["pose_splits_source_indices"].items()} for r in data.records],
                         [{"train": 1032, "val": 129, "test": 129}, {"train": 1247, "val": 155, "test": 157}])
        for record in data.records:
            self.assertTrue(verify_record(record, 2026)["conditioning_reconstructed_exactly"])
            bad = {**record, "radius_m": record["radius_m"]*1.01}
            with self.assertRaisesRegex(ValueError, "normalization"):
                verify_record(bad, 2026)

    def test_primary_methods_exact_resume_optimizer_rng_and_heldout_exports(self):
        from . import OUTPUT, ROOT
        from .runner import evaluate, train
        from .smoke import assert_state_equal
        destination = OUTPUT/"verification_ready"
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination) as tmp:
            for method in ("mgm", "rmf_mt", "rfm"):
                args = SimpleNamespace(command="train", manifest=str(OUTPUT/"data/public_within_object/manifest.json"),
                    method=method, seed=9, device="cpu", gpu_release=False, cpu_threads=2, allow_smoke_data=True,
                    iterations=3, batch=8, width=16, lr=.0002, translation_weight=1., critic_steps=2, warmup=2,
                    save_every=2, resume=False, out=str(Path(tmp)/method/"full"))
                full = train(args)
                args.out, args.iterations = str(Path(tmp)/method/"resumed"), 2
                train(args)
                args.resume, args.iterations = True, 3
                resumed = train(args)
                a, b = [torch.load(p, map_location="cpu", weights_only=False) for p in (full, resumed)]
                for key in ("step", "model", "critic", "optimizer", "critic_optimizer", "rng", "torch_rng", "python_rng", "numpy_rng"):
                    assert_state_equal(a[key], b[key], f"{method}.{key}")
                args.batch = 9
                with self.assertRaisesRegex(ValueError, "batch"):
                    train(args)
                ev = SimpleNamespace(command="evaluate", checkpoint=resumed, device="cpu", gpu_release=False, cpu_threads=2,
                    allow_smoke_data=True, split="val", samples=20, steps=1, eval_seed=96231,
                    collision_samples=20, skip_collision=False,
                    official_gripper=str(ROOT/"external/acronym-data/public/data/franka_gripper_collision_mesh.stl"), out=None)
                report = json.loads(Path(evaluate(ev)).read_text())
                manifest = json.loads(Path(args.manifest).read_text())
                from .audit import audit_report
                from .data import PreparedData
                self.assertEqual(len(audit_report(report, PreparedData(args.manifest, "cpu", True))), 2)
                report["objects"][0]["distribution"]["mmd2_unbiased_fixed_chordal_product_kernel"] += .1
                with self.assertRaisesRegex(ValueError, "MMD"):
                    audit_report(report, PreparedData(args.manifest, "cpu", True))
                self.assertIsNone(report["physical_success_rate"])
                self.assertEqual(len(report["reference_baseline"]), 2)
                for obj, record in zip(report["objects"], manifest["objects"]):
                    with np.load(obj["samples_path"]) as f:
                        self.assertTrue(set(f["reference_source_indices"]) <= set(record["pose_splits_source_indices"]["val"]))
                        self.assertFalse(set(f["reference_source_indices"]) & set(record["pose_splits_source_indices"]["train"]))
                        validate_transforms(f["transforms_m"])
                        validate_transforms(f["target_transforms_m"])
                        from .evaluation import static_geometry
                        expected = static_geometry(record, f["target_rotations"][:20], f["target_transforms_m"][:20, :3, 3], ev.official_gripper)
                        self.assertEqual(obj["heldout_reference_static_geometry"], expected)
                        self.assertEqual(obj["heldout_reference_proxy_source_indices"], f["reference_source_indices"][:20].tolist())
                    self.assertIn("official_franka_mesh_collision_fraction", obj["static_geometry"])

    def test_reference_proxy_and_historical_labels_remain_separate(self):
        from . import OUTPUT
        from .data import PreparedData
        from .evaluation import static_geometry
        data = PreparedData(OUTPUT/"data/public_within_object/manifest.json", "cpu", True)
        passing = []
        for record, target in zip(data.records, data.objects):
            ids = target["pose_indices"]["val"][:20]
            reference = static_geometry(record, target["r"][ids].numpy(),
                target["p"][ids].numpy()*record["radius_m"]+np.asarray(record["center_m"]))
            self.assertIsNone(reference["physical_success_rate"])
            passing.extend(reference["per_pose_proxy_pass"])
        self.assertEqual(len(passing), 40)
        # A collision overrides historical labels; never assume a fixed proxy rate.
        with patch("rmgm.grasp.evaluation.CollisionManager") as manager:
            manager.return_value.in_collision_single.return_value = True
            manager.return_value.min_distance_single.return_value = 0.
            counterexample = static_geometry(record, target["r"][ids].numpy(),
                target["p"][ids].numpy()*record["radius_m"]+np.asarray(record["center_m"]))
        self.assertEqual(counterexample["author_static_feasible_pose_fraction"], 0.)
        self.assertIsNone(counterexample["physical_success_rate"])

    def test_source_mt_finite_gradients_and_unchanged_loss_terms(self):
        from . import OUTPUT
        from .data import PreparedData
        from .source_mt import load_source, objective_parts
        manifest = OUTPUT / "data/public_smoke/manifest.json"
        if not manifest.exists():
            self.skipTest("Prepare the official public data first")
        data = PreparedData(manifest, "cpu", True)
        field = Field(32)
        l1, l2 = objective_parts(field, data, make_rng(94, "cpu"), 8)
        native, pcgrad, provenance = load_source()
        optimizer = torch.optim.Adam(field.parameters())
        pcgrad(optimizer).pc_backward([l1, l2])
        self.assertTrue(torch.isfinite(l1+l2))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in field.parameters()))
        self.assertEqual(provenance["diagonal_fraction"], 0.)
        self.assertIn("adaptation", provenance["classification"])

    def test_real_within_object_training_draws_never_use_val_or_test(self):
        from . import OUTPUT
        from .data import PreparedData
        manifest = OUTPUT / "data/public_within_object/manifest.json"
        if not manifest.exists():
            self.skipTest("Prepare the within-object public pilot first")
        data = PreparedData(manifest, "cpu", True)
        ids, r, p = data.sample(512, make_rng(93, "cpu"))
        for index in ids.unique().tolist():
            obj = data.objects[index]
            train = obj["pose_indices"]["train"]
            heldout = torch.cat((obj["pose_indices"]["val"], obj["pose_indices"]["test"]))
            self.assertFalse(set(train.tolist()) & set(heldout.tolist()))
            for rr, pp in zip(r[ids == index], p[ids == index]):
                candidates = ((obj["r"][train]-rr).abs().amax((-1, -2)) == 0) & ((obj["p"][train]-pp).abs().amax(-1) == 0)
                self.assertTrue(candidates.any())

    def test_static_proxy_matches_released_author_functions(self):
        import ast
        from typing import List, Tuple, Union
        import trimesh
        from trimesh.collision import CollisionManager
        from . import OUTPUT, ROOT
        from .data import PreparedData
        from .evaluation import static_geometry
        manifest = OUTPUT / "data/public_smoke/manifest.json"
        source = ROOT / "external/riemannian-meanflow-author/grasp/src/core/visualize.py"
        if not manifest.exists() or not source.exists():
            self.skipTest("Public data or author clone unavailable")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {"create_parallel_gripper_mesh", "create_grasp_volume", "check_collision"}
        module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        # Omit only contact visualization: it does not participate in the rule.
        namespace = dict(torch=torch, np=np, trimesh=trimesh, CollisionManager=CollisionManager,
                         List=List, Tuple=Tuple, Union=Union, enforce_trimesh=lambda m: m,
                         find_contact_points=lambda *a, **kw: np.empty((0, 3)), random_blue=lambda: [0, 0, 255, 255])
        exec(compile(module, str(source), "exec"), namespace)
        data = PreparedData(manifest, "cpu", True)
        for obj, record in zip(data.objects, data.records):
            r = obj["r"][:8]
            p = obj["p"][:8]*record["radius_m"]+torch.tensor(record["center_m"])
            original_collision, _, _, original_overlap = namespace["check_collision"](r, p, record["mesh_path"], record["mesh_scale"])
            local = static_geometry(record, r.numpy(), p.numpy())
            self.assertEqual(local["per_pose_collision"], original_collision)
            self.assertEqual(local["per_pose_volume_overlap"], original_overlap)
            self.assertIsNone(local["physical_success_rate"])

    def test_all_methods_finite_gradients_on_real_public_h5(self):
        from . import OUTPUT
        from .data import PreparedData
        manifest = OUTPUT / "data/public_smoke/manifest.json"
        if not manifest.exists():
            self.skipTest("Prepare the official public data first")
        data = PreparedData(manifest, "cpu", True)
        torch.set_num_threads(2)
        for method in ("rfm", "rmf_ccadapt", "glsd"):
            with self.subTest(method=method):
                field = Field(32)
                loss = flow_objective(method, field, data, make_rng(87, "cpu"), 8)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in field.parameters()))
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in field.parameters() if p.grad is not None), 0)
        generator, critic = Generator(32), Field(32)
        loss = mgm_objective(generator, critic, data, make_rng(88, "cpu"), 8, True)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in generator.parameters()))
        critic.zero_grad(set_to_none=True)
        critic.requires_grad_(False)
        loss = mgm_objective(generator, critic, data, make_rng(89, "cpu"), 8, False)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None for p in critic.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in generator.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in generator.parameters() if p.grad is not None), 0)
        r, _ = generator(torch.randn(32, 16), data, torch.zeros(32, dtype=torch.long))
        self.assertLess(max(rotation_residual(r).values()), 5e-5)
