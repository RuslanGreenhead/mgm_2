"""Synthetic CPU-only worker protocol regressions; never opens real datasets."""

import ast
import copy
import io
import pickle
import random
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack, redirect_stdout
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import audit, data_metrics, make_mgm_refinement_plan, mgm, select_configs, worker
from .common import ManifoldOps, make_rng


class _CPUOnlyTorch:
    """Run the unchanged worker orchestration without any CUDA API calls."""

    def __init__(self):
        self.cuda = SimpleNamespace(
            manual_seed_all=lambda seed: None,
            get_rng_state=lambda: torch.zeros(1, dtype=torch.uint8),
            set_rng_state=lambda state: None,
            current_device=lambda: 0,
            get_device_name=lambda: "synthetic CPU worker fixture",
        )
        self.random = SimpleNamespace(
            get_rng_state=torch.random.get_rng_state,
            set_rng_state=torch.random.set_rng_state,
            fork_rng=lambda **kwargs: torch.random.fork_rng(devices=[]),
        )

    def device(self, name):
        if name != "cuda":
            raise AssertionError(f"Unexpected worker device request: {name}")
        return torch.device("cpu")

    def manual_seed(self, seed):
        return torch.random.default_generator.manual_seed(seed)

    def __getattr__(self, name):
        return getattr(torch, name)


class _SyntheticArchive:
    def __init__(self, forbid_test=False):
        self.accesses = []
        self.forbid_test = forbid_test
        rng = np.random.default_rng(991)
        self.arrays = {
            "train": rng.uniform(0, 2 * np.pi, (24, 2)).astype(np.float32),
            "val_target": rng.uniform(0, 2 * np.pi, (8, 2)).astype(np.float32),
            "test_target": np.zeros((8, 2), dtype=np.float32),
        }

    def __getitem__(self, name):
        self.accesses.append(name)
        if name == "test_target" and self.forbid_test:
            raise AssertionError("Calibration accessed forbidden test_target")
        return self.arrays[name].copy()


class WorkerProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.root = Path(self.resources.enter_context(tempfile.TemporaryDirectory()))
        self.resources.enter_context(torch.random.fork_rng(devices=[]))
        py_state, np_state = random.getstate(), np.random.get_state()
        self.addCleanup(random.setstate, py_state)
        self.addCleanup(np.random.set_state, np_state)
        self.resources.enter_context(patch.object(torch.cuda, "_lazy_init",
                                                side_effect=AssertionError("CUDA forbidden")))
        self.job = {
            "data": "synthetic", "method": "mgm", "seed": 0,
            "steps": 1, "eval_every": 1, "validation_draws": 1,
            "score_test": False, "output": "synthetic_job",
            "config": {"width": 8, "depth": 2, "batch_size": 8,
                       "critic_warmup": 0, "critic_steps": 1},
        }
        self.folder = self.root / self.job["output"]
        self.archive = _SyntheticArchive()
        self.manifest = {
            "kind": "torus", "dim": 2, "arrays_sha256": "fixture",
            "reference": {"bandwidth2": 1.0, "coverage_radius": 0.2},
        }

    def invoke(self, fail_write=None, fail_save=None, fail_save_step=None,
               fail_write_prefix=None, metric_score=None):
        self.restored_iterations = []
        self.last_bytes_before_failed_save = None

        def read(path):
            if Path(path).name == "job_input.json":
                return copy.deepcopy(self.job)
            if Path(path).name == "manifest.json":
                return copy.deepcopy(self.manifest)
            raise AssertionError(f"Unexpected input read: {path}")

        def write(path, value):
            if ((fail_write and Path(path).name == fail_write)
                    or (fail_write_prefix and Path(path).name.startswith(fail_write_prefix))):
                raise OSError("Synthetic interrupted completion")
            data_metrics.write(path, value)

        real_save = torch.save

        def save(value, path):
            if (fail_save and Path(path).name == fail_save
                    and (fail_save_step is None or value["step"] == fail_save_step)):
                last = Path(path).with_name("last.pt")
                if last.exists():
                    self.last_bytes_before_failed_save = last.read_bytes()
                # Emulate interruption after opening/truncating the checkpoint.
                with Path(path).open("wb") as stream:
                    stream.write(b"partial checkpoint")
                raise OSError("Synthetic interrupted checkpoint write")
            real_save(value, path)

        real_restore = mgm.MGM.load_checkpoint

        def restore(method, state):
            self.restored_iterations.append(state["iteration"])
            real_restore(method, state)

        def metrics(ops, x, y, reference):
            row = data_metrics.metrics(ops, x, y, reference)
            if metric_score is not None:
                row["kernel_mmd2_v"] = metric_score
            return row

        numpy_proxy = SimpleNamespace(
            load=lambda path: self.archive, mean=np.mean, isfinite=np.isfinite,
            savez_compressed=np.savez_compressed,
            random=SimpleNamespace(seed=np.random.seed),
        )
        with ExitStack() as stack:
            stack.enter_context(patch.object(worker, "torch", _CPUOnlyTorch()))
            stack.enter_context(patch.object(worker, "np", numpy_proxy))
            stack.enter_context(patch.object(worker, "read", read))
            stack.enter_context(patch.object(worker, "write", write))
            stack.enter_context(patch.object(worker, "sha", lambda path: "fixture"))
            stack.enter_context(patch.object(worker, "metrics", metrics))
            stack.enter_context(patch.object(worker, "synchronize", time.perf_counter))
            stack.enter_context(patch.object(torch, "save", save))
            stack.enter_context(patch.object(mgm.MGM, "load_checkpoint", autospec=True,
                                            side_effect=restore))
            stack.enter_context(patch.object(sys, "argv", [
                "worker.py", "--job", str(self.root / "job_input.json"),
                "--root", str(self.root),
            ]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            worker.main()

    def test_calibration_never_materializes_test_target(self):
        self.archive = _SyntheticArchive(forbid_test=True)
        self.invoke()
        self.assertNotIn("test_target", self.archive.accesses)

    def test_resume_rejects_changed_training_seed(self):
        with self.assertRaisesRegex(OSError, "interrupted completion"):
            self.invoke(fail_write="completed.json")
        original = torch.load(self.folder / "last.pt", weights_only=False)
        self.assertEqual(original["job"]["seed"], 0)
        self.job["seed"] = 1
        with self.assertRaisesRegex((ValueError, RuntimeError), "(?i)(job|seed|identity|resume)"):
            self.invoke()

    def test_resume_recovers_complete_snapshot_after_torn_last_write(self):
        with self.assertRaisesRegex(OSError, "interrupted completion"):
            self.invoke(fail_write="completed.json")
        interrupted = torch.load(self.folder / "interrupted.pt", weights_only=False)
        self.assertEqual(interrupted["step"], self.job["steps"])
        self.assertEqual(interrupted["stage"], "training")
        # Old direct-write checkpoints can already be truncated on disk.
        with (self.folder / "last.pt").open("wb") as stream:
            stream.write(b"partial checkpoint")
        with self.assertRaises(pickle.UnpicklingError):
            torch.load(self.folder / "last.pt", weights_only=False)
        self.invoke()
        self.assertTrue((self.folder / "completed.json").is_file())
        self.assertEqual(self.restored_iterations, [interrupted["method_state"]["iteration"]])

    def test_failed_temporary_write_preserves_last_and_recovers_newer_interrupted(self):
        self.job["steps"] = 2
        with self.assertRaisesRegex(OSError, "interrupted checkpoint write"):
            self.invoke(fail_save="last.tmp.pt", fail_save_step=2)
        last = self.folder / "last.pt"
        self.assertIsNotNone(self.last_bytes_before_failed_save)
        self.assertEqual(last.read_bytes(), self.last_bytes_before_failed_save)
        self.assertEqual(torch.load(last, weights_only=False)["step"], 1)
        self.assertEqual((self.folder / "last.tmp.pt").read_bytes(), b"partial checkpoint")
        interrupted = torch.load(self.folder / "interrupted.pt", weights_only=False)
        self.assertEqual((interrupted["stage"], interrupted["step"]), ("training", 2))
        self.invoke()
        self.assertEqual(self.restored_iterations, [2])
        done = data_metrics.read(self.folder / "completed.json")
        self.assertEqual((done["steps"], done["attempts"]), (2, 2))

    def test_resume_chooses_highest_training_step_from_each_candidate(self):
        self.job["steps"] = 2
        with self.assertRaisesRegex(OSError, "interrupted checkpoint write"):
            self.invoke(fail_save="last.tmp.pt", fail_save_step=2)
        lower = torch.load(self.folder / "last.pt", weights_only=False)
        higher = torch.load(self.folder / "interrupted.pt", weights_only=False)
        self.assertEqual((lower["step"], higher["step"]), (1, 2))
        for newest in ("last.pt", "interrupted.pt", "best.pt"):
            with self.subTest(newest=newest):
                for name in ("last.pt", "interrupted.pt", "best.pt"):
                    torch.save(higher if name == newest else lower, self.folder / name)
                with self.assertRaisesRegex(OSError, "interrupted completion"):
                    self.invoke(fail_write="completed.json")
                self.assertEqual(self.restored_iterations, [2])

    def test_validation_and_test_stage_interrupted_snapshots_are_ignored(self):
        with self.assertRaisesRegex(OSError, "interrupted completion"):
            self.invoke(fail_write="completed.json")
        valid = torch.load(self.folder / "last.pt", weights_only=False)
        for stage in ("validation", "test"):
            with self.subTest(stage=stage):
                invalid = copy.deepcopy(valid)
                invalid["stage"] = stage
                invalid["step"] = 100
                invalid["method_state"]["iteration"] = 999
                torch.save(invalid, self.folder / "interrupted.pt")
                with self.assertRaisesRegex(OSError, "interrupted completion"):
                    self.invoke(fail_write="completed.json")
                self.assertEqual(self.restored_iterations, [1])

    def test_failed_validation_does_not_publish_training_interruption_state(self):
        self.archive = _SyntheticArchive(forbid_test=True)
        with self.assertRaisesRegex(OSError, "interrupted completion"):
            self.invoke(fail_write="val_step1_metrics.json")
        self.assertFalse((self.folder / "interrupted.pt").exists())
        initial = torch.load(self.folder / "best.pt", weights_only=False)
        self.assertEqual((initial["stage"], initial["step"]), ("training", 0))
        self.invoke()
        self.assertEqual(self.restored_iterations, [0])
        self.assertNotIn("test_target", self.archive.accesses)
        self.assertEqual(data_metrics.read(self.folder / "completed.json")["steps"], 1)

    def test_failed_test_restarts_from_training_not_selected_sampling_weights(self):
        self.job.update(steps=2, score_test=True)
        with self.assertRaisesRegex(OSError, "interrupted completion"):
            self.invoke(fail_write_prefix="test_step", metric_score=0.5)
        self.assertFalse((self.folder / "interrupted.pt").exists())
        last = self.folder / "last.pt"
        final_bytes = last.read_bytes()
        final = torch.load(last, weights_only=False)
        selected = torch.load(self.folder / "best.pt", weights_only=False)
        self.assertEqual((final["stage"], final["step"]), ("training", 2))
        self.assertEqual((selected["stage"], selected["step"]), ("training", 0))
        self.assertEqual(self.archive.accesses.count("test_target"), 1)
        self.invoke(metric_score=0.5)
        self.assertEqual(self.restored_iterations, [2, 0])
        self.assertEqual(last.read_bytes(), final_bytes)
        done = data_metrics.read(self.folder / "completed.json")
        self.assertEqual((done["steps"], done["attempts"], done["selected_step"]), (2, 2, 0))
        self.assertEqual(done["test_evaluations"], 3)

    def test_plain_resume_preserves_model_and_both_explicit_rngs(self):
        with self.assertRaisesRegex(OSError, "interrupted completion"):
            self.invoke(fail_write="completed.json")
        before = torch.load(self.folder / "last.pt", weights_only=False)
        self.invoke()
        after = torch.load(self.folder / "last.pt", weights_only=False)
        for key in ("target", "method", "cpu", "cuda"):
            self.assertTrue(torch.equal(before["rng"][key], after["rng"][key]))
        for key, value in before["method_state"]["model"].items():
            self.assertTrue(torch.equal(value, after["method_state"]["model"][key]))
        done = data_metrics.read(self.folder / "completed.json")
        self.assertEqual(done["selected_step"], before["best_step"])
        self.assertEqual(done["attempts"], before["attempts"])


class SelectionAndMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_selection_excludes_completed_jobs_outside_frozen_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            groups = {"sphere": "fire", "torus2": "general", "torus7": "rna"}
            data_metrics.write(out / "solver_probe.json", {
                "group_solvers": {group: "log" for group in groups},
                "iteration_choices": {group: {"0.3": 512} for group in groups},
            })
            for phase_name in ("mgm_calibration_v1", "calibration_v2", "mgm_refinement_v1"):
                phase = out / phase_name
                jobs = []
                methods = ("mgm",) if phase_name.startswith("mgm") else (
                    "rwflow", "kgdm", "meanflow")
                for data in groups.values():
                    for method in methods:
                        folder = phase / f"{data}_{method}"
                        job = {"data": data, "method": method, "seed": 0,
                               "score_test": False, "output": str(folder),
                               "config": {"marker": "planned"}}
                        if method == "rwflow":
                            job["config"]["epsilon"] = 0.3
                        jobs.append(job)
                        data_metrics.write(folder / "job.json", job)
                        data_metrics.write(folder / "completed.json", {
                            "data": data, "method": method, "test_evaluations": 0,
                            "selected_step": 0, "selected_validation_mmd2": 0.5,
                        })
                data_metrics.write(phase / "protocol.json", {"plan": {"jobs": jobs}})
                data_metrics.write(phase / "completed.json", {"all_completed": True})
            stale = out / "mgm_calibration_v1" / "unplanned_stale_run"
            data_metrics.write(stale / "job.json", {
                "data": "fire", "method": "mgm", "seed": 0, "score_test": False,
                "config": {"marker": "unplanned"},
            })
            data_metrics.write(stale / "completed.json", {
                "data": "fire", "method": "mgm", "test_evaluations": 0,
                "selected_step": 0, "selected_validation_mmd2": 0.0,
            })
            with patch.object(select_configs, "OUT", out), redirect_stdout(io.StringIO()):
                select_configs.main()
            frozen = data_metrics.read(out / "selected_config_v1.json")
            self.assertEqual(frozen["groups"]["sphere"]["mgm"]["marker"], "planned")
            planned_job = out / "mgm_calibration_v1" / "fire_mgm" / "job.json"
            metadata = data_metrics.read(planned_job)
            metadata["seed"] = 99
            data_metrics.write(planned_job, metadata)
            with patch.object(select_configs, "OUT", out), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(AssertionError, "Job differs from frozen plan"):
                    select_configs.main()

    def test_atomic_json_failure_preserves_existing_or_absent_completion_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            for existing in (False, True):
                with self.subTest(existing=existing):
                    path = Path(directory) / str(existing) / "completed.json"
                    if existing:
                        data_metrics.write(path, {"all_completed": False})
                        before = path.read_bytes()
                    with patch.object(Path, "replace", side_effect=OSError("replace interrupted")):
                        with self.assertRaisesRegex(OSError, "replace interrupted"):
                            data_metrics.write(path, {"all_completed": True})
                    if existing:
                        self.assertEqual(path.read_bytes(), before)
                        self.assertIs(data_metrics.read(path)["all_completed"], False)
                    else:
                        self.assertFalse(path.exists())
                    self.assertIs(data_metrics.read(path.with_name("completed.json.tmp"))["all_completed"], True)

    def test_refinement_has_six_validation_only_jobs_and_unchanged_generator(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            jobs = []
            for data in ("fire", "general", "rna"):
                folder = out / "mgm_calibration_v1" / data
                config = {"width": 128, "depth": 3, "batch_size": 256,
                          "latent_dim": 16, "lr": 2e-4, "critic_warmup": 500,
                          "critic_steps": 5, "lr_c": 2e-4,
                          "noise": 0.0 if data == "fire" else 0.7}
                done = {"data": data, "method": "mgm", "selected_validation_mmd2": 0.2}
                jobs.append(done)
                data_metrics.write(folder / "job.json", {"config": config})
                data_metrics.write(folder / "completed.json", done)
            data_metrics.write(out / "mgm_calibration_v1" / "completed.json", {
                "all_completed": True, "jobs": jobs,
            })
            with patch.object(make_mgm_refinement_plan, "OUT", out):
                make_mgm_refinement_plan.main()
            plan = data_metrics.read(out / "plans" / "mgm_refinement_v1.json")
            self.assertEqual(len(plan["jobs"]), 6)
            for data in ("fire", "general", "rna"):
                rows = [job for job in plan["jobs"] if job["data"] == data]
                self.assertEqual(len(rows), 2)
                self.assertEqual({(r["config"]["critic_steps"], r["config"]["lr_c"])
                                  for r in rows}, {(5, 8e-4), (10, 4e-4)})
                for job in rows:
                    self.assertIs(job["score_test"], False)
                    self.assertEqual((job["steps"], job["seed"]), (4000, 0))
                    self.assertEqual(job["config"]["width"], 128)
                    self.assertEqual(job["config"]["critic_width"], 256)
                    self.assertEqual(job["config"]["noise"], 0.0 if data == "fire" else 0.7)

    def test_metric_mmd_matches_independent_numpy_on_synthetic_manifolds(self):
        for kind, dim in (("sphere", 3), ("torus", 2), ("torus", 7)):
            with self.subTest(kind=kind, dim=dim):
                ops = ManifoldOps(kind, dim)
                rng = make_rng(997)
                x = ops.base(12, rng, "cpu", dtype=torch.float64)
                y = ops.base(13, rng, "cpu", dtype=torch.float64)
                reference = {"bandwidth2": 1.5, "coverage_radius": 0.3}
                row = data_metrics.metrics(ops, x, y, reference)
                independent = audit.independent_mmd(x.numpy(), y.numpy(), kind, 1.5)
                self.assertAlmostEqual(row["kernel_mmd2_v"], independent, places=12)

    def test_draw_mean_precedes_sample_sd_across_training_seeds(self):
        # Execute just the pure aggregation functions, never plot or read outputs.
        path = Path(__file__).with_name("plots.py")
        parsed = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        nodes = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"summarize_seed", "seed_group", "aggregate_seeds"}]
        self.assertGreaterEqual(len(nodes), 2)
        env = {"np": np, "defaultdict": defaultdict,
               "METRICS": ("kernel_mmd2_v",), "EXPECTED_SEEDS": (0, 1, 2)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
        seed_rows = []
        metadata = {"phase": "synthetic", "config_id": "fixed", "variant": None,
                    "config": "{}", "resolved_config": "{}"}
        for seed in range(3):
            run = SimpleNamespace(key=("synthetic", "mgm", seed), selected_step=1,
                                  done={"steps": 1}, folder=Path("synthetic"),
                                  config_key="fixed", metadata=lambda: metadata)
            draws = [{"kernel_mmd2_v": 2 * seed + draw} for draw in range(3)]
            seed_rows.append(env["summarize_seed"](run, draws, "test"))
        summary = env["aggregate_seeds"](seed_rows)[0]
        self.assertEqual(summary["kernel_mmd2_v_mean"], 3.0)
        sd_key = ("kernel_mmd2_v_training_seed_sd" if "kernel_mmd2_v_training_seed_sd" in summary
                  else "kernel_mmd2_v_sd")
        self.assertEqual(summary[sd_key], 2.0)
        with self.assertRaisesRegex(ValueError, "One seed"):
            env["aggregate_seeds"](seed_rows + seed_rows[:1])


if __name__ == "__main__":
    unittest.main()
