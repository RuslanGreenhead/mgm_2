"""CPU-only queue/profile regressions; all processes and data are synthetic."""

import copy
import io
import os
import random
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import CancelledError, as_completed as real_as_completed
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from . import data_metrics, gpu_guard, mgm, profile_all, run_jobs, worker
from .common import ManifoldOps


class _CPUOnlyTorch:
    def __init__(self):
        self.cuda = SimpleNamespace(manual_seed_all=lambda seed: None,
                                    current_device=lambda: 0)
        self.random = SimpleNamespace(
            fork_rng=lambda **kwargs: torch.random.fork_rng(devices=[]))

    def device(self, name):
        if name != "cuda":
            raise AssertionError(f"Unexpected device: {name}")
        return torch.device("cpu")

    def manual_seed(self, seed):
        return torch.random.default_generator.manual_seed(seed)

    def __getattr__(self, name):
        return getattr(torch, name)


class _Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.out = self.root / "outputs/comparison2026"
        self.phase = self.out / "synthetic_phase"
        self.stack.enter_context(torch.random.fork_rng(devices=[]))
        py_state, np_state = random.getstate(), np.random.get_state()
        self.addCleanup(random.setstate, py_state)
        self.addCleanup(np.random.set_state, np_state)
        self.stack.enter_context(patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")))
        self.stack.enter_context(patch.object(
            torch.cuda, "synchronize", side_effect=AssertionError("CUDA forbidden")))
        # No test may accidentally launch nvidia-smi, Python, or another child.
        self.stack.enter_context(patch.object(
            subprocess, "Popen", side_effect=AssertionError("Real process forbidden")))
        self.stack.enter_context(patch.object(
            subprocess, "run", side_effect=AssertionError("Real process forbidden")))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        source = self.root / "paper/benchmarks2026"
        source.mkdir(parents=True)
        (source / "worker.py").write_text("# synthetic frozen worker\n", encoding="ascii")
        (source / "common.py").write_text("# synthetic dependency\n", encoding="ascii")
        (source / "FIXTURE_NOTES.md").write_text("Synthetic notes.\n", encoding="ascii")
        self.jobs = [self.job(i) for i in range(5)]
        self.plan = {"phase": self.phase.name, "max_workers": 2, "jobs": self.jobs}
        self.plan_path = self.root / "plan.json"
        data_metrics.write(self.plan_path, self.plan)

    def job(self, index):
        return {"data": "synthetic", "method": "mgm", "seed": index,
                "steps": 7, "score_test": True, "concurrent_training_workers": 2,
                "output": f"outputs/comparison2026/synthetic_phase/job_{index}",
                "config": {"width": 8, "depth": 2, "critic_warmup": 0,
                           "critic_steps": 1}}

    def done(self, index):
        job = self.jobs[index]
        return {"data": job["data"], "method": job["method"], "seed": job["seed"],
                "train_seconds": index + 0.5, "selected_step": 3}

    def freeze(self):
        snapshot = self.phase / "source_snapshot"
        relative = Path("paper/benchmarks2026/worker.py")
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# synthetic frozen worker\n", encoding="ascii")
        data_metrics.write(self.phase / "protocol.json", {
            "plan": self.plan, "source_sha256": {str(relative): data_metrics.sha(target)}})
        for index, job in enumerate(self.jobs):
            folder = self.root / job["output"]
            data_metrics.write(folder / "completed.json", self.done(index))
            (folder / "best.pt").write_bytes(b"synthetic selected checkpoint")
            data_metrics.write(self.phase / "jobs" / f"{index:03d}.json", job)
        data_metrics.write(self.phase / "completed.json", {"all_completed": True})

    def assert_state_equal(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)


class _Processes:
    """Real executor threads, with deterministic gates instead of child processes."""

    def __init__(self, case, failures=()):
        self.case = case
        self.failures = set(failures)
        self.condition = threading.Condition()
        self.calls, self.exited, self.runtime = [], [], []
        self.gates, self.active = {}, set()
        self.maximum = 0
        self.release_everything = False
        self.third_started = threading.Event()

    def popen(self, command, **kwargs):
        path = Path(command[command.index("--job") + 1])
        index = int(path.stem)
        job = data_metrics.read(path)
        with self.condition:
            self.calls.append((index, command, kwargs))
            gate = self.gates[index] = threading.Event()
            if self.release_everything:
                gate.set()
            self.active.add(index)
            self.maximum = max(self.maximum, len(self.active))
            if len(self.calls) >= 3:
                self.third_started.set()
            self.condition.notify_all()

        def wait():
            if not gate.wait(5):
                raise AssertionError(f"Synthetic process {index} was not released")
            code = 17 if index in self.failures else 0
            if not code:
                data_metrics.write(self.case.root / job["output"] / "completed.json",
                                   self.case.done(index))
            with self.condition:
                self.active.remove(index)
                self.exited.append(index)
                self.condition.notify_all()
            return code

        return SimpleNamespace(pid=42000 + index, wait=wait)

    def write(self, path, value):
        data_metrics.write(path, value)
        with self.condition:
            if Path(path).name == "runtime.json":
                self.runtime.append(copy.deepcopy(value))
            self.condition.notify_all()

    def until(self, predicate):
        with self.condition:
            if not self.condition.wait_for(predicate, timeout=5):
                raise AssertionError("Timed out waiting for synthetic queue state")

    def release(self, index):
        with self.condition:
            self.gates[index].set()

    def release_all(self):
        with self.condition:
            self.release_everything = True
            for gate in self.gates.values():
                gate.set()


class QueueTests(_Fixture):
    def start_queue(self, processes, completed_iterator=None):
        for name, value in (("ROOT", self.root), ("OUT", self.out),
                            ("write", processes.write)):
            self.stack.enter_context(patch.object(run_jobs, name, value))
        self.guard = self.stack.enter_context(patch.object(run_jobs, "wait_until_free"))
        self.stack.enter_context(patch.object(run_jobs.subprocess, "Popen", processes.popen))
        self.stack.enter_context(patch.object(sys, "argv", [
            "run_jobs", "--plan", str(self.plan_path)]))
        if completed_iterator is not None:
            self.stack.enter_context(patch.object(run_jobs, "as_completed", completed_iterator))
        self.queue_errors = []
        self.finished = threading.Event()

        def invoke():
            try:
                run_jobs.main()
            except BaseException as error:
                self.queue_errors.append(error)
            finally:
                self.finished.set()

        self.thread = threading.Thread(target=invoke, name="synthetic-queue")
        self.thread.start()

        def cleanup():
            processes.release_all()
            self.thread.join(10)
            self.assertFalse(self.thread.is_alive(), "Queue thread failed to exit")

        self.addCleanup(cleanup)

    def join_queue(self):
        self.assertTrue(self.finished.wait(5), "Queue did not exit")
        self.thread.join(5)

    def test_two_process_cap_and_pid_runtime_bookkeeping(self):
        processes = _Processes(self)
        self.start_queue(processes)
        processes.until(lambda: any(len(row["workers"]) == 2 for row in processes.runtime))
        self.assertEqual(len(processes.calls), 2)
        self.assertEqual(processes.maximum, 2)
        processes.release(0)
        processes.until(lambda: len(processes.calls) == 3)
        self.assertEqual(processes.maximum, 2)
        processes.release_all()
        self.join_queue()
        self.assertEqual(self.queue_errors, [])
        for row in processes.runtime:
            self.assertEqual(row["queue_pid"], os.getpid())
            self.assertEqual(row["max_workers"], 2)
            self.assertLessEqual(len(row["workers"]), 2)
            for key, info in row["workers"].items():
                self.assertEqual(info["worker_pid"], 42000 + int(key))
                self.assertEqual(info["job"], self.jobs[int(key)])
                self.assertGreaterEqual(info["started_unix"], row["started_unix"])
                self.assertEqual(Path(info["worker_path"]),
                                 self.phase / "source_snapshot/paper/benchmarks2026/worker.py")
        self.assertEqual(processes.runtime[-1]["status"], "completed")
        self.assertEqual(processes.runtime[-1]["workers"], {})
        done = data_metrics.read(self.phase / "completed.json")
        self.assertEqual({row["seed"] for row in done["jobs"]}, set(range(5)))
        self.assertEqual(done["summed_training_seconds"], sum(self.done(i)["train_seconds"] for i in range(5)))
        self.assertIn("contention", done["training_times"])
        self.assertIn("not an isolated", done["training_times"])
        self.assertEqual(self.guard.call_count, 5)
        stop_events = [call.kwargs["stop_event"] for call in self.guard.call_args_list]
        self.assertTrue(all(event is stop_events[0] for event in stop_events))
        self.assertFalse(stop_events[0].is_set())

    def test_single_worker_plan_is_serial(self):
        self.plan["max_workers"] = 1
        data_metrics.write(self.plan_path, self.plan)
        processes = _Processes(self)
        self.start_queue(processes)
        processes.until(lambda: bool(processes.runtime))
        self.assertEqual(len(processes.calls), 1)
        processes.release_all()
        self.join_queue()
        self.assertEqual(self.queue_errors, [])
        self.assertEqual(processes.maximum, 1)

    def test_completed_jobs_skip_and_resume_without_relaunch(self):
        for index in (0, 2):
            data_metrics.write(self.root / self.jobs[index]["output"] / "completed.json", self.done(index))
        processes = _Processes(self)
        processes.release_all()
        self.start_queue(processes)
        self.join_queue()
        self.assertEqual(self.queue_errors, [])
        self.assertEqual({row[0] for row in processes.calls}, {1, 3, 4})
        frozen = (self.phase / "protocol.json").read_bytes()
        calls_before = len(processes.calls)
        run_jobs.main()
        self.assertEqual(len(processes.calls), calls_before)
        self.assertEqual((self.phase / "protocol.json").read_bytes(), frozen)
        self.assertEqual(len(data_metrics.read(self.phase / "completed.json")["jobs"]), 5)

    def test_source_snapshot_paths_and_frozen_worker_command(self):
        self.jobs[0]["method"] = "kgdm"
        author = self.root / "external/kgdm-author/earth/drift_core.py"
        author.parent.mkdir(parents=True)
        author.write_text("# synthetic author field\n", encoding="ascii")
        (author.parents[1] / "LICENSE").write_text("Synthetic license\n", encoding="ascii")
        data_metrics.write(self.plan_path, self.plan)
        processes = _Processes(self)
        processes.release_all()
        self.start_queue(processes)
        self.join_queue()
        self.assertEqual(self.queue_errors, [])
        protocol = data_metrics.read(self.phase / "protocol.json")
        self.assertEqual(protocol["plan"], self.plan)
        self.assertEqual(protocol["plan_sha256"], data_metrics.sha(self.plan_path))
        self.assertEqual({Path(rel).as_posix() for rel in protocol["source_sha256"]}, {
            "paper/benchmarks2026/worker.py", "paper/benchmarks2026/common.py",
            "paper/benchmarks2026/FIXTURE_NOTES.md", "external/kgdm-author/earth/drift_core.py",
            "external/kgdm-author/LICENSE"})
        for rel, digest in protocol["source_sha256"].items():
            frozen = self.phase / "source_snapshot" / rel
            self.assertEqual(frozen.read_bytes(), (self.root / rel).read_bytes())
            self.assertEqual(data_metrics.sha(frozen), digest)
        for index, command, kwargs in processes.calls:
            self.assertEqual(command[:3], [sys.executable, "-u", str(
                self.phase / "source_snapshot/paper/benchmarks2026/worker.py")])
            self.assertEqual(command[3:], ["--root", str(self.root), "--job",
                                          str(self.phase / "jobs" / f"{index:03d}.json")])
            self.assertEqual(kwargs["cwd"], self.root)
            self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
            self.assertNotIn("shell", kwargs)

    def test_failure_cancels_queued_and_waits_active_exit_without_phase_completion(self):
        processes = _Processes(self, failures=(0,))

        def delayed_consumer(futures):
            order = getattr(self, "failure_order", "delayed")
            iterator = iter(real_as_completed(futures))
            if order in ("failure_first", "cancel_first", "reverse_cancel_first"):
                # Job 1 remains blocked; failure and all abort cancellations finish.
                ready = [next(iterator) for _ in range(len(self.jobs) - 1)]
                cancelled = [future for future in ready
                             if isinstance(future.exception(), CancelledError)]
                self.assertEqual(len(cancelled), len(self.jobs) - 2)
                ready.sort(key=lambda future: (
                    0 if futures[future][0] == 0 else 1,
                    futures[future][0]))
                if order != "failure_first":
                    ready = ready[1:] + ready[:1]
                if order == "reverse_cancel_first":
                    ready = list(reversed(ready[:-1])) + ready[-1:]
                yield from ready
                yield from iterator
                return
            # A legal scheduling order: executor gets CPU before the failure consumer.
            for future in iterator:
                if (order == "delayed" and not future.cancelled()
                        and future.exception() is not None):
                    processes.third_started.wait(0.25)
                yield future

        self.start_queue(processes, delayed_consumer)
        processes.until(lambda: any(len(row["workers"]) == 2 for row in processes.runtime))
        processes.release(0)
        processes.until(lambda: (self.phase / "queue_failure.json").exists())
        self.assertFalse(self.finished.is_set(), "Queue returned before active worker exit")
        self.assertNotIn(1, processes.exited)
        self.assertFalse((self.phase / "completed.json").exists())
        processes.release_all()
        self.join_queue()
        self.assertEqual(len(self.queue_errors), 1)
        self.assertIsInstance(self.queue_errors[0], RuntimeError)
        self.assertIn(1, processes.exited)
        self.assertEqual(processes.runtime[-1]["status"], "failed")
        self.assertEqual(processes.runtime[-1]["workers"], {})
        self.assertFalse((self.phase / "completed.json").exists())
        self.assertEqual(data_metrics.read(self.root / self.jobs[0]["output"] / "failure.json")["return_code"], 17)
        self.assertEqual(data_metrics.read(self.phase / "queue_failure.json")["index"], 0)
        self.assertEqual({row[0] for row in processes.calls}, {0, 1},
                         "A queued job launched after the first worker failure")

    def test_runtime_publication_failure_after_spawn_still_waits_for_child_exit(self):
        processes = _Processes(self)
        real_write = processes.write
        failed = threading.Event()

        def write(path, value):
            if (Path(path).name == "runtime.json" and value["workers"]
                    and not failed.is_set()):
                failed.set()
                raise OSError("Synthetic runtime publication failure after Popen")
            real_write(path, value)

        processes.write = write
        self.start_queue(processes)
        self.assertTrue(failed.wait(5), "Runtime failure was not injected")
        returned_before_child_exit = self.finished.wait(0.1)
        processes.release_all()
        self.join_queue()
        self.assertTrue(failed.is_set())
        self.assertFalse((self.phase / "completed.json").exists())
        self.assertEqual(len(self.queue_errors), 1)
        self.assertIsInstance(self.queue_errors[0], OSError)
        self.assertFalse(returned_before_child_exit,
                         "Queue returned without waiting for its already spawned child")
        self.assertEqual(processes.active, set())
        self.assertEqual(processes.runtime[-1]["workers"], {})

    def test_tampered_snapshot_and_changed_frozen_plan_rejected_before_spawn(self):
        self.freeze()
        for job in self.jobs:
            (self.root / job["output"] / "completed.json").unlink()
        target = self.phase / "source_snapshot/paper/benchmarks2026/worker.py"
        target.write_text("# tampered\n", encoding="ascii")
        processes = _Processes(self)
        processes.release_all()
        self.start_queue(processes)
        self.join_queue()
        self.assertIsInstance(self.queue_errors[0], AssertionError)
        self.assertEqual(processes.calls, [])
        self.plan["max_workers"] = 1
        data_metrics.write(self.plan_path, self.plan)
        with self.assertRaises(AssertionError):
            run_jobs.main()
        self.assertEqual(processes.calls, [])

    def test_unvalidated_concurrency_rejected_before_spawn(self):
        for workers in (0, 3, -1, "2"):
            with self.subTest(workers=workers):
                self.plan["max_workers"] = workers
                data_metrics.write(self.plan_path, self.plan)
                with patch.object(run_jobs, "ROOT", self.root), patch.object(run_jobs, "OUT", self.out), \
                        patch.object(sys, "argv", ["run_jobs", "--plan", str(self.plan_path)]):
                    with self.assertRaisesRegex(ValueError, "concurrency"):
                        run_jobs.main()


class ProfileDriverTests(_Fixture):
    def invoke(self, blocked=(), run=None):
        with ExitStack() as stack:
            stack.enter_context(patch.object(profile_all, "ROOT", self.root))
            stack.enter_context(patch.object(profile_all, "OUT", self.out))
            guard = stack.enter_context(patch.object(profile_all, "foreign_python", return_value=blocked))
            runner = stack.enter_context(patch.object(profile_all.subprocess, "run", side_effect=run))
            stack.enter_context(patch.object(sys, "argv", ["profile_all", "--phase", self.phase.name]))
            try:
                profile_all.main()
            finally:
                self.guard, self.runner = guard, runner

    def test_requires_completed_phase_and_no_other_gpu_python(self):
        self.freeze()
        for state in (None, False):
            with self.subTest(state=state):
                marker = self.phase / "completed.json"
                if state is None:
                    marker.unlink()
                else:
                    data_metrics.write(marker, {"all_completed": state})
                with self.assertRaises((FileNotFoundError, AssertionError)):
                    self.invoke()
                self.runner.assert_not_called()
        data_metrics.write(self.phase / "completed.json", {"all_completed": True})
        with self.assertRaisesRegex(RuntimeError, "GPU Python"):
            self.invoke(blocked=[{"pid": 999, "process": "synthetic/python.exe"}])
        self.guard.assert_called_once_with(include_project=True)
        self.runner.assert_not_called()
        self.assertFalse((self.phase / "profiles_completed.json").exists())

    def test_profiles_serial_frozen_paths_and_resumes_valid_profiles(self):
        self.freeze()
        folder = self.root / self.jobs[0]["output"]
        data_metrics.write(folder / "isolated_latency.json", {
            "isolated": True, "selected_checkpoint_sha256": data_metrics.sha(folder / "best.pt")})
        old = (folder / "isolated_latency.json").read_bytes()
        calls, active = [], []

        def run(command, **kwargs):
            self.assertEqual(active, [])
            active.append(command)
            index = int(Path(command[command.index("--job") + 1]).stem)
            self.assertEqual(command, [sys.executable, "-u", str(
                self.phase / "source_snapshot/paper/benchmarks2026/worker.py"),
                "--root", str(self.root), "--job", str(self.phase / "jobs" / f"{index:03d}.json"),
                "--latency-only"])
            self.assertEqual(data_metrics.read(Path(command[-2])), self.jobs[index])
            self.assertEqual(kwargs["cwd"], self.root)
            self.assertIs(kwargs["check"], True)
            calls.append(index)
            selected = self.root / self.jobs[index]["output"]
            data_metrics.write(selected / "isolated_latency.json", {
                "isolated": True, "selected_checkpoint_sha256": data_metrics.sha(selected / "best.pt")})
            active.pop()

        self.invoke(run=run)
        self.guard.assert_called_once_with(include_project=True)
        self.assertEqual(calls, [1, 2, 3, 4])
        self.assertEqual((folder / "isolated_latency.json").read_bytes(), old)
        self.assertEqual(data_metrics.read(self.phase / "profiles_completed.json"), {
            "all_completed": True, "jobs": 5, "isolated": True})
        self.invoke(run=run)
        self.runner.assert_not_called()

    def test_rejects_nonisolated_or_stale_checkpoint_profile(self):
        self.freeze()
        folder = self.root / self.jobs[0]["output"]
        for isolated, digest in ((False, data_metrics.sha(folder / "best.pt")), (True, "stale")):
            with self.subTest(isolated=isolated, digest=digest):
                data_metrics.write(folder / "isolated_latency.json", {
                    "isolated": isolated, "selected_checkpoint_sha256": digest})
                with self.assertRaises(AssertionError):
                    self.invoke()
                self.runner.assert_not_called()
                self.assertFalse((self.phase / "profiles_completed.json").exists())

    def test_snapshot_tamper_or_worker_failure_never_publishes_profiles_completed(self):
        self.freeze()
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke(run=subprocess.CalledProcessError(17, "synthetic worker"))
        self.assertEqual(self.runner.call_count, 1)
        self.assertFalse((self.phase / "profiles_completed.json").exists())
        (self.phase / "source_snapshot/paper/benchmarks2026/worker.py").write_text(
            "# tampered\n", encoding="ascii")
        with self.assertRaises(AssertionError):
            self.invoke()
        self.runner.assert_not_called()
        self.assertFalse((self.phase / "profiles_completed.json").exists())


class LatencyWorkerTests(_Fixture):
    def setUp(self):
        super().setUp()
        self.job_path = self.root / "job.json"
        self.job_config = self.jobs[0]
        data_metrics.write(self.job_path, self.job_config)
        self.folder = self.root / self.job_config["output"]
        data_metrics.write(self.folder / "completed.json", self.done(0))
        ops = ManifoldOps("torus", 2)
        method = mgm.create_method(ops, self.job_config["config"], "cpu")
        method.iteration = 17
        # Nonempty optimizer state makes mutation checks meaningful without training.
        for optimizer in (method.opt, method.opt_c):
            parameter = optimizer.param_groups[0]["params"][0]
            optimizer.state[parameter] = {"step": torch.tensor(11.),
                                          "exp_avg": torch.ones_like(parameter),
                                          "exp_avg_sq": torch.full_like(parameter, 2.)}
        self.selected = {"job": self.job_config, "method_state": method.checkpoint(),
                         "step": 3, "attempts": 17, "train_seconds": 12.5,
                         "rng": {"target": torch.tensor([1, 2]), "method": torch.tensor([3, 4])},
                         "stage": "training"}
        for name in ("best.pt", "last.pt", "interrupted.pt"):
            torch.save(self.selected, self.folder / name)
        for name in ("job.json", "curve.json", "progress.json", "latency.json"):
            data_metrics.write(self.folder / name, {"sentinel": name})
        data = self.out / "data/synthetic"
        data.mkdir(parents=True)
        (data / "data.npz").write_bytes(b"synthetic archive; no arrays or held-out data")
        data_metrics.write(data / "manifest.json", {
            "kind": "torus", "dim": 2, "arrays_sha256": data_metrics.sha(data / "data.npz")})

    def invoke(self, blocked=()):
        self.created = []
        real_create, real_load = mgm.create_method, torch.load

        def create(*args):
            method = real_create(*args)
            method.update = self.stack.enter_context(patch.object(
                method, "update", side_effect=AssertionError("Profiling must not train")))
            method.sample = self.stack.enter_context(patch.object(method, "sample", wraps=method.sample))
            original = method.load_checkpoint

            def restore(state):
                original(state)
                self.loaded_state = copy.deepcopy(method.checkpoint())

            method.load_checkpoint = restore
            self.created.append(method)
            return method

        with ExitStack() as stack:
            stack.enter_context(patch.object(worker, "torch", _CPUOnlyTorch()))
            stack.enter_context(patch.object(worker, "synchronize", side_effect=(
                float(i) / 1000 for i in range(100))))
            self.guard = stack.enter_context(patch.object(worker, "foreign_python", return_value=blocked))
            self.arrays = stack.enter_context(patch.object(
                worker.np, "load", side_effect=AssertionError("No train/val/test array reads")))
            self.metric = stack.enter_context(patch.object(
                worker, "metrics", side_effect=AssertionError("No held-out scoring")))
            self.loader = stack.enter_context(patch.object(torch, "load", wraps=real_load))
            stack.enter_context(patch.object(
                torch, "save", side_effect=AssertionError("No checkpoint writes during profiling")))
            stack.enter_context(patch.object(mgm, "create_method", create))
            stack.enter_context(patch.object(sys, "argv", ["worker", "--job", str(self.job_path),
                                                         "--root", str(self.root), "--latency-only"]))
            worker.main()

    def test_isolated_profile_never_reads_arrays_or_advances_weights_training_state(self):
        before = {path.name: path.read_bytes() for path in self.folder.iterdir()}
        self.invoke()
        self.guard.assert_called_once_with(include_project=True)
        self.arrays.assert_not_called()
        self.metric.assert_not_called()
        self.assertEqual(self.loader.call_count, 1)
        self.assertEqual(self.loader.call_args.args[0], self.folder / "best.pt")
        self.assertEqual(len(self.created), 1)
        method = self.created[0]
        method.update.assert_not_called()
        self.assertEqual(method.sample.call_count, 35)
        self.assert_state_equal(self.selected["method_state"], self.loaded_state)
        self.assert_state_equal(self.loaded_state, method.checkpoint())
        for name, content in before.items():
            self.assertEqual((self.folder / name).read_bytes(), content, name)
        self.assertEqual({path.name for path in self.folder.iterdir()}, set(before) | {"isolated_latency.json"})
        profile = data_metrics.read(self.folder / "isolated_latency.json")
        self.assertIs(profile["isolated"], True)
        self.assertEqual((profile["batch"], profile["warmups"], len(profile["gpu_seconds"])), (1024, 5, 30))
        self.assertEqual(profile["model_forward_calls_per_draw"], 1)
        self.assertEqual(profile["selected_checkpoint_sha256"], data_metrics.sha(self.folder / "best.pt"))
        self.assertEqual(profile["worker_sha256"], data_metrics.sha(worker.__file__))
        self.assertTrue(all(value > 0 for value in profile["gpu_seconds"]))

    def test_missing_completed_job_or_other_gpu_worker_rejected(self):
        (self.folder / "completed.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "completed training job"):
            self.invoke()
        self.assertEqual(self.created, [])
        self.arrays.assert_not_called()
        data_metrics.write(self.folder / "completed.json", self.done(0))
        with self.assertRaisesRegex(RuntimeError, "GPU Python"):
            self.invoke(blocked=[{"pid": 999, "process": "synthetic/python.exe"}])
        self.guard.assert_called_once_with(include_project=True)
        self.assertEqual(self.created, [])
        self.loader.assert_not_called()
        self.assertFalse((self.folder / "isolated_latency.json").exists())

    def test_selected_checkpoint_job_identity_mismatch_rejected(self):
        self.selected["job"] = {**self.job_config, "seed": 999}
        torch.save(self.selected, self.folder / "best.pt")
        with self.assertRaisesRegex(ValueError, "identity"):
            self.invoke()
        self.arrays.assert_not_called()
        self.created[0].sample.assert_not_called()
        self.assertFalse((self.folder / "isolated_latency.json").exists())

    def test_sampling_profile_restores_cpu_rng_and_removes_hooks(self):
        method = mgm.create_method(ManifoldOps("torus", 2), self.job_config["config"], "cpu")
        rng = torch.random.get_rng_state().clone()
        with patch.object(worker, "torch", _CPUOnlyTorch()), \
                patch.object(worker, "synchronize", side_effect=(float(i) for i in range(100))):
            worker.sampling_profile(method, self.folder, "cpu", isolated=True)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(len(method.model._forward_hooks), 0)


class GpuGuardTests(_Fixture):
    def test_include_project_blocks_project_workers_but_excludes_self(self):
        text = (f"{os.getpid()}, C:\\synthetic\\python.exe\n"
                "42001, E:\\Codex Projects\\Riemmanian Midpoint Generative Models\\.conda\\rmgm\\python.exe\n"
                "42002, C:\\foreign\\PYTHON.EXE\n"
                "42003, C:\\foreign\\pythonw.exe\n"
                "42004, C:\\foreign\\non_python.exe\n")
        with patch.object(gpu_guard.subprocess, "run", return_value=SimpleNamespace(stdout=text)) as run:
            self.assertEqual({row["pid"] for row in gpu_guard.foreign_python()}, {42002, 42003})
            self.assertEqual({row["pid"] for row in gpu_guard.foreign_python(include_project=True)},
                             {42001, 42002, 42003})
        self.assertTrue(all(call.args[0][0] == "nvidia-smi" for call in run.call_args_list))

    def test_training_guard_waits_without_modifying_another_process(self):
        with patch.object(gpu_guard, "foreign_python", side_effect=[[{"pid": 999}], []]) as guard, \
                patch.object(gpu_guard.time, "sleep") as sleep:
            gpu_guard.wait_until_free()
        self.assertEqual(guard.call_count, 2)
        sleep.assert_called_once_with(60)
        stop = threading.Event()
        with patch.object(gpu_guard, "foreign_python", return_value=[{"pid": 999}]) as guard, \
                patch.object(stop, "wait", side_effect=lambda seconds: stop.set()) as wait, \
                patch.object(gpu_guard.time, "sleep", side_effect=AssertionError("Not interruptible")):
            with self.assertRaises(CancelledError):
                gpu_guard.wait_until_free(stop_event=stop)
        wait.assert_called_once_with(60)
        guard.assert_called_once_with()
        with patch.object(gpu_guard, "foreign_python") as guard:
            with self.assertRaises(CancelledError):
                gpu_guard.wait_until_free(stop_event=stop)
        guard.assert_not_called()


if __name__ == "__main__":
    unittest.main()
