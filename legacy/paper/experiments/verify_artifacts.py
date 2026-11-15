"""Verify the finished empirical bundle and emit a factual machine-readable manifest."""
import importlib.metadata
import hashlib
import json
import shutil

import numpy as np
import torch

import controlled_suite as s


def main():
    torch.set_num_threads(1)
    primary = json.loads((s.OUT / "protocol.json").read_text())
    extension = json.loads((s.OUT / "mmd_protocol.json").read_text())
    jobs = primary["jobs"] + extension["jobs"]
    checks, failures, totals = [], [], {"training_jobs": 0, "checkpoints": 0, "evaluation_rows": 0, "sample_npz": 0}
    for name, method, seed in jobs:
        d = s.OUT / "controlled" / name / method / f"seed{seed}"
        expected_steps = 10000 if method == "rfm" else 2000
        try:
            c = json.loads((d / "completed.json").read_text())
            assert c["steps"] == expected_steps and c["finite_parameters"]
            if method == "mgm_gmm":
                assert c["parameter_change_l2"]["component_logits"] == 0.0
            steps = [2000, 10000] if method == "rfm" else ([0, 2000] if method == "mgm_gmm" else [2000])
            for step in steps:
                ck = torch.load(d / f"step{step}.pt", map_location="cpu", weights_only=True)
                assert ck["step"] == step and ck["method"] == method
                assert all(torch.isfinite(v).all() for v in ck["model"].values() if v.is_floating_point())
                totals["checkpoints"] += 1
                rows = json.loads((d / f"evaluation_step{step}.json").read_text())
                assert len(rows) == (12 if method == "rfm" else 6)
                for row in rows:
                    assert row["generated_n"] == row["target_n"] <= 1024
                    assert row["training_seed"] == seed and row["step"] == step
                    assert row["vector_field_nfe"] == (row["sampler_nfe"] if method == "rfm" else 0)
                    assert np.isfinite(row["mmd2_biased"]) and row["mmd2_biased"] > -1e-12
                totals["evaluation_rows"] += len(rows)
                for nfe in ([128, 1] if method == "rfm" else [1]):
                    a = np.load(d / f"samples_step{step}_nfe{nfe}.npz")
                    assert set(a.files) == {f"generated_draw_{i}" for i in range(3)}
                    assert all(np.isfinite(a[k]).all() for k in a.files)
                    totals["sample_npz"] += 1
            checks.append({"data": name, "method": method, "seed": seed, "steps": expected_steps,
                           "train_seconds": c["train_seconds"], "setup_seconds": c["setup_seconds_including_data_and_initialization"]})
            totals["training_jobs"] += 1
        except Exception as exc:
            failures.append({"data": name, "method": method, "seed": seed, "error": repr(exc)})
    for name in s.REAL:
        a = np.load(s.OUT / "datasets" / name / "split_indices_and_targets.npz")
        for x, y in [("train", "val"), ("train", "test"), ("val", "test")]:
            if np.intersect1d(a[x + "_indices"], a[y + "_indices"]).size:
                failures.append({"data": name, "error": "overlapping split indices"})
    mesh = json.loads((s.OUT / "legacy_mesh_results.json").read_text())
    assert len(mesh) == 54
    assert all(r["step"] == 800 and r["data_seed"] == 0 and r["target_n"] == 1024 and r["generated_n"] == 1024 for r in mesh)
    calibration = json.loads((s.OUT / "calibration_results.json").read_text())
    assert len(calibration) == 54 and all(r["n_training_seeds"] == 0 for r in calibration)
    stabilization = []
    for dirname, expected in (("stabilization_general", 6), ("stabilization_general_refinement", 2)):
        folder = s.OUT / dirname
        if not folder.exists():
            continue
        finished = json.loads((folder / "results.json").read_text())
        assert len(finished) == expected and all(r["test_evaluations"] == 0 for r in finished)
        probe_rows = 0
        for path in folder.glob("*/*_metrics.json"):
            probe = json.loads(path.read_text())
            assert all(r["split"] in ("train", "val") for r in probe)
            probe_rows += len(probe)
        for result in finished:
            if result["trial"] == "periodic_rfm":
                continue
            initial = torch.load(folder / result["trial"] / "initial_step0.pt", map_location="cpu", weights_only=True)
            warmup = torch.load(folder / result["trial"] / "warmup_step0.pt", map_location="cpu", weights_only=True)
            assert all(torch.equal(value, warmup["model"][key]) for key, value in initial["model"].items())
        selection = json.loads((folder / "selection.json").read_text())
        stabilization.append({"study": dirname, "planned_trials": expected, "finished_trials": len(finished),
                              "new_test_evaluations": 0, "train_val_probe_rows": probe_rows,
                              "train_seconds": sum(r["train_seconds"] for r in finished),
                              "selected_trial": selection["selected_trial"]})
    snapshot = s.OUT / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    source_hashes = {}
    for src in s.Path(__file__).parent.glob("*.py"):
        shutil.copyfile(src, snapshot / src.name)
        source_hashes[src.name] = s.sha(src)
    for src in s.Path(__file__).parent.glob("*.json"):
        shutil.copyfile(src, snapshot / src.name)
        source_hashes[src.name] = s.sha(src)
    # Recover the exact primary runner before the separately requested controls
    # and mesh diagnostics were added; verify bytes against its locked hash.
    original = s.Path(s.__file__).read_bytes().decode("utf-8")
    original = original.replace('        torch.random.default_generator.manual_seed(draw_seed)',
                                '        torch.manual_seed(draw_seed)', 1)
    original = original.replace('if method in ("mgm_mlp", "mmd_mlp"):', 'if method == "mgm_mlp":', 1)
    original = original.replace('critic = None if method in ("rfm", "mmd_mlp") else', 'critic = None if method == "rfm" else', 1)
    for first, last in [
        ('        elif method == "mmd_mlp":', '        else:'),
        ('            with torch.no_grad():\n                warmup =', '            for i in range(args.draws):'),
        ('                with torch.no_grad():\n                    sx =', '                row.update(data=name'),
        ('            if variant == "sphere_map_gmm":', '            print(json.dumps({"event": "mesh_finished"'),
    ]:
        begin = original.index(first)
        end = original.index(last, begin)
        original = original[:begin] + original[end:]
    primary_hash = hashlib.sha256(original.encode("utf-8")).hexdigest()
    expected_hash = primary["code_sha256"][str(s.Path(s.__file__).relative_to(s.ROOT))]
    assert primary_hash == expected_hash
    (snapshot / "primary_executed_runner.py").write_bytes(original.encode("utf-8"))
    packages = {p.metadata["Name"]: p.version for p in importlib.metadata.distributions() if p.metadata["Name"]}
    s.write_json(s.OUT / "package_versions.json", packages)
    main_legacy_hashes_match = {path: s.sha(s.ROOT / path) == digest
                               for path, digest in primary["code_sha256"].items() if path.startswith("external")}
    manifest = {"verified_at": s.utc(), "all_planned_jobs_completed": len(checks) == len(jobs) and not failures,
                "planned_training_jobs": len(jobs), "totals": totals, "failures": failures,
                "completed_jobs": checks, "mesh_checkpoints": 18, "mesh_mc_rows": len(mesh),
                "calibration_rows": len(calibration), "legacy_trainer_hashes_unchanged": main_legacy_hashes_match,
                "validation_only_stabilization": stabilization,
                "final_supplement_source_sha256": source_hashes,
                "primary_executed_runner_sha256": primary_hash,
                "training_seconds_sum_completed_jobs": sum(c["train_seconds"] for c in checks),
                "setup_seconds_sum_completed_jobs": sum(c["setup_seconds"] for c in checks),
                "compute_note": "Synchronized training times exclude evaluation, failed startup MLP attempt and 100-step probes. Logs preserve full wall times. One failed initial MLP CPU-timing attempt was rerun.",
                "scope": "No original trainer/manuscript/figure edits. No author-budget or one-step-SOTA reproduction claimed."}
    s.write_json(s.OUT / "run_manifest.json", manifest)
    s.summarize()
    print(json.dumps({"all_planned_jobs_completed": manifest["all_planned_jobs_completed"], "totals": totals,
                      "mesh_rows": len(mesh), "failures": failures,
                      "training_seconds_sum": manifest["training_seconds_sum_completed_jobs"]}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
