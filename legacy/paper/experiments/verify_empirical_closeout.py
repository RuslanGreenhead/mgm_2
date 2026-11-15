"""CPU-only checks and explicit GPU ownership release after the fixed queue."""
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/paper_icml2026"
INDEPENDENT = BASE / "controlled_independent"
LONG = BASE / "stabilization_general_long"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    torch.set_num_threads(1)
    primary = read(INDEPENDENT / "verification.json")
    ready = read(INDEPENDENT / "FINAL_TABLES_READY.json")
    assert primary["all_checks_passed"] and primary["rerun_jobs"] == 54
    assert ready["metric_provenance_verified"]
    calibration = read(INDEPENDENT / "calibration_metric_provenance.json")
    assert calibration["metric_version"] == primary["sphere_metric_version"]
    assert calibration["arrays_unchanged"] and calibration["mmd_unchanged"]
    for path, expected in calibration["source_npz_sha256"].items():
        assert sha(Path(path)) == expected
    assert sha(BASE / "calibration_results.json") == calibration["corrected_results_sha256"]
    assert sha(INDEPENDENT / "calibration_results.json") == calibration["corrected_results_sha256"]
    calibration_rows = read(INDEPENDENT / "calibration_results.json")
    with zipfile.ZipFile(BASE / "sphere_metric_v1_calibration.zip") as archive:
        original_calibration = json.loads(archive.read("calibration_results.json"))
    row_key = lambda r: (r["data"], r["split"], r["draw"])
    original_mmd = {row_key(r): r["mmd2_biased"] for r in original_calibration}
    assert all(original_mmd[row_key(r)] == r["mmd2_biased"] for r in calibration_rows)
    sphere = {"fire", "flood", "volcano", "earthquake"}
    sphere_calibration = [r for r in calibration_rows if r["data"] in sphere]
    assert len(sphere_calibration) == 24
    assert all(r["metric_version"] == primary["sphere_metric_version"] for r in sphere_calibration)

    protocol = read(LONG / "protocol.json")
    frozen = {key.replace("\\", "/"): value
              for key, value in read(INDEPENDENT / "frozen_sources.json")["source_sha256"].items()}
    assert protocol["source_sha256"] == frozen["paper/experiments/stabilization_long.py"]
    assert protocol["base_source_sha256"] == frozen["paper/experiments/stabilization.py"]
    results = read(LONG / "results.json")
    assert len(results) == 2
    expected_trials = {"periodic_random_mlp_c5_lr2e4_g10000", "periodic_random_gmm_c5_lr2e4_g10000"}
    assert {r["trial"] for r in results} == expected_trials
    hashes, row_count, checkpoint_count = {}, 0, 0
    probes = [("initial", 0), ("warmup", 0), ("joint", 100), ("joint", 500),
              ("joint", 1000), ("joint", 2000), ("joint", 5000), ("final", 10000)]
    for result in results:
        assert result["steps"] == 10000 and result["critic_updates"] == 50500
        assert result["test_evaluations"] == 0 and result["training_seed"] == 0
        assert np.isfinite(result["final_validation_mmd2_mean"])
        folder = LONG / result["trial"]
        curve = read(folder / "learning_curve.json")
        assert curve[-1]["step"] == 10000 and len(curve) == 101
        for phase, step in probes:
            path = folder / f"{phase}_step{step}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            assert checkpoint["step"] == step and checkpoint["phase"] == phase
            for key in ("model", "critic"):
                assert all(torch.isfinite(t).all().item() for t in checkpoint[key].values() if t.is_floating_point())
            rows = read(folder / f"{phase}_step{step}_metrics.json")
            assert len(rows) == 6 and {r["split"] for r in rows} == {"train", "val"}
            assert all(np.isfinite(r["mmd2_biased"]) for r in rows)
            with np.load(folder / f"{phase}_step{step}_samples.npz") as arrays:
                assert set(arrays.files) == {f"generated_draw_{i}" for i in range(3)}
                assert all(arrays[k].shape == (1024, 2) and np.isfinite(arrays[k]).all() for k in arrays.files)
            row_count += len(rows)
            checkpoint_count += 1
        for path in folder.iterdir():
            if path.is_file():
                hashes[str(path.relative_to(LONG))] = sha(path)
    selection = read(LONG / "selection.json")
    assert selection["new_test_evaluations"] == 0
    assert selection["selected_trial"] == min(results, key=lambda r: r["final_validation_mmd2_mean"])["trial"]

    tests = subprocess.run([sys.executable, "-B", str(ROOT / "paper/experiments/test_controlled_suite.py")],
                           capture_output=True, text=True, check=False)
    (INDEPENDENT / "metric_regression_tests.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    assert tests.returncode == 0
    verification = {"all_checks_passed": True, "completed_trials": 2, "planned_trials": 2,
                    "checkpoints_verified": checkpoint_count, "train_validation_metric_rows": row_count,
                    "new_test_evaluations": 0, "source_sha256_matches_frozen_snapshot": True,
                    "file_sha256": hashes, "results": results,
                    "selection_metadata_note": "Inherited base config selection text mentions 2000; the long wrapper's operative rule uses fixed-final step 10000, verified here.",
                    "verification_script_sha256": sha(Path(__file__))}
    write(LONG / "verification.json", verification)
    release = {"released_at_utc": datetime.now(timezone.utc).isoformat(), "gpu_ownership": "released",
               "primary_reruns_completed": 54, "primary_reruns_planned": 54,
               "unchanged_provenance_copies": 18, "sphere_metric_version": primary["sphere_metric_version"],
               "sphere_primary_rows_corrected": primary["sphere_rows_corrected"],
               "sphere_calibration_rows_corrected": len(sphere_calibration),
               "long_validation_trials_completed": 2, "long_validation_trials_planned": 2,
               "all_checks_passed": True, "new_validation_study_test_evaluations": 0,
               "no_additional_gpu_jobs_queued_by_this_sidecar": True,
               "scope": "Releases this sidecar's ownership only; does not assert other users' GPU jobs are absent.",
               "primary_verification_sha256": sha(INDEPENDENT / "verification.json"),
               "long_verification_sha256": sha(LONG / "verification.json")}
    write(BASE / "GPU_RELEASED.json", release)
    print(json.dumps(release), flush=True)


if __name__ == "__main__":
    main()
