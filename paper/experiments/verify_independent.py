"""CPU-only final verification of the independent RNG rerun bundle."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import zipfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/paper_icml2026"
DEST = BASE / "controlled_independent"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    torch.set_num_threads(1)
    protocol = json.loads((DEST / "protocol.json").read_text())
    manifest = json.loads((DEST / "run_manifest.json").read_text())
    assert manifest["all_completed"] and manifest["rerun_jobs"] == 54
    assert manifest["copied_unchanged_jobs"] == 18
    frozen = json.loads((DEST / "frozen_sources.json").read_text())
    for relative, expected in frozen["source_sha256"].items():
        assert sha(DEST / "source_snapshot" / relative) == expected
    modules = json.loads((DEST / "executed_source_hashes.json").read_text())
    assert modules and all(m["verified"] and frozen["source_sha256"][m["path"]] == m["sha256"] for m in modules)
    provenance = json.loads((DEST / "provenance.json").read_text())
    assert len(provenance) == 72
    metric = json.loads((DEST / "metric_provenance.json").read_text())
    assert metric["metric_version"] == "sphere_normalized_float64_atan2_v2" and metric["corrected_rows"] == 468
    corrected_files = {item["path"]: item for item in metric["corrected_files"]}
    raw_archive = zipfile.ZipFile(DEST / "sphere_metric_v1_raw.zip")
    for relative, item in corrected_files.items():
        assert sha(DEST / relative) == item["corrected_json_sha256"]
        assert hashlib.sha256(raw_archive.read(item["archive_member"])).hexdigest() == item["raw_json_sha256"]
    for name, expected in metric["metric_source_sha256"].items():
        assert sha(DEST / "metric_source_snapshot" / name) == expected
    copy_checks = 0
    corrected_copy_checks = 0
    for item in provenance:
        if item["origin"] != "copied_unchanged":
            continue
        src = Path(item["source_directory"])
        dest = DEST / "controlled" / item["data"] / item["method"] / f'seed{item["training_seed"]}'
        for relative, expected in item["copied_sha256"].items():
            assert sha(src / relative) == expected
            root_relative = str((dest / relative).relative_to(DEST))
            if root_relative in corrected_files:
                assert corrected_files[root_relative]["raw_json_sha256"] == expected
                corrected_copy_checks += 1
            else:
                assert sha(dest / relative) == expected
            copy_checks += 1
    rng_groups, row_count, checkpoint_count, invariant_count = {}, 0, 0, 0
    for name, method, seed in protocol["rerun_jobs"]:
        folder = DEST / "controlled" / name / method / f"seed{seed}"
        done = json.loads((folder / "completed.json").read_text())
        assert done["finite_parameters"] and done["evaluation_rng_invariance_verified"]
        for step in ([0, 2000] if method == "mgm_gmm" else [2000, 10000]):
            state = torch.load(folder / f"step{step}.pt", map_location="cpu", weights_only=True)
            assert state["step"] == step and state["method"] == method
            assert all(torch.isfinite(v).all().item() for v in state["model"].values() if v.is_floating_point())
            assert state["source_sha256"] == frozen["source_sha256"]
            assert set(state["rng_states"]) == {"target", "latent", "path", "base", "global_cpu", "global_cuda"}
            for stream, value in state["rng_states"].items():
                digest = hashlib.sha256(value.numpy().tobytes()).hexdigest()
                rng_groups.setdefault((name, method, step, stream), []).append(digest)
            invariance = json.loads((folder / f"rng_invariance_step{step}.json").read_text())
            assert invariance["all_streams_unchanged"]
            invariant_count += 1
            rows = json.loads((folder / f"evaluation_step{step}.json").read_text())
            assert len(rows) == (6 if method == "mgm_gmm" else 12)
            assert all(r["generated_n"] == r["target_n"] <= 1024 and np.isfinite(r["mmd2_biased"]) for r in rows)
            assert all(r["vector_field_nfe"] == (r["sampler_nfe"] if method == "rfm" else 0) for r in rows)
            row_count += len(rows)
            checkpoint_count += 1
            for nfe in ([1] if method == "mgm_gmm" else [1, 128]):
                data = np.load(folder / f"samples_step{step}_nfe{nfe}.npz")
                assert set(data.files) == {f"generated_draw_{i}" for i in range(3)}
                assert all(np.isfinite(data[k]).all() for k in data.files)
    assert all(len(values) == 3 and len(set(values)) == 3 for values in rng_groups.values())
    for name in protocol["rerun_jobs"]:
        data_name = name[0]
        before = json.loads((BASE / "datasets" / data_name / "manifest.json").read_text())
        after = json.loads((DEST / "datasets" / data_name / "manifest.json").read_text())
        assert all(after[k] == value for k, value in before.items() if k != "coverage_radius")
        if data_name in metric["training_radius_changes"]:
            assert after["coverage_radius"] == metric["training_radius_changes"][data_name]["corrected_radius"]
        else:
            assert before["coverage_radius"] == after["coverage_radius"]
    all_rows = json.loads((DEST / "results.json").read_text())
    assert len(all_rows) == 1080
    sphere_rows = [r for r in all_rows if r["data"] in ("volcano", "earthquake", "fire", "flood")]
    assert len(sphere_rows) == 468 and all(r["metric_version"] == metric["metric_version"] for r in sphere_rows)
    raw_rows = json.loads(raw_archive.read("results.json"))
    key = lambda r: (r["data"], r["method"], r["training_seed"], r["step"], r["sampler_nfe"], r["split"], r["draw"])
    old_mmd = {key(r): r["mmd2_biased"] for r in raw_rows}
    assert all(old_mmd[key(r)] == r["mmd2_biased"] for r in all_rows)
    assert all(sha(path) == value for path, value in metric["source_npz_sha256"].items())
    raw_archive.close()
    packages = {p.metadata["Name"]: p.version for p in importlib.metadata.distributions() if p.metadata["Name"]}
    write(DEST / "package_versions.json", packages)
    result = {"all_checks_passed": True, "rerun_jobs": 54, "copied_unchanged_jobs": 18,
              "rerun_checkpoints_with_saved_rng_states": checkpoint_count, "evaluation_invariance_checks": invariant_count,
              "rng_groups_each_distinct_across_three_training_seeds": len(rng_groups),
              "copied_files_verified_unchanged": copy_checks, "rerun_evaluation_rows": row_count,
              "copied_metric_jsons_verified_via_v1_archive": corrected_copy_checks,
              "all_evaluation_rows": len(all_rows), "frozen_source_files_verified": len(frozen["source_sha256"]),
              "executed_source_modules_verified": len(modules), "split_and_kernel_manifests_equal_original": True,
              "sphere_metric_version": metric["metric_version"], "sphere_rows_corrected": len(sphere_rows),
              "verification_script_sha256": sha(__file__), "new_training_seconds": manifest["new_train_seconds"]}
    write(DEST / "verification.json", result)
    write(DEST / "FINAL_TABLES_READY.json", {"all_primary_jobs_completed": True, "rng_correction_verified": True,
          "sphere_metric_version": metric["metric_version"], "metric_provenance_verified": True,
          "historical_controlled_bundle_untouched": True, "sample_arrays_unchanged_by_metric_correction": True})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
