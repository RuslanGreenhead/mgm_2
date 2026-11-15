"""Versioned NPZ-only correction of sphere NN, coverage and training radii."""
import argparse
import hashlib
import json
import shutil
import zipfile

import numpy as np
import torch

import controlled_suite as s

BASE = s.ROOT / "outputs/paper_icml2026"
SPHERE = ("volcano", "earthquake", "fire", "flood")
VERSION = s.SPHERE_METRIC_VERSION


def radius(reference):
    x = torch.from_numpy(reference).double()
    directions = s.normalize_sphere_metric(x)
    d = s.pairwise(s.Sphere(), x, x)
    duplicate = (directions[:, None] == directions[None, :]).all(-1)
    d[duplicate] = float("inf")
    values = d.min(1).values
    values = values[torch.isfinite(values)]
    if not len(values):
        raise ValueError("No distinct reference directions available")
    return float(values.median()), int(torch.unique(directions, dim=0).shape[0])


def rescore(samples, target, threshold):
    d = s.pairwise(s.Sphere(), torch.from_numpy(samples), torch.from_numpy(target))
    a, b = d.min(1).values, d.min(0).values
    return {"nn_sample_to_data": float(a.mean()), "nn_data_to_sample": float(b.mean()),
            "symmetric_nn": float((a.mean() + b.mean()) / 2),
            "coverage": float((b <= threshold).double().mean()), "coverage_radius": threshold,
            "metric_version": VERSION}


def archived_bytes(archive, path, relative):
    if relative in archive.namelist():
        return archive.read(relative)
    raw = path.read_bytes()
    archive.writestr(relative, raw)
    return raw


def calibration():
    marker = BASE / "calibration_metric_provenance.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if old.get("metric_version") == VERSION:
            print("Sphere calibration already corrected", flush=True)
            return
    archive_path = BASE / "sphere_metric_v1_calibration.zip"
    source = BASE / "calibration_results.json"
    with zipfile.ZipFile(archive_path, "a", zipfile.ZIP_DEFLATED) as archive:
        raw = archived_bytes(archive, source, "calibration_results.json")
        archived_bytes(archive, BASE / "calibration_summary.json", "calibration_summary.json")
        rows = json.loads(raw)
        radii, input_hashes = {}, {}
        for name in SPHERE:
            target_path = BASE / "datasets" / name / "split_indices_and_targets.npz"
            sample_path = BASE / "calibration" / name / "samples.npz"
            targets, samples = np.load(target_path), np.load(sample_path)
            threshold, unique = radius(targets["kernel_reference_train"])
            radii[name] = {"corrected_radius": threshold, "distinct_reference_directions": unique}
            input_hashes[str(target_path)] = s.sha(target_path)
            input_hashes[str(sample_path)] = s.sha(sample_path)
            for row in rows:
                if row["data"] == name:
                    row.update(rescore(samples[f'generated_draw_{row["draw"]}'], targets[row["split"] + "_target"], threshold))
        s.write_json(source, rows)
        summary = []
        for name in s.REAL:
            for split in ("train", "val", "test"):
                rr = [r for r in rows if r["data"] == name and r["split"] == split]
                if not rr:
                    continue
                item = {"data": name, "method": "empirical_train_resampling", "split": split, "n_training_seeds": 0, "mc_draws": len(rr)}
                for key in ("mmd2_biased", "symmetric_nn", "coverage"):
                    values = [r[key] for r in rr]
                    item[key] = {"mean": float(np.mean(values)), "sd_across_mc_draws": float(np.std(values, ddof=1))}
                if name in SPHERE:
                    item["metric_version"] = VERSION
                    item["coverage_radius"] = radii[name]["corrected_radius"]
                summary.append(item)
        s.write_json(BASE / "calibration_summary.json", summary)
    assert all(s.sha(path) == expected for path, expected in input_hashes.items())
    s.write_json(marker, {"metric_version": VERSION, "corrected_at": s.utc(), "training_radii": radii,
                 "raw_archive": str(archive_path), "raw_results_sha256": hashlib.sha256(raw).hexdigest(),
                 "corrected_results_sha256": s.sha(source), "source_npz_sha256": input_hashes,
                 "arrays_unchanged": True, "mmd_unchanged": True, "script_sha256": s.sha(s.Path(__file__)),
                 "metric_helper_sha256": s.sha(s.Path(s.__file__))})
    print(json.dumps({"calibration_sphere_rows_corrected": 24, "radii": radii}), flush=True)


def independent(root):
    complete = json.loads((root / "run_manifest.json").read_text())
    if not complete.get("all_completed"):
        raise ValueError("Wait until all 54 independent reruns complete")
    marker = root / "metric_provenance.json"
    if marker.exists() and json.loads(marker.read_text()).get("metric_version") == VERSION:
        print("Independent sphere evaluations already corrected", flush=True)
        return
    archive_path = root / "sphere_metric_v1_raw.zip"
    files, arrays, changes, rows_count = [], {}, {}, 0
    with zipfile.ZipFile(archive_path, "a", zipfile.ZIP_DEFLATED) as archive:
        for name in ("results.json", "summary.json"):
            archived_bytes(archive, root / name, name)
        targets = {}
        for name in SPHERE:
            target_path = root / "datasets" / name / "split_indices_and_targets.npz"
            manifest_path = target_path.with_name("manifest.json")
            raw_manifest = archived_bytes(archive, manifest_path, str(manifest_path.relative_to(root)))
            metadata = json.loads(raw_manifest)
            targets[name] = np.load(target_path)
            new_radius, unique = radius(targets[name]["kernel_reference_train"])
            changes[name] = {"old_radius": metadata["coverage_radius"], "corrected_radius": new_radius,
                             "distinct_reference_directions": unique, "reference_n": len(targets[name]["kernel_reference_train"])}
            metadata.update(coverage_radius=new_radius, coverage_metric_version=VERSION)
            s.write_json(manifest_path, metadata)
            arrays[str(target_path)] = s.sha(target_path)
        for path in (root / "controlled").glob("*/*/seed*/evaluation_step*.json"):
            name = path.parents[2].name
            if name not in SPHERE:
                continue
            relative = str(path.relative_to(root))
            raw = archived_bytes(archive, path, relative)
            rows = json.loads(raw)
            for row in rows:
                sample_path = path.parent / f'samples_step{row["step"]}_nfe{row["sampler_nfe"]}.npz'
                arrays[str(sample_path)] = s.sha(sample_path)
                with np.load(sample_path) as samples:
                    x = samples[f'generated_draw_{row["draw"]}']
                row.update(rescore(x, targets[name][row["split"] + "_target"], changes[name]["corrected_radius"]))
                rows_count += 1
            s.write_json(path, rows)
            files.append({"path": relative, "raw_json_sha256": hashlib.sha256(raw).hexdigest(),
                          "corrected_json_sha256": s.sha(path), "archive_member": relative})
    assert all(s.sha(path) == expected for path, expected in arrays.items())
    previous_output = s.OUT
    s.OUT = root
    s.summarize()
    s.OUT = previous_output
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    for group in summary:
        if group["data"] in SPHERE:
            group.update(metric_version=VERSION, coverage_radius=changes[group["data"]]["corrected_radius"])
    s.write_json(summary_path, summary)
    for name in ("calibration_results.json", "calibration_summary.json", "calibration_metric_provenance.json"):
        shutil.copyfile(BASE / name, root / name)
    metric_snapshot = root / "metric_source_snapshot"
    metric_snapshot.mkdir(exist_ok=True)
    metric_hashes = {}
    for path in (s.Path(__file__), s.Path(s.__file__), s.Path(__file__).with_name("test_controlled_suite.py")):
        shutil.copyfile(path, metric_snapshot / path.name)
        metric_hashes[path.name] = s.sha(path)
    s.write_json(marker, {"metric_version": VERSION, "corrected_at": s.utc(), "corrected_rows": rows_count,
                 "corrected_files": files, "training_radius_changes": changes, "source_npz_sha256": arrays,
                 "raw_archive": str(archive_path), "arrays_unchanged": True, "mmd_unchanged": True,
                 "definition": "Normalize float64 vectors; angle=2 atan2(norm(xhat-yhat),norm(xhat+yhat)); exclude identical normalized directions explicitly for reference NN",
                 "script_sha256": s.sha(s.Path(__file__)), "training_rerun_required": False})
    provenance = json.loads(marker.read_text())
    provenance["metric_source_sha256"] = metric_hashes
    s.write_json(marker, provenance)
    print(json.dumps({"sphere_rows_corrected": rows_count, "files": len(files), "radii": changes}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=s.Path, default=BASE / "controlled_independent")
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    calibration()
    if not args.calibration_only:
        independent(args.root.resolve())


if __name__ == "__main__":
    main()
