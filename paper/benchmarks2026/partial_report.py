"""CPU-only snapshot of a user-paused experiment; never completes missing cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

import numpy as np

from . import plots

ROOT = plots.ROOT
BASE = plots.BASE
METHODS = plots.METHODS
LABELS = {"mgm": "MGM", "rwflow": "RW-Flow-C", "kgdm": "KGDM", "meanflow": "RMF-MT"}
SCORES = ("kernel_mmd2_v", "symmetric_nn", "radius_coverage", "one_nna", "constraint_mean")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["data"], row["method"]].append(row)
    result = {}
    for key, values in groups.items():
        seeds = sorted(row["seed"] for row in values)
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate training seeds: {key}")
        cell = {"data": key[0], "method": key[1], "seeds": seeds, "complete_three_seed": seeds == [0, 1, 2]}
        for score in SCORES:
            numbers = [row[score] for row in values]
            cell[score + "_mean"] = float(np.mean(numbers))
            cell[score + "_seed_sd"] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
        result[key] = cell
    return result


def independent_mmd(x, y, kind, h2):
    if kind == "torus":
        x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
        y = np.concatenate((np.cos(y), np.sin(y)), axis=1)

    def mean_kernel(a, b):
        total = 0.0
        for start in range(0, len(a), 128):
            d2 = ((a[start:start + 128, None] - b[None]) ** 2).sum(-1)
            total += float(np.exp(-d2 / h2).sum())
        return total / (len(a) * len(b))

    return max(0.0, mean_kernel(x, x) + mean_kernel(y, y) - 2 * mean_kernel(x, y))


def verify_completed(job, protocol):
    folder = ROOT / job["output"]
    done, meta = read(folder / "completed.json"), read(folder / "job.json")
    for key in ("data", "method", "seed", "steps"):
        assert done[key] == job[key], (folder, key)
    assert meta["config"] == job["config"] and done["test_evaluations"] == 3
    assert done["attempts"] == job["steps"] + (job["config"].get("critic_warmup", 500) if job["method"] == "mgm" else 0)
    expected_critic = (job["config"].get("critic_warmup", 500) + job["steps"] * job["config"].get("critic_steps", 5)) if job["method"] == "mgm" else 0
    assert done["critic_updates"] == expected_critic
    hashes = {PureWindowsPath(rel).as_posix(): value for rel, value in protocol["source_sha256"].items()}
    assert meta["executed_worker_sha256"] == hashes["paper/benchmarks2026/worker.py"]
    assert meta["method_source_sha256"] == hashes[f"paper/benchmarks2026/{job['method']}.py"]
    data = BASE / "data" / job["data"]
    manifest = read(data / "manifest.json")
    assert digest(data / "manifest.json") == meta["data_manifest_sha256"]
    assert digest(data / "data.npz") == manifest["arrays_sha256"]
    step = done["selected_step"]
    assert 0 <= step <= job["steps"]
    test_path = folder / f"test_step{step}_metrics.json"
    assert list(folder.glob("test_step*_metrics.json")) == [test_path]
    records = read(test_path)
    assert len(records) == 3
    with np.load(data / "data.npz", allow_pickle=False) as archive:
        target = archive["test_target"].astype(np.float64)
    errors = []
    with np.load(folder / f"test_step{step}_samples.npz", allow_pickle=False) as archive:
        for draw, row in enumerate(records):
            assert all(row[key] == job[key] for key in ("data", "method", "seed"))
            assert row["draw"] == draw and row["split"] == "test" and row["step"] == step
            assert row["model_forward_calls"] == 1
            x = archive[f"generated_draw_{draw}"].astype(np.float64)
            assert np.isfinite(x).all() and len(x) == len(target) == row["target_n"] == row["generated_n"]
            if manifest["kind"] == "sphere":
                assert np.max(np.abs(np.linalg.norm(x, axis=1) - 1)) < 1e-5
            else:
                assert np.all((x >= 0) & (x < 2 * np.pi))
            errors.append(abs(independent_mmd(x, target, manifest["kind"], manifest["reference"]["bandwidth2"]) - row["kernel_mmd2_v"]))
    assert max(errors) < 1e-10, (folder, errors)
    val = read(folder / f"val_step{step}_metrics.json")
    assert abs(np.mean([r["kernel_mmd2_v"] for r in val]) - done["selected_validation_mmd2"]) < 1e-12
    assert abs(np.mean([r["kernel_mmd2_v"] for r in records]) - done["test_mmd2"]) < 1e-12
    row = {key: job[key] for key in ("data", "method", "seed")}
    row.update(steps=done["steps"], selected_step=step, critic_updates=done["critic_updates"],
               target_n=len(target), train_seconds=done["train_seconds"])
    row.update({score: float(np.mean([r[score] for r in records])) for score in SCORES})
    return row, max(errors)


def validation_curves(status, out):
    records = []
    for item in status:
        if item["status"] == "not_started":
            continue
        folder = ROOT / item["output"]
        for path in folder.glob("val_step*_metrics.json"):
            rows = read(path)
            records.append({"data": item["data"], "method": item["method"], "seed": item["seed"],
                            "step": rows[0]["step"], "mmd2": float(np.mean([r["kernel_mmd2_v"] for r in rows])),
                            "status": item["status"]})
    write_csv(out / "validation_curves.csv", records)
    datasets = list(dict.fromkeys(row["data"] for row in records))
    fig, axes = plots.plt.subplots(1, len(datasets), figsize=(10 * len(datasets), 5), squeeze=False)
    for ax, data in zip(axes.flat, datasets):
        for item in status:
            if item["data"] != data or item["status"] == "not_started":
                continue
            rows = sorted((r for r in records if (r["data"], r["method"], r["seed"]) == (data, item["method"], item["seed"])), key=lambda r: r["step"])
            if not rows:
                continue
            label = f"{LABELS[item['method']]} s{item['seed']}" + (" [STOPPED]" if item["status"] == "interrupted" else "")
            ax.plot([r["step"] for r in rows], [1000 * r["mmd2"] for r in rows],
                    color=plots.COLORS[item["method"]], linestyle=("-", "--", ":")[item["seed"]], label=label)
        ax.set(title=data.title(), xlabel="Generator/field updates", ylabel="Validation kernel MMD2 x 1000")
        ax.set_yscale("symlog", linthresh=.1)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.suptitle("PAUSED EXPERIMENT | recorded validation only; unequal completed budgets", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(out / "paused_validation_curves.png", dpi=240)
    plots.plt.close(fig)


def earthquake_preview(status, out):
    seed_zero = {r["method"]: r for r in status if r["data"] == "earthquake" and r["seed"] == 0 and r["status"] != "not_started"}
    if not seed_zero:
        return
    with np.load(BASE / "data/earthquake/data.npz", allow_pickle=False) as archive:
        target = archive["val_target"]
    entries = [("Held-out validation", target, "Fixed held-out validation target")]
    for method in METHODS:
        item = seed_zero.get(method)
        if item is None:
            continue
        folder = ROOT / item["output"]
        progress = read(folder / ("completed.json" if item["status"] == "completed" else "progress.json"))
        best = progress["selected_step"] if item["status"] == "completed" else progress["best_step"]
        with np.load(folder / f"val_step{best}_samples.npz", allow_pickle=False) as archive:
            x = archive["generated_draw_0"]
        meta = f"selected step {best:,}; trained {item['saved_step']:,}/50,000"
        if item["status"] == "interrupted":
            meta += " | STOPPED"
        entries.append((LABELS[method], x, meta))
    densities = [plots.sphere_density(points) for _, points, _ in entries]
    vmax = max(float(d[2].max()) for d in densities)
    coast = plots.coast_segments(read(ROOT / "paper/vendor/ne_110m_land.geojson"))
    nrows = (len(entries) + 1) // 2
    fig = plots.plt.figure(figsize=(18, 3 + 4 * nrows))
    grid = fig.add_gridspec(nrows, 3, width_ratios=(1, 1, .025), left=.07, right=.93, top=.80, bottom=.10, hspace=.48, wspace=.20)
    fig.text(.07, .96, "EARTHQUAKE | PAUSED VALIDATION PREVIEW, NOT TEST", fontsize=18, weight="bold", color="#9e292b")
    fig.text(.07, .91, "Training seed 0, draw 0. Checkpoints chosen by validation; MGM has an unfinished training budget.", fontsize=11)
    for index, ((label, points, meta), density) in enumerate(zip(entries, densities)):
        ax = fig.add_subplot(grid[index // 2, index % 2], projection="mollweide")
        plots.render_panel(ax, density, vmax, True, (0, 1), coast)
        ax.set_title(f"{label}\n{meta}\nn={len(points):,}", fontsize=11, pad=12)
        ax.set_ylabel("Latitude (degrees)", labelpad=12)
    plots.density_colorbar(fig, fig.add_subplot(grid[:, 2]), vmax, True)
    fig.text(.07, .035, "Same KDE/color scale. No RW-Flow run yet. Unequal training budgets; this is not a completed benchmark.", fontsize=11)
    fig.savefig(out / "earthquake_paused_validation.png", dpi=240)
    plots.plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="final_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    phase, out = BASE / args.phase, args.output.resolve()
    if not out.is_relative_to(BASE.resolve()):
        parser.error("Report must stay inside outputs/comparison2026")
    out.mkdir(parents=True, exist_ok=False)
    protocol = read(phase / "protocol.json")
    assert read(phase / "runtime.json")["status"] == "paused_by_user"
    for rel, expected in protocol["source_sha256"].items():
        assert digest(phase / "source_snapshot" / rel) == expected, rel
    rows, statuses, errors = [], [], []
    for job in protocol["plan"]["jobs"]:
        folder = ROOT / job["output"]
        state = {key: job[key] for key in ("data", "method", "seed", "output")}
        state.update(planned_steps=job["steps"], saved_step=0, status="not_started", selected_step=None)
        if (folder / "completed.json").exists():
            row, error = verify_completed(job, protocol)
            rows.append(row)
            errors.append(error)
            state.update(status="completed", saved_step=row["steps"], selected_step=row["selected_step"])
        elif (folder / "progress.json").exists():
            progress = read(folder / "progress.json")
            assert (folder / "last.pt").exists()
            assert not list(folder.glob("test_step*_metrics.json")), "Interrupted job must not be promoted into test results"
            state.update(status="interrupted", saved_step=progress["step"], selected_step=progress["best_step"])
        statuses.append(state)
    counts = {name: sum(r["status"] == name for r in statuses) for name in ("completed", "interrupted", "not_started")}
    summary = summarize(rows)
    write_csv(out / "job_status.csv", statuses)
    write_csv(out / "completed_test_per_seed.csv", rows)
    write_csv(out / "completed_test_summary.csv", list(summary.values()))
    validation_curves(statuses, out)
    earthquake_preview(statuses, out)
    datasets = list(dict.fromkeys(j["data"] for j in protocol["plan"]["jobs"]))
    lines = ["# Paused Comparison: Intermediate Results", "", "User requested GPU release. Training is stopped; this report uses only saved CPU-readable artifacts.", "",
             f"Completed {counts['completed']}/{len(statuses)} main jobs; interrupted {counts['interrupted']}; not started {counts['not_started']}.",
             "No training, sampling or GPU timing is launched by this report. No missing seed or method is imputed.", "",
             "## Completed Test Scores", "", "Biased embedded Gaussian kernel MMD squared x1000; lower is better. Mean +/- sample SD across training seeds after averaging three draws per seed.",
             "Only cells with all three completed seeds receive an aggregate. Missing cells mean unavailable, not poor performance.", "",
             "| Dataset | MGM | RW-Flow-C | KGDM | RMF-MT |", "| --- | ---: | ---: | ---: | ---: |"]
    for data in datasets:
        cells = []
        for method in METHODS:
            cell = summary.get((data, method))
            if cell and cell["complete_three_seed"]:
                cells.append(f"{1000 * cell['kernel_mmd2_v_mean']:.3f} +/- {1000 * cell['kernel_mmd2_v_seed_sd']:.3f}")
            else:
                cells.append("not complete")
        lines.append("| " + " | ".join([data, *cells]) + " |")
    lines += ["", "Only Volcano has a complete four-method test comparison at this pause. MGM does not have the smallest observed mean there. This is not a statistical-significance claim; Volcano has only 82 test points.", "",
              "## Interrupted Jobs", "", "The table shows the latest durable progress/checkpoint, not the last in-flight optimizer update. Forced process termination can discard work after the last saved checkpoint.", "",
              "| Dataset | Method | Seed | Saved / planned updates | Validation-selected step |", "| --- | --- | ---: | ---: | ---: |"]
    for row in statuses:
        if row["status"] == "interrupted":
            lines.append(f"| {row['data']} | {LABELS[row['method']]} | {row['seed']} | {row['saved_step']:,} / {row['planned_steps']:,} | {row['selected_step']:,} |")
    lines += ["", "[Paused validation curves](paused_validation_curves.png) | [Earthquake validation preview](earthquake_paused_validation.png)", "",
              "Earthquake preview is validation-selected, single-seed, and partly trained. It is deliberately separate from completed test scores; it must not support a completed-test or convergence claim.", "",
              "## Completed SE(3) Pilot", "", "The separate two-object Mug/Table pilot finished all nine fixed 10,000-update jobs and three seeds. It is a within-object held-out-pose pilot, not full ACRONYM or unseen-object generalization.", "",
              "[SE(3) report and three-seed metrics](../grasp/report.md) | [Mug image](../grasp/publication_figures/seed0/mug_comparison.png) | [Table image](../grasp/publication_figures/seed0/table_comparison.png)", "",
              "MGM improves pose MMD over the one-step RMF adaptation but does not beat 50/100-step RFM. Substantial full-gripper collision remains unresolved; no physical grasp success was measured.", "",
              "## Scope and Verification", "", "Small-network controlled adaptations, not original-author model sizes or budgets. Common point generators use width128/depth3; RMF has a different time-network parameter count. MGM also performs extra critic updates. Optimizer work, tuning grids and compute are not matched.",
              "Saved historically exposed row splits are not a newly blinded or source-group-held-out benchmark. Calibration/refinement validation results are excluded from the test table.",
              "All completed jobs were checked on CPU: frozen source/data hashes, update/critic counts, selected validation/test arithmetic, three unique draws, one model forward, manifold constraints and independent NumPy kernel-MMD recomputation.",
              "GPU latency profiling was intentionally NOT run after the stop request. This is a partial-artifact audit, not the complete 108-job final audit.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "phase": args.phase,
                "status": "paused_by_user", "counts": counts, "planned": len(statuses),
                "saved_generator_updates": sum(r["saved_step"] for r in statuses),
                "planned_generator_updates": sum(r["planned_steps"] for r in statuses),
                "cpu_artifact_audit_passed": True, "completed_test_jobs_verified": len(rows),
                "max_independent_mmd_error": max(errors, default=0), "gpu_profiling": "not_run_gpu_released",
                "protocol_sha256": digest(phase / "protocol.json"), "script_sha256": digest(__file__),
                "job_status": statuses, "summary": list(summary.values())}
    write_json(out / "partial_audit_manifest.json", manifest)
    print(json.dumps({"output": str(out), **counts, "mmd_error": manifest["max_independent_mmd_error"]}))


if __name__ == "__main__":
    main()
