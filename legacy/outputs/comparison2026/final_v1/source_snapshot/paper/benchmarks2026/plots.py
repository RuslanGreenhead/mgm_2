"""CPU-only, completed-run scientific figures for the 2026 comparison.

No model, checkpoint, training, torch, or legacy result-builder is imported.
See plot_notes.md for split guards, selection, KDEs, and aggregation semantics.
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/comparison2026"
DATASETS = ("volcano", "earthquake", "fire", "flood", "general", "glycine", "proline", "prepro", "rna")
SPHERE = frozenset(DATASETS[:4])
LABELS = {name: name.title() for name in DATASETS}
LABELS.update(prepro="Pre-Pro", rna="RNA (7D)")
METHODS = ("mgm", "rwflow", "kgdm", "meanflow")
METHOD_LABELS = dict(zip(METHODS, ("MGM", "RWFlow", "KGDM", "MeanFlow")))
COLORS = dict(zip(METHODS, ("#087f8c", "#c4475e", "#5257a6", "#a77810")))
METRICS = (
    "kernel_mmd2_v", "symmetric_nn", "minimum_matching_distance", "radius_coverage",
    "nn_occupancy_coverage", "one_nna", "constraint_mean", "train_seconds",
)
EXPECTED_SEEDS = (0, 1, 2)
RNA_PAIRS = tuple((i, i + 1) for i in range(6))
KAPPA = 45.0
TORUS_BINS = 120
TORUS_SIGMA = 2.4
METRIC_LABELS = {
    "kernel_mmd2_v": "Biased kernel MMD2_V (primary)",
    "minimum_matching_distance": "MinimumMatchingDistance (directed NN)",
    "symmetric_nn": "Symmetric NN distance", "radius_coverage": "Radius coverage",
    "nn_occupancy_coverage": "NN occupancy coverage", "one_nna": "1-NNA (ideal near 0.5)",
    "constraint_mean": "Mean constraint residual", "train_seconds": "Recorded training seconds",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
    "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "axes.titleweight": "normal",
})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_final_phase(path):
    return re.match(r"^final(?:_|$)", Path(path).name, re.IGNORECASE) is not None


class Inputs:
    def __init__(self, split):
        self.split = split
        self.files = {}
        self.archives = []
        self.warnings = []

    def read(self, path):
        path = Path(path).resolve()
        if self.split == "val" and path.name.startswith("test_step"):
            raise ValueError("Validation plotting cannot read test metrics")
        raw = path.read_bytes()
        self.files[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw.decode("utf-8-sig"))

    def array(self, path, key, *, held_out=False):
        if held_out and key != f"{self.split}_target":
            raise ValueError("Only the authorized held-out split may be loaded")
        if self.split == "val" and (key == "test_target" or Path(path).name.startswith("test_step")):
            raise ValueError("Validation plotting cannot read test arrays")
        # NPZ entries are decompressed lazily. In validation mode the test member
        # is never indexed, extracted, or hashed, even in the shared data archive.
        with np.load(path, allow_pickle=False) as archive:
            result = np.array(archive[key], dtype=np.float64, copy=True)
        self.archives.append({
            "path": str(Path(path).resolve()), "key": key, "shape": list(result.shape),
            "array_sha256_float64": hashlib.sha256(result.tobytes()).hexdigest(),
        })
        if result.ndim != 2 or not len(result) or not np.isfinite(result).all():
            raise ValueError(f"Invalid point array: {path}:{key}")
        return result


@dataclass
class Run:
    folder: Path
    phase: Path
    job: dict
    done: dict

    @property
    def key(self):
        return self.job["data"], self.job["method"], int(self.job["seed"])

    @property
    def selected_step(self):
        return int(self.done["selected_step"])

    @property
    def config_key(self):
        return json.dumps(self.job["config"], sort_keys=True, separators=(",", ":"))

    @property
    def variant(self):
        return self.job["config"].get("variant", self.job.get("resolved_config", {}).get("variant")) or "unspecified"

    @property
    def config_id(self):
        identity = {key: self.job.get(key) for key in (
            "method", "config", "resolved_config", "method_source_sha256", "executed_worker_sha256", "steps",
        )}
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()

    def metadata(self):
        return dict(phase=self.phase.name, variant=self.variant, config_id=self.config_id,
                    recipe_id=self.folder.name, config=self.config_key,
                    resolved_config=json.dumps(self.job.get("resolved_config"), sort_keys=True),
                    method_source_sha256=self.job.get("method_source_sha256", ""),
                    executed_worker_sha256=self.job.get("executed_worker_sha256", ""),
                    run=str(self.folder))


def choose_runs(candidates, split):
    grouped = defaultdict(list)
    for run in candidates:
        grouped[run.key].append(run)
    chosen = {}
    for key, group in grouped.items():
        if split == "test" and len(group) != 1:
            raise ValueError(f"Final test has duplicate jobs for {key}; freeze one config, do not select by test")
        if split == "val":
            group = sorted(group, key=lambda r: (float(r.done["selected_validation_mmd2"]), str(r.folder)))
        chosen[key] = group[0]
    if split == "test":
        configs = defaultdict(set)
        for run in chosen.values():
            configs[run.key[:2]].add(run.config_id)
        for key, values in configs.items():
            if len(values) != 1:
                raise ValueError(f"Final training seeds do not share a frozen config: {key}")
    return chosen


def discover(phases, split, datasets, inputs):
    candidates = []
    for phase in phases:
        if split == "test" and not is_final_phase(phase):
            raise ValueError(f"Test input must be a final_* phase, not {phase.name}")
        for completed in sorted(phase.rglob("completed.json")):
            if "source_snapshot" in completed.relative_to(phase).parts:
                continue
            folder = completed.parent
            jobfile = folder / "job.json"
            if not jobfile.is_file():
                continue
            job = inputs.read(jobfile)
            if job.get("data") not in datasets or job.get("method") not in METHODS:
                continue
            if int(job.get("seed", -1)) not in EXPECTED_SEEDS:
                inputs.warnings.append(f"Ignored non-primary seed: {folder}")
                continue
            if not is_final_phase(phase) and job.get("score_test") is True:
                inputs.warnings.append(f"Ignored non-final test-enabled job: {folder}")
                continue
            if split == "test" and job.get("score_test") is not True:
                inputs.warnings.append(f"Ignored test-disabled completed job: {folder}")
                continue
            done = inputs.read(completed)
            for field in ("data", "method", "seed"):
                if done.get(field) != job.get(field):
                    raise ValueError(f"Job/completion mismatch for {field}: {folder}")
            if int(done["steps"]) != int(job["steps"]):
                raise ValueError(f"Completion does not reach declared budget: {folder}")
            selected = int(done["selected_step"])
            if selected < 0 or selected > int(done["steps"]):
                raise ValueError(f"Invalid selected checkpoint: {folder}")
            if not math.isfinite(float(done["selected_validation_mmd2"])):
                raise ValueError(f"Nonfinite validation selection score: {folder}")
            if split == "test" and (int(done.get("test_evaluations", 0)) < 1 or done.get("test_mmd2") is None):
                inputs.warnings.append(f"No completed test evaluation: {folder}")
                continue
            candidates.append(Run(folder, phase, job, done))
    return choose_runs(candidates, split), candidates


def metric_rows(run, split, inputs, step=None):
    if split == "test":
        if (not is_final_phase(run.phase) or run.job.get("score_test") is not True
                or run.done.get("steps") != run.job.get("steps")
                or int(run.done.get("test_evaluations", 0)) < 1
                or run.done.get("test_mmd2") is None):
            raise ValueError("Test metrics require a completed, test-enabled final job")
        if step is not None and int(step) != run.selected_step:
            raise ValueError("Only the validation-selected test checkpoint is authorized")
    step = run.selected_step if step is None else int(step)
    path = run.folder / f"{split}_step{step}_metrics.json"
    rows = inputs.read(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected nonempty metric records: {path}")
    draws = []
    for row in rows:
        expected = dict(data=run.key[0], method=run.key[1], seed=run.key[2], split=split, step=step)
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Metric identity mismatch: {path}")
        if row.get("model_forward_calls") != 1:
            raise ValueError(f"Expected exactly one model forward call: {path}")
        if "variant" in row and row["variant"] != run.job["config"].get("variant"):
            raise ValueError(f"Metric variant mismatch: {path}")
        if "recipe_id" in row and row["recipe_id"] != run.folder.name:
            raise ValueError(f"Metric recipe mismatch: {path}")
        draws.append(int(row["draw"]))
        for key in METRICS:
            if key in row and not math.isfinite(float(row[key])):
                raise ValueError(f"Nonfinite {key}: {path}")
    if sorted(draws) != list(range(len(rows))):
        raise ValueError(f"Invalid draw identifiers: {path}")
    expected_draws = run.done["test_evaluations"] if split == "test" else run.job.get("validation_draws", len(rows))
    if len(rows) != int(expected_draws):
        raise ValueError(f"Draw count does not match job/completion: {path}")
    if step == run.selected_step:
        expected_score = run.done["test_mmd2"] if split == "test" else run.done["selected_validation_mmd2"]
        if not np.isclose(np.mean([r["kernel_mmd2_v"] for r in rows]), expected_score, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Selected metric does not match completion: {path}")
    return rows


def summarize_seed(run, rows, split):
    result = dict(data=run.key[0], method=run.key[1], seed=run.key[2], split=split,
                  selected_step=run.selected_step, budget=int(run.done["steps"]),
                  n_draws=len(rows), completed_train_seconds=run.done.get("train_seconds"), **run.metadata())
    for metric in METRICS:
        if all(metric in row for row in rows):
            values = [float(row[metric]) for row in rows]
            result[metric] = float(np.mean(values))
            result[metric + "_draw_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return result


def seed_group(row):
    return row["data"], row["method"], row["split"], row["phase"], row["config_id"]


def aggregate_seeds(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[seed_group(row)].append(row)
    result = []
    for (data, method, split, phase, config_id), group in sorted(groups.items()):
        seeds = sorted(row["seed"] for row in group)
        if len(seeds) != len(set(seeds)):
            raise ValueError("One seed must contribute exactly once to aggregation")
        entry = dict(data=data, method=method, split=split, phase=phase, config_id=config_id,
                     variant=group[0]["variant"], config=group[0]["config"],
                     resolved_config=group[0]["resolved_config"], n_seeds=len(seeds),
                     seeds=",".join(map(str, seeds)), missing_seeds=",".join(str(s) for s in EXPECTED_SEEDS if s not in seeds),
                     all_three_seeds=seeds == list(EXPECTED_SEEDS))
        for key in METRICS:
            values = [row[key] for row in group if key in row]
            if len(values) == len(group):
                entry[key + "_mean"] = float(np.mean(values))
                entry[key + "_training_seed_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        result.append(entry)
    return result


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compact(value):
    if value is None:
        return "--"
    return f"{value:.5g}"


def tables(out, rows, aggregate, datasets, split, *, candidates=False):
    prefix = "calibration_candidates_" if candidates else ""
    write_csv(out / f"{prefix}per_seed_metrics.csv", rows)
    write_csv(out / f"{prefix}seed_mean_metrics.csv", aggregate)
    per_seed = {(*seed_group(r), r["seed"]): r for r in rows}
    heading = "VALIDATION ONLY: non-final tuning preview" if split == "val" else "TEST: frozen configurations, validation-selected checkpoints"
    intro = [f"# {heading}", "", "Draws are averaged within each training seed first. Parentheses in seed cells are within-seed draw SD; the last column is training-seed sample SD (both ddof=1).",
             "Only identical phase/config/source/budget identities are aggregated. Different variants or seed-wise selected configurations are never pooled.",
             "Missing seeds remain missing; one seed has no estimable training-seed SD, and one draw has no estimable draw SD. No convergence or significance claim is made.",
             "The primary biased kernel MMD2_V is NOT MinimumMatchingDistance (a separately reported directed nearest-neighbor distance).", ""]
    if split == "val":
        policy = "All completed candidates are retained below; selected_for_comparison marks the density-preview choices." if candidates else "Candidates are selected by minimum completed selected_validation_mmd2 per dataset/method/seed."
        intro += [policy + " These are post-selection validation previews, not test results.", ""]
    summary = intro + ["| Dataset | Method | Variant | Phase | Config ID | Seeds | Missing seeds | Biased kernel MMD2_V mean | Training-seed SD |",
                       "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    table_dir = out / ("calibration_tables" if candidates else "tables")
    table_dir.mkdir()
    for data in datasets:
        lines = intro + [f"## {LABELS[data]}", "", "| Method | Variant | Config ID | Metric | Seed 0 mean (draw SD) | Seed 1 mean (draw SD) | Seed 2 mean (draw SD) | Seed mean | Training-seed SD |",
                         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for method in METHODS:
            stats = [r for r in aggregate if (r["data"], r["method"]) == (data, method)] or [{}]
            for stat in stats:
                identity = [METHOD_LABELS[method], stat.get("variant", "--"), stat.get("config_id", "--")[:12]]
                summary.append("| " + " | ".join([LABELS[data], *identity[:2], stat.get("phase", "--"), identity[2],
                               stat.get("seeds", "--"), stat.get("missing_seeds", "0,1,2") or "none",
                               compact(stat.get("kernel_mmd2_v_mean")), compact(stat.get("kernel_mmd2_v_training_seed_sd"))]) + " |")
                for metric in METRICS:
                    values = []
                    for seed in EXPECTED_SEEDS:
                        row = per_seed.get((*seed_group(stat), seed), {}) if stat else {}
                        value = compact(row.get(metric))
                        values.append(value if value == "--" else f"{value} ({compact(row.get(metric + '_draw_sd'))})")
                    lines.append("| " + " | ".join([*identity, METRIC_LABELS[metric], *values,
                                                   compact(stat.get(metric + "_mean")), compact(stat.get(metric + "_training_seed_sd"))]) + " |")
        lines += ["", "## Configuration Ledger", ""]
        for stat in (r for r in aggregate if r["data"] == data):
            lines += [f"### {METHOD_LABELS[stat['method']]} / {stat['variant']} / {stat['phase']} / {stat['config_id'][:12]}",
                      "", "```json", json.dumps(dict(config_id=stat["config_id"], config=json.loads(stat["config"]),
                                                     resolved_config=json.loads(stat["resolved_config"])), indent=2), "```", ""]
        (table_dir / f"{data}_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / f"{prefix}seed_mean_metrics.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def sphere_density(points):
    norm = np.linalg.norm(points, axis=1, keepdims=True)
    if points.shape[1] != 3 or np.any(norm <= 0):
        raise ValueError("Sphere plots require nonzero XYZ points")
    points = points / norm
    longitude_edges = np.linspace(-np.pi, np.pi, 242)
    latitude_edges = np.linspace(-np.pi / 2, np.pi / 2, 122)
    lon = (longitude_edges[:-1] + longitude_edges[1:]) / 2
    lat = (latitude_edges[:-1] + latitude_edges[1:]) / 2
    lo, la = np.meshgrid(lon, lat)
    grid = np.stack((np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)), -1).reshape(-1, 3)
    density = np.zeros(len(grid), dtype=np.float64)
    for start in range(0, len(points), 256):
        cosine = np.clip(grid @ points[start:start + 256].T, -1, 1)
        density += np.exp(KAPPA * (cosine - 1)).sum(1)
    density *= KAPPA / (2 * np.pi * (1 - np.exp(-2 * KAPPA)) * len(points))
    return lo, la, density.reshape(lo.shape)


def torus_density(points, pair=(0, 1)):
    xy = (points[:, pair] + np.pi) % (2 * np.pi) - np.pi
    hist, edges_x, edges_y = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=TORUS_BINS, range=((-np.pi, np.pi), (-np.pi, np.pi)),
    )
    density = gaussian_filter(hist, sigma=TORUS_SIGMA, mode="wrap")
    density /= len(points) * (2 * np.pi / TORUS_BINS)**2
    centers = (edges_x[:-1] + edges_x[1:]) / 2
    centers_y = (edges_y[:-1] + edges_y[1:]) / 2
    # Periodic padding reaches both +/-pi borders without a blank seam.
    axis_x = np.r_[centers[-1] - 2 * np.pi, centers, centers[0] + 2 * np.pi]
    axis_y = np.r_[centers_y[-1] - 2 * np.pi, centers_y, centers_y[0] + 2 * np.pi]
    return *np.meshgrid(np.rad2deg(axis_x), np.rad2deg(axis_y)), np.pad(density.T, 1, mode="wrap")


def coast_segments(geojson):
    segments = []
    for feature in geojson["features"]:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygons:
            line = np.asarray(polygon[0], dtype=np.float64)
            cuts = np.flatnonzero(np.abs(np.diff(line[:, 0])) > 180) + 1
            for part in np.split(line, cuts):
                if len(part) > 1:
                    segments.append(np.deg2rad(part))
    return segments


def render_panel(ax, density, vmax, sphere, pair, coastline):
    norm = LogNorm(vmin=vmax * 1e-3, vmax=vmax)
    levels = np.geomspace(norm.vmin, norm.vmax, 40)
    xx, yy, zz = density
    if sphere:
        # QuadMesh projects vertices directly. Mollweide contour paths otherwise
        # receive excessive interpolation in the vector PDF backend.
        edges_x = np.linspace(-np.pi, np.pi, zz.shape[1] + 1)
        edges_y = np.linspace(-np.pi / 2, np.pi / 2, zz.shape[0] + 1)
        artist = ax.pcolormesh(edges_x, edges_y, np.clip(zz, norm.vmin, norm.vmax),
                              cmap="YlGnBu", norm=norm, shading="flat",
                              rasterized=False, antialiased=False,
                              edgecolors="face", linewidth=0.12)
    else:
        artist = ax.contourf(xx, yy, np.clip(zz, norm.vmin, norm.vmax), levels=levels,
                             cmap="YlGnBu", norm=norm, antialiased=False)
    if sphere:
        ax.add_collection(LineCollection(coastline, colors="#666c73", linewidths=0.35,
                                         alpha=0.8, transform=ax.transData))
        ax.set_xticks(np.deg2rad((-120, -60, 0, 60, 120)))
        ax.set_yticks(np.deg2rad((-60, -30, 0, 30, 60)))
        ax.grid(alpha=0.2, linewidth=0.4)
        ax.set_xlabel("Longitude (degrees)", fontsize=9)
    else:
        ax.set(xlim=(-180, 180), ylim=(-180, 180), xticks=(-180, -90, 0, 90, 180), yticks=(-180, -90, 0, 90, 180))
        ax.set_aspect("equal")
        ax.set_xlabel(f"Angle {pair[0] + 1} (degrees)", fontsize=9)
        ax.grid(alpha=0.15, linewidth=0.4)
    ax.tick_params(labelsize=8, length=2)
    return artist


def save_figure(fig, path, dpi, files, *, min_width=4000):
    png, pdf = path.with_suffix(".png"), path.with_suffix(".pdf")
    if round(fig.get_figwidth() * dpi) < min_width:
        raise ValueError(f"Figure PNG must be at least {min_width} pixels wide")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)  # Contour paths, coastlines, ticks, and text remain vector.
    plt.close(fig)
    files.extend((str(png), str(pdf)))


def density_colorbar(fig, cax, vmax, sphere):
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(vmax * 1e-3, vmax), cmap="YlGnBu"), cax=cax)
    if bar.solids is not None:
        bar.solids.set_rasterized(False)
        bar.solids.set_edgecolor("face")
    bar.ax.tick_params(labelsize=8, length=2)
    bar.set_label("Density / sr" if sphere else "Density / rad$^2$", fontsize=9)
    return bar


def individual_panels(row_specs, panels, densities, selected, split, coast, out, dpi, files):
    folder = out / "panels"
    folder.mkdir(exist_ok=True)
    records = []
    for data, pair in row_specs:
        sphere = data in SPHERE
        available = densities[(data, pair)]
        vmax = max(float(value[2].max()) for value in available.values())
        for method in (None, *METHODS):
            if method not in available:
                continue
            fig = plt.figure(figsize=(12, 7.2 if sphere else 11.4))
            grid = fig.add_gridspec(1, 2, width_ratios=(1, .035), left=.12, right=.89,
                                    top=.78 if sphere else .82, bottom=.17, wspace=.12)
            ax = fig.add_subplot(grid[0, 0], projection="mollweide" if sphere else None)
            label = f"Held-out {split} target" if method is None else METHOD_LABELS[method]
            pair_label = f" | angles {pair[0] + 1}-{pair[1] + 1}" if data == "rna" else ""
            stage = "VALIDATION ONLY" if split == "val" else "TEST"
            fig.text(.09, .96, f"{stage} | {LABELS[data]}{pair_label} | {label}", fontsize=17,
                     va="top", weight="bold", color="#9e292b" if split == "val" else "#20272b")
            if method is None:
                meta = f"Fixed held-out {split} target | n={len(panels[data][None]):,}"
                identity = "Same smoothing and density limits as all method panels in this comparison."
            else:
                run = selected[(data, method, 0)]
                meta = f"Training seed 0 | draw 0 | step {run.selected_step:,} / {int(run.done['steps']):,} | n={len(panels[data][method]):,}"
                identity = f"Variant: {run.variant} | config {run.config_id[:12]} | phase {run.phase.name}"
            fig.text(.09, .90, meta, fontsize=11, va="top")
            fig.text(.09, .86, identity, fontsize=10, va="top")
            render_panel(ax, available[method], vmax, sphere, pair, coast)
            ax.set_ylabel("Latitude (degrees)" if sphere else f"Angle {pair[1] + 1} (degrees)", fontsize=11, labelpad=12)
            ax.tick_params(labelsize=10)
            ax.xaxis.label.set_size(11)
            density_colorbar(fig, fig.add_subplot(grid[0, 1]), vmax, sphere)
            caption = "Sample KDE, not model likelihood. Identical row-wise log density limits to the comparison."
            if data == "rna":
                caption += "\nPair marginal only, not full-joint verification."
            if split == "val":
                caption += "\nValidation preview, not a test result or a convergence claim."
            fig.text(.09, .045, caption, fontsize=10, va="bottom", linespacing=1.5)
            suffix = f"_pair{pair[0] + 1}_{pair[1] + 1}" if data == "rna" else ""
            name = f"{data}{suffix}_{method or 'target'}"
            path = folder / name
            save_figure(fig, path, dpi, files, min_width=2000)
            records.append(dict(data=data, pair=list(pair) if data == "rna" else None,
                                method=method or "target", vmin=vmax * 1e-3, vmax=vmax,
                                png=str(path.with_suffix(".png")), pdf=str(path.with_suffix(".pdf"))))
    return records


def panel_index(out, records, datasets, split):
    lines = [f"# {'VALIDATION ONLY' if split == 'val' else 'TEST'}: individual density panels", "",
             "Each PNG is at least 2000 px wide; each PDF is vector. Density arrays and color limits exactly match the corresponding comparison row.",
             "Missing methods are not imputed or exported. RNA exports contain six separate pair marginals per available method/target.", ""]
    for data in datasets:
        lines += [f"## {LABELS[data]}", "", f"[Dataset comparison PNG](<{(out / (data + '_comparison.png')).as_posix()}>) | [Vector PDF](<{(out / (data + '_comparison.pdf')).as_posix()}>)",
                  "", "| Panel | Angle pair | PNG zoom | Vector PDF |", "| --- | --- | --- | --- |"]
        for row in (r for r in records if r["data"] == data):
            label = "Held-out target" if row["method"] == "target" else METHOD_LABELS[row["method"]]
            pair = "--" if row["pair"] is None else "-".join(str(i + 1) for i in row["pair"])
            lines.append(f"| {label} | {pair} | [PNG](<{Path(row['png']).as_posix()}>) | [PDF](<{Path(row['pdf']).as_posix()}>) |")
        lines.append("")
    (out / "panel_index.md").write_text("\n".join(lines), encoding="utf-8")


def comparison_figure(row_specs, panels, densities, selected, split, coast, path, dpi, files, scales):
    nrows = len(row_specs)
    height = 2.5 + 3.0 * nrows
    fig = plt.figure(figsize=(20, height))
    grid = fig.add_gridspec(nrows, 6, width_ratios=(1, 1, 1, 1, 1, 0.045),
                            left=0.075, right=0.95, top=1 - 1.4 / height,
                            bottom=0.95 / height, hspace=0.53, wspace=0.15)
    title = ("VALIDATION ONLY - non-final tuning preview" if split == "val"
             else "TEST - validation-selected checkpoints")
    fig.text(0.075, 1 - 0.16 / height, title, va="top", fontsize=17,
             color="#9e292b" if split == "val" else "#20272b", weight="bold")
    fig.text(0.075, 1 - 0.49 / height, "Training seed 0, generated draw 0. Identical KDE and log color scale within each row. Missing runs are not imputed.",
             va="top", fontsize=10)
    for i, (data, pair) in enumerate(row_specs):
        sphere = data in SPHERE
        key = (data, pair)
        available = densities[key]
        vmax = max(float(value[2].max()) for value in available.values())
        scales[f"{data}:{pair}"] = {"vmin": vmax * 1e-3, "vmax": vmax,
                                         "unit": "per_steradian" if sphere else "per_radian_squared"}
        for j, method in enumerate((None, *METHODS)):
            ax = fig.add_subplot(grid[i, j], projection="mollweide" if sphere else None)
            label = f"Held-out {split}" if method is None else METHOD_LABELS[method]
            run = selected.get((data, method, 0)) if method else None
            if method not in available:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "No completed seed-0 run", ha="center", va="center", transform=ax.transAxes, color="#71767c", fontsize=10)
                ax.set_title(label, fontsize=12)
                continue
            render_panel(ax, available[method], vmax, sphere, pair, coast)
            meta = f"n={len(panels[data][method]):,}" if method is None else f"step {run.selected_step:,} / {int(run.done['steps']):,} | n={len(panels[data][method]):,}"
            identity = "" if method is None else f"\n{run.variant} | cfg {run.config_id[:8]}"
            ax.set_title(label + "\n" + meta + identity, fontsize=10, pad=9)
            if j == 0:
                ylabel = LABELS[data] + ("\nLatitude (degrees)" if sphere else f"\nAngle {pair[1] + 1} (degrees)")
                ax.set_ylabel(ylabel, fontsize=11, labelpad=15 if sphere else 7)
            elif not sphere:
                ax.set_yticklabels([])
        density_colorbar(fig, fig.add_subplot(grid[i, 5]), vmax, sphere)
    footer = "Sample KDEs, not model likelihoods. RNA panels are pair marginals, not full-joint verification. No convergence claim."
    if split == "val":
        footer = "VALIDATION, NOT TEST. Candidate choice uses recorded validation scores only. " + footer
    fig.text(0.075, 0.15 / height, footer, fontsize=9, va="bottom")
    save_figure(fig, path, dpi, files)


def curves(out, selected, datasets, inputs, dpi, files):
    records, training = [], []
    for run in selected.values():
        for path in sorted(run.folder.glob("val_step*_metrics.json")):
            match = re.fullmatch(r"val_step(\d+)_metrics\.json", path.name)
            if not match or int(match.group(1)) > int(run.done["steps"]):
                continue
            rows = metric_rows(run, "val", inputs, int(match.group(1)))
            values = [r["kernel_mmd2_v"] for r in rows]
            records.append(dict(data=run.key[0], method=run.key[1], seed=run.key[2],
                                step=int(match.group(1)), validation_mmd2=float(np.mean(values)), n_draws=len(values),
                                validation_mmd2_draw_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                **run.metadata()))
        loss_file = run.folder / "curve.json"
        if loss_file.is_file():
            for row in inputs.read(loss_file):
                # MGM warmup uses the same log key for critic and generator losses.
                if ("loss" in row and row.get("generator_updated", 1) != 0
                        and 0 <= int(row["step"]) <= int(run.done["steps"]) and math.isfinite(float(row["loss"]))):
                    training.append(dict(data=run.key[0], method=run.key[1], seed=run.key[2],
                                         step=int(row["step"]), loss=float(row["loss"]), **run.metadata()))
    write_csv(out / "validation_learning_curves.csv", records)
    write_csv(out / "training_loss_curves.csv", training)
    if records:
        ncols = 3
        nrows = math.ceil(len(datasets) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 1.0 + nrows * 3.8), squeeze=False)
        for ax, data in zip(axes.flat, datasets):
            for method in METHODS:
                groups = defaultdict(list)
                for row in records:
                    if row["data"] == data and row["method"] == method:
                        groups[(row["seed"], row["run"])].append(row)
                for (seed, _), group in sorted(groups.items()):
                    group.sort(key=lambda row: row["step"])
                    ax.plot([r["step"] for r in group], [r["validation_mmd2"] for r in group],
                            color=COLORS[method], marker="o", markersize=3, linewidth=1.4,
                            linestyle=("-", "--", ":")[seed], alpha=0.85,
                            label=f"{METHOD_LABELS[method]} s{seed} [{group[0]['config_id'][:6]}]")
            ax.set(title=LABELS[data], xlabel="Generator updates", ylabel="Validation biased kernel MMD$^2_V$")
            ax.set_yscale("symlog", linthresh=1e-5)
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend(fontsize=8, frameon=False, ncol=2)
        for ax in list(axes.flat)[len(datasets):]:
            ax.set_axis_off()
        fig.suptitle("VALIDATION learning curves - completed jobs only; not test or a convergence certificate", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save_figure(fig, out / "validation_learning_curves", dpi, files)
    if training:
        fig, axes = plt.subplots(len(datasets), len(METHODS), figsize=(20, 1.0 + 2.8 * len(datasets)), squeeze=False)
        for i, data in enumerate(datasets):
            for j, method in enumerate(METHODS):
                ax = axes[i, j]
                for seed in EXPECTED_SEEDS:
                    group = sorted((r for r in training if (r["data"], r["method"], r["seed"]) == (data, method, seed)), key=lambda r: r["step"])
                    if group:
                        ax.plot([r["step"] for r in group], [r["loss"] for r in group], color=COLORS[method],
                                linestyle=("-", "--", ":")[seed], alpha=0.8, lw=1, label=f"seed {seed} [{group[0]['config_id'][:6]}]")
                ax.set_title(f"{LABELS[data]} | {METHOD_LABELS[method]}", fontsize=11)
                ax.set_xlabel("Generator updates", fontsize=9)
                ax.set_ylabel("Method-specific loss", fontsize=9)
                ax.grid(alpha=0.2)
                if ax.lines:
                    ax.legend(fontsize=8, frameon=False)
                else:
                    ax.text(0.5, 0.5, "No recorded loss", ha="center", transform=ax.transAxes, color="#71767c")
        stage = "VALIDATION-PHASE PREVIEW" if inputs.split == "val" else "Frozen final training runs"
        fig.suptitle(stage + " - method-specific objectives, independent y-scales; not a cross-method loss ranking.", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save_figure(fig, out / "training_loss_curves", dpi, files)


def self_test():
    import unittest

    suite = unittest.defaultTestLoader.loadTestsFromName("paper.benchmarks2026.test_plots")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", nargs="+", type=Path)
    parser.add_argument("--extra-phase", action="append", type=Path, default=[])
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--require-complete", action="store_true", help="Require all four methods and seeds 0,1,2 for requested datasets")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.phase:
        parser.error("--phase is required unless --self-test is used")
    if args.dpi < 200:
        parser.error("--dpi must be >=200 (overview width >=4000 px)")
    phases = list(dict.fromkeys(path.resolve() for path in (*args.phase, *args.extra_phase)))
    for phase in phases:
        if not phase.is_dir() or not phase.is_relative_to(BASE.resolve()) or phase == BASE.resolve():
            parser.error(f"Expected an existing phase inside {BASE}: {phase}")
    split = args.split or ("test" if all(is_final_phase(p) for p in phases) else "val")
    if split == "test" and not all(is_final_phase(p) for p in phases):
        parser.error("Test plotting is restricted to final_* phases with completed score_test=True jobs")
    requested = [data for data in DATASETS if data in args.datasets]
    inputs = Inputs(split)
    selected, candidates = discover(phases, split, requested, inputs)
    if not selected:
        parser.error("No eligible completed jobs. No figures or held-out arrays were read.")
    missing = [(data, method, seed) for data in requested for method in METHODS for seed in EXPECTED_SEEDS
               if (data, method, seed) not in selected]
    if args.require_complete and missing:
        parser.error(f"Missing {len(missing)} dataset/method/seed cells; first: {missing[:8]}")

    rows, panels = [], {}
    for run in selected.values():
        records = metric_rows(run, split, inputs)
        rows.append({**summarize_seed(run, records, split), "selected_for_comparison": True})
        if run.key[2] != 0:
            continue
        data, method, _ = run.key
        if data not in panels:
            target = inputs.array(BASE / "data" / data / "data.npz", f"{split}_target", held_out=True)
            panels[data] = {None: target}
        points = inputs.array(run.folder / f"{split}_step{run.selected_step}_samples.npz", "generated_draw_0")
        target = panels[data][None]
        dim = 3 if data in SPHERE else 7 if data == "rna" else 2
        draw_zero = next(row for row in records if row["draw"] == 0)
        if target.shape[1] != dim or points.shape != (int(draw_zero["generated_n"]), dim) or len(target) != int(draw_zero["target_n"]):
            raise ValueError(f"Point dimension/count mismatch: {run.folder}")
        panels[data][method] = points
    if not panels:
        parser.error("No completed seed-0 sample panels are available")
    data_order = [data for data in requested if data in panels]
    aggregate = aggregate_seeds(rows)
    candidate_rows = []
    if split == "val":
        for run in candidates:
            candidate_rows.append({**summarize_seed(run, metric_rows(run, "val", inputs), "val"),
                                   "selected_for_comparison": selected[run.key].folder == run.folder})
    outbase = BASE / "visualizations" / phases[0].name
    out = outbase / split
    revision = 2
    while out.exists():
        out = outbase / f"{split}_{revision:03d}"
        revision += 1
    out.mkdir(parents=True, exist_ok=False)
    files, scales = [], {}
    tables(out, rows, aggregate, requested, split)
    if candidate_rows:
        tables(out, candidate_rows, aggregate_seeds(candidate_rows), requested, split, candidates=True)
    coast = coast_segments(inputs.read(ROOT / "paper/vendor/ne_110m_land.geojson"))
    row_specs, densities, zoom_panels = [], {}, []
    for data in data_order:
        pairs = RNA_PAIRS if data == "rna" else ((0, 1),)
        for pair in pairs:
            row_specs.append((data, pair))
            densities[(data, pair)] = {
                method: sphere_density(points) if data in SPHERE else torus_density(points, pair)
                for method, points in panels[data].items()
            }
        specs = [(data, pair) for pair in pairs]
        comparison_figure(specs, panels, densities, selected, split, coast,
                          out / f"{data}_comparison", args.dpi, files, scales)
        zoom_panels.extend(individual_panels(specs, panels, densities, selected, split, coast,
                                            out, args.dpi, files))
    comparison_figure(row_specs, panels, densities, selected, split, coast,
                      out / "overview", args.dpi, files, scales)
    if len(row_specs) > 6:
        groups = {"sphere": SPHERE, "torus2": frozenset(DATASETS[4:8]), "rna": {"rna"}}
        for name, members in groups.items():
            specs = [(data, pair) for data, pair in row_specs if data in members]
            if specs:
                comparison_figure(specs, panels, densities, selected, split, coast,
                                  out / f"overview_{name}", args.dpi, files, scales)
    curves(out, selected, data_order, inputs, args.dpi, files)
    panel_index(out, zoom_panels, data_order, split)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "split": split,
        "phases": [str(p) for p in phases], "phase_name": phases[0].name,
        "requested_datasets": requested, "plotted_datasets": data_order,
        "missing_cells": missing, "complete_three_seed_comparison": not missing,
        "preview_rule": "minimum completed selected_validation_mmd2 per dataset/method/seed" if split == "val" else "one frozen config; no test-based selection",
        "candidate_count": len(candidates), "selected_run_count": len(selected),
        "selected_runs": [{"data": r.key[0], "method": r.key[1], "seed": r.key[2],
                           "folder": str(r.folder), "step": r.selected_step,
                           "budget": r.done["steps"], **r.metadata()} for r in selected.values()],
        "candidate_runs": [{"data": r.key[0], "method": r.key[1], "seed": r.key[2],
                            "selected_for_comparison": selected[r.key].folder == r.folder,
                            **r.metadata()} for r in candidates],
        "figure_seed": 0, "figure_draw": 0,
        "table_seed_policy": "draw mean and draw SD within seed; seed mean and training-seed sample SD across available seeds 0,1,2, within identical phase/config/source/budget only; ddof=1",
        "kde": {"sphere": "vMF on S2, kappa=45, per steradian", "torus": "120x120 histogram, wrapped Gaussian sigma=2.4 bins, per radian squared",
                "scale": "row max across held-out and all available methods; LogNorm 1e-3*max to max", "rna_pairs_zero_based": RNA_PAIRS},
        "row_scales": scales, "dpi": args.dpi, "vector_pdf": True,
        "individual_panels": zoom_panels, "panel_index": str(out / "panel_index.md"),
        "source_files_sha256": inputs.files, "accessed_npz_members": inputs.archives,
        "script_sha256": digest(__file__), "warnings": inputs.warnings,
        "figures": files,
    }
    (out / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "split": split, "datasets": data_order,
                      "selected_runs": len(selected), "missing_cells": len(missing), "figure_files": len(files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
