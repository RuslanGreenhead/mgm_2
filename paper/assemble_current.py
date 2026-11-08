"""Build manuscript tables and figures from saved results, without using CUDA."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.benchmarks2026 import plots
from paper.benchmarks2026 import grasp_publication_figures as grasp_fig
from mpl_toolkits.mplot3d import proj3d

PAPER = ROOT / "paper"
BASE = ROOT / "outputs/comparison2026"
LABELS = {"mgm": "R-MGM (ours)", "rwflow": "RW-Flow-C", "kgdm": "KGDM", "meanflow": "RMF-MT"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path, caption, label, columns, heading, rows, star=True):
    env = "table*" if star else "table"
    placement = "t" if star else "H"
    text = [f"\\begin{{{env}}}[{placement}]", "\\centering\\small", f"\\caption{{{caption}}}",
            f"\\label{{{label}}}", f"\\begin{{tabular}}{{@{{}}{columns}@{{}}}}", "\\toprule",
            heading + r" \\", r"\midrule", *[row + r" \\" for row in rows],
            r"\bottomrule", r"\end{tabular}", f"\\end{{{env}}}"]
    (PAPER / "generated" / path).write_text("\n".join(text) + "\n", encoding="ascii")


def pm(mean, sd, scale=1, digits=3):
    return f"${scale*mean:.{digits}f}\\,\\pm\\,{scale*sd:.{digits}f}$"


def make_tables():
    audit = read(BASE / "paused_2026_10_26/partial_audit_manifest.json")
    assert audit["cpu_artifact_audit_passed"] and audit["counts"] == {"completed": 18, "interrupted": 2, "not_started": 88}
    cells = {(r["data"], r["method"]): r for r in audit["summary"]}
    rows = []
    for data, n in (("volcano", 82), ("earthquake", 612)):
        values = []
        for method in plots.METHODS:
            cell = cells.get((data, method))
            values.append(pm(cell["kernel_mmd2_v_mean"], cell["kernel_mmd2_v_seed_sd"], 1000) if cell else "n/a")
        rows.append(" & ".join([data.title(), str(n), *values]))
    table("recent_one_step_results.tex",
          r"\textbf{Completed modern one-step comparisons.} Test $\widehat{\mathrm{MMD}}_V^2\times10^3$, mean $\pm$ training-seed SD (three seeds); lower is better. Each available cell completed 50,000 generator/field updates and uses a validation-selected checkpoint. All samplers use one network evaluation. The paused 108-job plan has only 18 completed jobs; n/a means unavailable, not a performance score. These are small-network adaptations, not original-author benchmark reproductions.",
          "tab:recent", "llrrrr", "Dataset & Test $n$ & R-MGM (ours) & RW-Flow-C & KGDM & RMF-MT", rows)
    extra = []
    for data in ("volcano", "earthquake"):
        for method in plots.METHODS:
            r = cells.get((data, method))
            if r:
                extra.append(" & ".join([data.title(), LABELS[method],
                    pm(r["symmetric_nn_mean"], r["symmetric_nn_seed_sd"]),
                    pm(r["radius_coverage_mean"], r["radius_coverage_seed_sd"], 100, 2),
                    pm(r["one_nna_mean"], r["one_nna_seed_sd"]), f"${r['constraint_mean_mean']:.2e}$"]))
    table("recent_extra_results.tex", r"Additional completed-test metrics. NN is intrinsic symmetric nearest-neighbor distance in radians (lower is better); coverage uses the fixed training-reference radius (higher is better); 1-NNA has an ideal reference near 0.5. Norm error is mean absolute sphere-unit-norm error. Different metrics need not induce the same ranking.",
          "tab:recent-extra", "llrrrr", r"Data & Method & NN & Coverage (\%) & 1-NNA & Norm error", extra, star=False)
    summary = read(BASE / "grasp/final_summary.json")
    assert read(BASE / "grasp/final_audit.json")["status"] == "PASSED"
    groups = defaultdict(list)
    for row in summary["rows"]:
        if row["split"] == "test":
            groups[row["method"], row["nfe"], row["seed"]].append(row)
    grasp_rows, per_object = [], []
    keys = (("mgm", 1, "R-MGM (ours)"), ("rmf_mt", 1, "RMF-MT adaptation"), ("rfm", 50, "RFM"), ("rfm", 100, "RFM"))
    for method, nfe, title in keys:
        scores = []
        for key, scale, digits in (("mmd2", 1000, 3), ("static_proxy_pose_fraction", 100, 2), ("official_franka_collision_fraction", 100, 2)):
            values = []
            for seed in (0, 1, 2):
                records = groups[method, nfe, seed]
                assert len(records) == 2 and len({r["object_id"] for r in records}) == 2
                values.append(np.mean([r[key] for r in records]))
            scores.append(pm(np.mean(values), np.std(values, ddof=1), scale, digits))
        grasp_rows.append(" & ".join([title, str(nfe), *scores]))
        for category in ("Mug", "Table"):
            scores = []
            for key, scale in (("mmd2", 1000), ("static_proxy_pose_fraction", 100), ("official_franka_collision_fraction", 100)):
                values = [next(r[key] for r in groups[method, nfe, seed] if r["object_id"].startswith(category + "_")) for seed in (0, 1, 2)]
                scores.append(pm(np.mean(values), np.std(values, ddof=1), scale))
            per_object.append(" & ".join([category, title, str(nfe), *scores]))
    table("grasp_results.tex", r"\textbf{Completed two-object SE(3) pilot.} Equal object means are formed within each training seed, then mean $\pm$ sample SD over three seeds. Every model trains for 10,000 generator/field updates. Pose MMD is an \emph{unbiased product-kernel} statistic, distinct from \cref{tab:main,tab:recent}. The thin-gripper static proxy is not physical success; full-Franka collision tests a different geometry. Reference poses have 92\%/100\% static-proxy passes on Mug/Table and 0\% full collisions. Lower pose MMD and collision, but higher proxy fraction, are better.",
          "tab:grasp", "lrrrr", r"Method & NFE & Pose MMD$^2\times10^3$ & Static proxy (\%) & Full collision (\%)", grasp_rows)
    table("grasp_per_object.tex", r"Per-object test results over three training seeds. MMD is multiplied by $10^3$; static-proxy and full-collision fractions are percentages. Metrics use 512 generated poses and 100 unfiltered geometry checks per object. These are held-out poses for two known objects, not unseen-object results.",
          "tab:grasp-object", "lllrrr", r"Object & Method & NFE & Pose MMD$^2\times10^3$ & Static proxy (\%) & Full collision (\%)", per_object, star=False)
    with (BASE / "paused_2026_10_26/completed_test_per_seed.csv").open(newline="") as stream:
        complete = list(csv.DictReader(stream))
    interrupted = [read(ROOT / item["output"] / "progress.json") for item in audit["job_status"] if item["status"] == "interrupted"]
    grasp_cost = read(BASE / "grasp/completion_report.json")
    resources = [
        f"Recent completed jobs & {len(complete)} & {sum(int(r['steps']) for r in complete):,} & {sum(int(r['critic_updates']) for r in complete):,} & {sum(float(r['train_seconds']) for r in complete)/60:.2f} & 2",
        f"Recent saved partial jobs & {len(interrupted)} & {sum(r['step'] for r in interrupted):,} & {sum(500 + 10*r['step'] for r in interrupted):,} & {sum(r['train_seconds'] for r in interrupted)/60:.2f} & 2",
        f"Completed grasp jobs & 9 & {grasp_cost['generator_or_field_updates']:,} & {grasp_cost['critic_updates']:,} & {grasp_cost['recorded_training_seconds']/60:.2f} & 1",
    ]
    table("current_resources.tex", r"Recorded training on one RTX 4090 (24 GB), PyTorch 2.5.1/CUDA 12.1. Minutes sum per-job training wall times, including critic warmup but excluding validation/evaluation; two concurrent workers cause overlapping wall time, so this is not device occupancy or an efficiency ranking. Partial rows stop at durable checkpoints and omit discarded in-flight updates. These rows exclude calibration/refinement, solver/throughput probes, failed preliminary attempts, and older studies; they are not total project GPU cost. The earlier 72-job controlled bundle separately records 17.03 training minutes.",
          "tab:current-cost", "lrrrrr", r"Evidence block & Jobs & G/field updates & Critic updates & Minutes & Max workers", resources, star=False)


def export(fig, name):
    fig.savefig(PAPER / "figures" / f"{name}.png", dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.08)
    fig.savefig(PAPER / "figures" / f"{name}.pdf", facecolor="white", bbox_inches="tight", pad_inches=.08)
    plots.plt.close(fig)


def volcano():
    manifest = read(BASE / "visualizations/final_v1/test_002/plot_manifest.json")
    selected = {r["method"]: r for r in manifest["selected_runs"] if r["data"] == "volcano" and r["seed"] == 0}
    with np.load(BASE / "data/volcano/data.npz", allow_pickle=False) as archive:
        entries = [("Held-out test", archive["test_target"].copy())]
    for method in plots.METHODS:
        r = selected[method]
        with np.load(Path(r["folder"]) / f"test_step{r['step']}_samples.npz", allow_pickle=False) as archive:
            entries.append((LABELS[method], archive["generated_draw_0"].copy()))
    density = [plots.sphere_density(x) for _, x in entries]
    vmax = max(float(d[2].max()) for d in density)
    coast = plots.coast_segments(read(PAPER / "vendor/ne_110m_land.geojson"))
    fig = plots.plt.figure(figsize=(13, 7.2))
    grid = fig.add_gridspec(2, 3, left=.07, right=.87, top=.91, bottom=.10, hspace=.53, wspace=.24)
    for i, ((title, x), d) in enumerate(zip(entries, density)):
        ax = fig.add_subplot(grid[i // 3, i % 3], projection="mollweide")
        plots.render_panel(ax, d, vmax, True, (0, 1), coast)
        ax.set_title(title, fontsize=16, pad=13)
        ax.tick_params(labelsize=13)
        ax.xaxis.label.set_size(13)
        if i % 3 == 0:
            ax.set_ylabel("Latitude", fontsize=14, labelpad=12)
    ax = fig.add_subplot(grid[1, 2]); ax.set_axis_off()
    ax.text(.02, .90, "Volcano | test\nTraining seed 0, draw 0\nn = 82 per panel\n\nSame KDE and density scale\nfor target and all methods.", fontsize=14, va="top", transform=ax.transAxes, linespacing=1.6)
    bar = plots.density_colorbar(fig, fig.add_axes((.91, .24, .012, .50)), vmax, True)
    bar.ax.tick_params(labelsize=13)
    bar.ax.yaxis.label.set_size(14)
    export(fig, "recent_volcano")


def projected_geometry(ax, points):
    xx, yy, _ = proj3d.proj_transform(*points.T, ax.get_proj())
    return ax.transData.transform(np.column_stack((xx, yy)))


def grasp():
    plan = read(BASE / "grasp/pilot_plan_gpu_v2.json")
    manifest = read(plan["manifest"])
    fig = plots.plt.figure(figsize=(16, 8.8))
    fig.subplots_adjust(left=.015, right=.985, bottom=.06, top=.89, wspace=.025, hspace=.20)
    receipt = []
    short_titles = ("Held-out reference", "R-MGM, 1 NFE", "RMF-MT adapter, 1 NFE", "RFM, 100 NFE")
    for row, record in enumerate(manifest["objects"]):
        panels, evidence = grasp_fig.load_panels(0, record["object_id"], plan)
        assert sha(record["mesh_path"]) == record["mesh_sha256"]
        mesh = grasp_fig.load_mesh(record["mesh_path"], record["mesh_scale"])
        points = [mesh.vertices]
        for panel in panels:
            panel["segments"] = np.einsum("nij,skj->nski", panel["rotations"], grasp_fig.SEGMENTS_M) + panel["translations"][:, None, None, :]
            points.append(panel["segments"].reshape(-1, 3))
        all_points = np.concatenate(points)
        center = (all_points.min(0) + all_points.max(0)) / 2
        half = np.ptp(all_points, axis=0).max() * .56
        limits = [(c - half, c + half) for c in center]
        axes = []
        for col, panel in enumerate(panels):
            panel["title"] = record["category"] + "\n" + short_titles[col]
            grasp_fig.draw_panel(fig, (2, 4, row * 4 + col + 1), panel, mesh, limits, fontsize=17)
            ax = fig.axes[-1]
            ax.set_proj_type("ortho")
            axes.append(ax)
        # Enlarge the common camera only as far as all actually displayed
        # mesh/gripper vertices remain inside every panel's viewport.
        low, high = .5, 4.0
        for _ in range(13):
            zoom = (low + high) / 2
            for ax in axes:
                ax.set_box_aspect((1, 1, 1), zoom=zoom)
            fig.canvas.draw()
            valid = True
            for ax, panel in zip(axes, panels):
                xy = projected_geometry(ax, np.concatenate((mesh.vertices, panel["segments"].reshape(-1, 3))))
                bbox = ax.get_window_extent()
                valid &= bool(((xy[:, 0] >= bbox.x0 + 12) & (xy[:, 0] <= bbox.x1 - 12) & (xy[:, 1] >= bbox.y0 + 12) & (xy[:, 1] <= bbox.y1 - 20)).all())
            if valid:
                low = zoom
            else:
                high = zoom
        for ax in axes:
            ax.set_box_aspect((1, 1, 1), zoom=low)
        receipt.append({"object": record["category"], "common_zoom": low, "shared_bounds": np.asarray(limits).tolist(), "evidence": evidence})
    export(fig, "grasp_current")
    return receipt


def curves():
    with (BASE / "paused_2026_10_26/validation_curves.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plots.plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, data in zip(axes, ("volcano", "earthquake")):
        for method in plots.METHODS:
            for seed in (0, 1, 2):
                subset = sorted((r for r in rows if r["data"] == data and r["method"] == method and int(r["seed"]) == seed), key=lambda r: int(r["step"]))
                if subset:
                    ax.plot([int(r["step"]) for r in subset], [1000 * float(r["mmd2"]) for r in subset], color=plots.COLORS[method], linestyle=("-", "--", ":")[seed], lw=1.3,
                            label=LABELS[method] if seed == 0 else "_nolegend_")
        ax.set(title=data.title(), xlabel="Generator/field updates", ylabel="Validation MMD squared x 1000")
        ax.set_yscale("symlog", linthresh=.1)
        ax.legend(ncol=2, fontsize=13, frameon=False)
        ax.tick_params(labelsize=13)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.title.set_size(16)
        ax.grid(alpha=.2)
    fig.tight_layout()
    export(fig, "recent_validation")


def main():
    make_tables()
    volcano()
    receipt = grasp()
    curves()
    assert not grasp_fig.torch.cuda.is_initialized()
    sources = [BASE / "paused_2026_10_26/partial_audit_manifest.json", BASE / "grasp/final_summary.json", BASE / "grasp/final_audit.json"]
    generated = list((PAPER / "generated").glob("recent_*.tex")) + list((PAPER / "generated").glob("grasp_*.tex"))
    generated.append(PAPER / "generated/current_resources.tex")
    generated += [PAPER / "figures" / (name + suffix) for name in ("recent_volcano", "grasp_current", "recent_validation") for suffix in (".pdf", ".png")]
    report = {"cpu_only": True, "cuda_initialized": False, "script_sha256": sha(__file__),
              "inputs": {str(p.relative_to(ROOT)): sha(p) for p in sources},
              "outputs": {str(p.relative_to(ROOT)): sha(p) for p in generated}, "grasp_views": receipt}
    (PAPER / "generated/current_snapshot_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tables": 5, "new_figures": 3, "cuda_initialized": False}))


if __name__ == "__main__":
    main()
