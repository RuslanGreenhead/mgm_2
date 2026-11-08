"""CPU-only publication views; deliberately outside the frozen grasp package."""

import argparse
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from .grasp import OUTPUT
from .grasp.data import load_mesh, sha256, write_json
from .grasp.runner import source_hashes

torch.set_num_threads(1)

SEGMENTS_M = np.array([[[.041, 0, .066], [.041, 0, .11217]],
                       [[-.041, 0, .066], [-.041, 0, .11217]],
                       [[0, 0, 0], [0, 0, .066]],
                       [[-.041, 0, .066], [.041, 0, .066]]])
COLORS = {"reference": "#23834a", "mgm": "#c56325", "rmf_mt": "#9c3a83", "rfm": "#187ca3"}
CAMERA = {"elevation": 26, "azimuth": -56}
METHODS = (("mgm", 1, "MGM"), ("rmf_mt", 1, "RMF-MT source-loss adaptation"), ("rfm", 100, "RFM"))


def load_panels(seed, object_id, plan):
    panels, evidence, reference_indices = [], [], None
    for method, nfe, title in METHODS:
        metric_path = OUTPUT / "public_within_pilot" / method / f"seed{seed}" / f"eval_test_nfe{nfe}" / "metrics.json"
        report = json.loads(metric_path.read_text())
        if report["training_iteration"] != 10000 or report["split"] != "test" or report["seed"] != seed:
            raise ValueError("Publication views require the frozen final test outputs")
        if report["evaluation_source_sha256"] != plan["source_sha256"] or report["manifest_sha256"] != plan["manifest_sha256"]:
            raise ValueError("Evaluation provenance changed")
        obj = next(o for o in report["objects"] if o["object_id"] == object_id)
        if sha256(obj["samples_path"]) != obj["samples_sha256"]:
            raise ValueError("Saved poses changed")
        with np.load(obj["samples_path"]) as samples:
            if reference_indices is None:
                reference_indices = samples["reference_source_indices"][:12].copy()
                reference = {"method": "reference", "title": "Held-out ACRONYM reference",
                    "rotations": samples["target_rotations"][:12].copy(),
                    "translations": samples["target_transforms_m"][:12, :3, 3].copy(),
                    "source_indices": reference_indices.tolist(), "draw_indices": list(range(12))}
            elif not np.array_equal(reference_indices, samples["reference_source_indices"][:12]):
                raise ValueError("Reference ordering differs across methods")
            panels.append({"method": method, "title": f"{title} | {nfe} NFE",
                "rotations": samples["rotations"][:12].copy(),
                "translations": samples["translations_m"][:12].copy(), "draw_indices": list(range(12))})
        evidence.append({"method": method, "metrics": str(metric_path.resolve()), "metrics_sha256": sha256(metric_path),
            "samples": obj["samples_path"], "samples_sha256": obj["samples_sha256"],
            "checkpoint": report["checkpoint"], "checkpoint_sha256": report["checkpoint_sha256"]})
    return [reference] + panels, evidence


def draw_panel(figure, location, panel, mesh, limits, fontsize=14):
    ax = figure.add_subplot(*location, projection="3d")
    light = np.array([.3, -.4, .86])
    colors = np.array([.66, .72, .75])[None] * (.67 + .33*np.abs(mesh.face_normals @ light))[:, None]
    ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors=colors, edgecolors="none", alpha=.98))
    ax.add_collection3d(Line3DCollection(panel["segments"].reshape(-1, 2, 3), colors=COLORS[panel["method"]],
                                       linewidths=1.15, alpha=.85))
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=CAMERA["elevation"], azim=CAMERA["azimuth"])
    ax.set_axis_off()
    ax.set_title(panel["title"], fontsize=fontsize, color=COLORS[panel["method"]], pad=6)


def export_figure(figure, path, dpi):
    figure.savefig(path.with_suffix(".png"), dpi=dpi, facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)
    return {"png": str(path.with_suffix(".png").resolve()), "pdf": str(path.with_suffix(".pdf").resolve()),
            "png_sha256": sha256(path.with_suffix(".png")), "pdf_sha256": sha256(path.with_suffix(".pdf"))}


def render(seed, object_record, plan, destination, provisional):
    panels, evidence = load_panels(seed, object_record["object_id"], plan)
    if sha256(object_record["mesh_path"]) != object_record["mesh_sha256"]:
        raise ValueError("Actual object mesh changed")
    mesh = load_mesh(object_record["mesh_path"], object_record["mesh_scale"])
    all_points = [mesh.vertices]
    for panel in panels:
        if len(panel["rotations"]) != 12:
            raise ValueError("Exactly the first 12 unfiltered draws are required")
        panel["segments"] = np.einsum("nij,skj->nski", panel["rotations"], SEGMENTS_M) + panel["translations"][:, None, None, :]
        all_points.append(panel["segments"].reshape(-1, 3))
    points = np.concatenate(all_points)
    center = (points.min(0)+points.max(0))/2
    half_extent = float(np.ptp(points, axis=0).max())*.56
    limits = [(float(c-half_extent), float(c+half_extent)) for c in center]
    assert all(((p["segments"] >= np.asarray(limits)[:, 0]) & (p["segments"] <= np.asarray(limits)[:, 1])).all() for p in panels)
    stage = "PROVISIONAL SEED 0 ONLY; NOT A THREE-SEED AGGREGATE" if provisional else f"FIXED FINAL WEIGHTS | SEED {seed}"
    category = object_record["category"]
    figure = plt.figure(figsize=(24, 6.7), facecolor="white")
    for index, panel in enumerate(panels):
        draw_panel(figure, (1, 4, index+1), panel, mesh, limits)
    figure.suptitle(f"{category} | {stage}", y=.985, fontsize=17)
    caption = ("First 12 unfiltered poses per panel, in meters. Shared camera and bounds contain the object and EVERY displayed gripper in full.\n"
               "Thin open-gripper centerlines match the author static-proxy geometry. No feasibility selection; no physical-success claim.\n"
               "This displayed subset is not an all-draw tail view; the separate full-range figures retain all 512 generated pose origins.")
    figure.text(.5, .035, caption, ha="center", fontsize=11)
    figure.subplots_adjust(left=.005, right=.995, bottom=.16, top=.83, wspace=.02)
    base = destination / f"{category.lower()}_comparison"
    artifacts = {"comparison": export_figure(figure, base, 220), "individual_zooms": []}
    for panel in panels:
        zoom = plt.figure(figsize=(8, 8), facecolor="white")
        draw_panel(zoom, (1, 1, 1), panel, mesh, limits, fontsize=15)
        zoom.suptitle(f"{category} | seed {seed} | first 12 unfiltered poses", y=.98, fontsize=13)
        zoom.text(.5, .035, "Same camera and common metric bounds as the comparison; no cropping or pose replacement.", ha="center", fontsize=9)
        zoom.subplots_adjust(left=0., right=1., bottom=.08, top=.88)
        artifacts["individual_zooms"].append({"method": panel["method"], **export_figure(zoom, destination/f"{category.lower()}_{panel['method']}_zoom", 300)})
    receipt = {"stage": stage, "seed": seed, "object": object_record, "camera": CAMERA, "axis_limits_m": limits,
        "selection": "Exactly first 12 unfiltered saved test draws; fixed ordering, no quality selection", "displayed_draw_count": 12,
        "full_displayed_geometry_inside_bounds": True, "all_512_tail_view": False,
        "retained_full_range_figure": str(OUTPUT/"progress_figures"/f"{category.lower()}_seed0_provisional.png") if provisional else
                                      str(OUTPUT/"figures"/f"{object_record['object_id']}_seed{seed}.png"),
        "wireframe_segments_m": SEGMENTS_M.tolist(), "wireframe_source": "Same four finger/stem/palm centerlines as frozen grasp.evaluation.author_gripper",
        "panels": [{k: p[k] for k in ("method", "title", "draw_indices")} for p in panels],
        "reference_source_indices": panels[0]["source_indices"], "source_reports": evidence,
        "renderer_sha256": sha256(__file__), "frozen_source_sha256": plan["source_sha256"],
        "physical_success_rate": None, "artifacts": artifacts}
    write_json(base.with_suffix(".json"), receipt)
    return {"object_id": object_record["object_id"], "seed": seed, "receipt": str(base.with_suffix(".json").resolve()), **artifacts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument("--provisional", action="store_true")
    args = parser.parse_args()
    if args.provisional and args.seeds != [0]:
        parser.error("Provisional mode is explicitly seed0 only")
    plan = json.loads((OUTPUT/"pilot_plan_gpu_v2.json").read_text())
    if source_hashes() != plan["source_sha256"]:
        raise ValueError("Frozen package changed")
    manifest = json.loads(Path(plan["manifest"]).read_text())
    results = []
    for seed in args.seeds:
        destination = OUTPUT/"publication_figures"/("provisional_seed0" if args.provisional else f"seed{seed}")
        destination.mkdir(parents=True, exist_ok=True)
        results.extend(render(seed, record, plan, destination, args.provisional) for record in manifest["objects"])
    assert not torch.cuda.is_initialized()
    write_json(OUTPUT/"publication_figures"/("provisional_seed0_manifest.json" if args.provisional else "final_manifest.json"),
               {"cpu_only": True, "cuda_initialized": False, "frozen_package_unchanged": source_hashes() == plan["source_sha256"], "figures": results})
    print(json.dumps(results))


if __name__ == "__main__":
    main()
