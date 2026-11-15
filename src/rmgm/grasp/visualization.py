"""Actual metric object/gripper meshes, full range plus disclosed spatial detail."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from . import ROOT
from .data import load_mesh, sha256, write_json


COLORS = {"target": "#23834a", "mgm": "#c56325", "rfm": "#187ca3",
          "rmf_ccadapt": "#9c3a83", "rmf_mt": "#9c3a83", "glsd": "#bd414b"}
GRIPPER = ROOT / "external/acronym-data/public/data/franka_gripper_collision_mesh.stl"


def mesh_poses(vertices, rotations, translations):
    return np.einsum("nij,vj->nvi", rotations, vertices) + translations[:, None, :]


def frame_counts(vertices, limits):
    limits = np.asarray(limits)
    inside = ((vertices >= limits[:, 0]) & (vertices <= limits[:, 1])).all(-1)
    return {"total": len(vertices), "fully_inside": int(inside.all(1).sum()),
            "partly_outside": int((inside.any(1) & ~inside.all(1)).sum()),
            "fully_outside": int((~inside.any(1)).sum()),
            "not_fully_inside": int((~inside.all(1)).sum())}


def visualize(metrics_paths, destination, count=12, object_id=None):
    if count < 1:
        raise ValueError("Need at least one unfiltered draw")
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in metrics_paths]
    manifest_path = Path(reports[0]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(r["manifest_sha256"] != sha256(manifest_path) or r["split"] != reports[0]["split"] for r in reports):
        raise ValueError("Comparison requires the same verified manifest and held-out split")
    common = set.intersection(*[{o["object_id"] for o in r["objects"]} for r in reports])
    if not common:
        raise ValueError("No common held-out object")
    selected = object_id or sorted(common)[0]
    if selected not in common:
        raise ValueError("Requested object was not evaluated by all methods")
    record = next(o for o in manifest["objects"] if o["object_id"] == selected)
    if sha256(record["mesh_path"]) != record["mesh_sha256"]:
        raise ValueError("Object mesh changed")
    mesh, gripper = load_mesh(record["mesh_path"], record["mesh_scale"]), load_mesh(GRIPPER, 1.)
    panels, receipts = [], []
    reference_indices = None
    for report, metric_path in zip(reports, metrics_paths):
        obj = next(o for o in report["objects"] if o["object_id"] == selected)
        if sha256(obj["samples_path"]) != obj["samples_sha256"]:
            raise ValueError("Evaluation samples changed")
        with np.load(obj["samples_path"]) as f:
            panels.append({"method": report["method"], "rotations": f["rotations"],
                           "translations": f["translations_m"],
                           "title": f"{report.get('method_label', report['method'].upper())}\n{report['nfe']} NFE | seed {report['seed']} | {report['training_iteration']} updates"})
            if reference_indices is None:
                reference_indices = f["reference_source_indices"].copy()
                target = {"method": "target", "rotations": f["target_rotations"],
                          "translations": f["target_translations"]*record["radius_m"]+np.asarray(record["center_m"]),
                          "title": f"ACRONYM {report['split']} reference\nHistorical positive FleX labels"}
            elif not np.array_equal(reference_indices, f["reference_source_indices"]):
                raise ValueError("Comparison references differ")
        receipts.append({"metrics_path": str(Path(metric_path).resolve()), "metrics_sha256": sha256(metric_path),
                         "checkpoint_sha256": report["checkpoint_sha256"],
                         "samples_path": obj["samples_path"], "samples_sha256": obj["samples_sha256"]})
    panels.insert(0, target)
    bounds = [mesh.bounds]
    detail_extent = max(float(mesh.extents.max())*.75, .14)
    detail_limits = [(float(c-detail_extent), float(c+detail_extent)) for c in mesh.bounds.mean(0)]
    for panel in panels:
        all_vertices = mesh_poses(gripper.vertices, panel["rotations"], panel["translations"])
        bounds.append(np.array([all_vertices.min((0, 1)), all_vertices.max((0, 1))]))
        panel["vertices"] = all_vertices[:count]
        panel["frame_all"] = frame_counts(all_vertices, detail_limits)
        panel["frame_drawn"] = frame_counts(all_vertices[:count], detail_limits)
    bounds = np.concatenate(bounds)
    center = (bounds.min(0)+bounds.max(0))/2
    extent = max(float(np.ptp(bounds, axis=0).max()), .1)*.55
    full_limits = [(float(c-extent), float(c+extent)) for c in center]
    first_bounds = np.concatenate([mesh.bounds]+[np.array([p["vertices"][0].min(0), p["vertices"][0].max(0)]) for p in panels])
    first_center = (first_bounds.min(0)+first_bounds.max(0))/2
    first_extent = max(float(np.ptp(first_bounds, axis=0).max()), .1)*.55
    first_limits = [(float(c-first_extent), float(c+first_extent)) for c in first_center]
    camera = {"elevation": 26, "azimuth": -56}
    fig = plt.figure(figsize=(5.2*len(panels), 17.2), facecolor="white")
    light = np.array([.3, -.4, .86])
    object_colors = np.asarray([.68, .74, .77])[None]*(.65+.35*np.abs(mesh.face_normals@light))[:, None]
    for row, limits in enumerate((full_limits, detail_limits, first_limits)):
        for col, panel in enumerate(panels):
            ax = fig.add_subplot(3, len(panels), row*len(panels)+col+1, projection="3d")
            ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors=object_colors, edgecolors="none", alpha=1.))
            if row == 0:
                ax.scatter(*panel["translations"].T, color=COLORS[panel["method"]], s=2, alpha=.25, depthshade=False)
            shown = 0
            for draw, vertices in enumerate(panel["vertices"][:1] if row == 2 else panel["vertices"]):
                if row == 1 and frame_counts(vertices[None], limits)["not_fully_inside"]:
                    continue
                normals = gripper.face_normals@panel["rotations"][draw].T
                colors = np.asarray(to_rgb(COLORS[panel["method"]]))[None]*(.6+.4*np.abs(normals@light))[:, None]
                ax.add_collection3d(Poly3DCollection(vertices[gripper.faces], facecolors=colors,
                                                     edgecolors="none", alpha=1. if row == 2 else .45))
                shown += 1
            ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=camera["elevation"], azim=camera["azimuth"])
            ax.set_axis_off()
            if row == 0:
                ax.set_title(panel["title"], fontsize=12, color=COLORS[panel["method"]], pad=4)
            elif row == 1:
                counts = panel["frame_drawn"]
                ax.set_title(f"Object-centered detail | {counts['not_fully_inside']}/{counts['total']} grippers off-frame\n"
                             f"{shown} fully contained shown; {panel['frame_all']['not_fully_inside']}/{panel['frame_all']['total']} off-frame in all draws",
                             fontsize=10, pad=4)
            else:
                ax.set_title("First unfiltered draw only | no cropping\nShared bounds include every first gripper in full", fontsize=10, pad=4)
    stage = "WITHIN-OBJECT HELD-OUT-POSE PILOT; NOT UNSEEN-OBJECT ACRONYM"
    if any(report["training_iteration"] != 10000 for report in reports):
        stage += "\nCPU INTEGRATION CHECK ONLY; NOT A TRAINED BENCHMARK"
    fig.suptitle(f"{record['category']} | {selected}\n{stage}", fontsize=14, y=.985)
    fig.text(.5, .025, f"Top: full range of ALL evaluated poses (faint origins); first {count} unfiltered draws shown as actual Franka collision meshes.\n"
             "Middle: mesh-defined metric detail; off-frame means a gripper is not fully contained and is explicitly omitted from this row only.\n"
             "Bottom: first unfiltered draw, shared bounds containing all first grippers and the object; no crop or substitution.\n"
             "Identical camera and shared bounds within each row. No feasibility filtering. No physical success evaluation.", ha="center", fontsize=11)
    fig.subplots_adjust(left=.005, right=.995, bottom=.09, top=.9, wspace=0., hspace=.08)
    destination = Path(destination).with_suffix(".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, facecolor="white")
    fig.savefig(destination.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    write_json(destination.with_suffix(".json"), {
        "image_sha256": sha256(destination), "pdf_sha256": sha256(destination.with_suffix(".pdf")), "object": record,
        "gripper_mesh": str(GRIPPER), "gripper_mesh_sha256": sha256(GRIPPER), "camera": camera,
        "full_range_axis_limits_m": full_limits, "detail_axis_limits_m": detail_limits,
        "first_draw_axis_limits_m": first_limits,
        "draws_per_panel_max": count, "selection": "First draws in stored order; no quality filtering",
        "full_range_includes_all_evaluated_draws": True, "stage": stage, "source_reports": receipts,
        "reference_source_indices_drawn": reference_indices[:count].tolist(),
        "off_frame_counts": [{"method": p["method"], "first_draws": p["frame_drawn"], "all_draws": p["frame_all"]} for p in panels],
        "physical_success_evaluated": False})
    return str(destination.resolve())
