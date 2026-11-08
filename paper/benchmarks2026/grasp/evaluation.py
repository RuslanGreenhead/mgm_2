"""Distribution and static geometry metrics. NEVER a physical success evaluator."""

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from trimesh.collision import CollisionManager

from .data import load_mesh


def author_gripper():
    segments = [([0.041, 0, 0.066], [0.041, 0, 0.11217]),
                ([-0.041, 0, 0.066], [-0.041, 0, 0.11217]),
                ([0, 0, 0], [0, 0, 0.066]),
                ([-0.041, 0, 0.066], [0.041, 0, 0.066])]
    return trimesh.util.concatenate([trimesh.creation.cylinder(radius=0.002, sections=12, segment=s) for s in segments])


def grasp_volume():
    mesh = trimesh.creation.box(extents=[0.082, 0.02, 0.11217-0.066])
    mesh.apply_translation([0, 0, (0.11217+0.066)/2])
    return mesh


def static_geometry(record, rotations, translations, official_gripper=None):
    mesh = load_mesh(record["mesh_path"], record["mesh_scale"])
    manager = CollisionManager()
    manager.add_object("object", mesh)
    gripper, volume = author_gripper(), grasp_volume()
    collision, overlap, clearance, official_collision = [], [], [], []
    full = trimesh.load(official_gripper, force="mesh") if official_gripper else None
    for r, p in zip(rotations, translations):
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = r, p
        collision.append(bool(manager.in_collision_single(gripper, transform=transform)))
        overlap.append(bool(manager.in_collision_single(volume, transform=transform)))
        clearance.append(float(manager.min_distance_single(gripper, transform=transform)))
        if full is not None:
            official_collision.append(bool(manager.in_collision_single(full, transform=transform)))
    passing = np.logical_and(~np.array(collision), overlap)
    k = min(20, len(passing))
    result = {"metric_kind": "static_geometry_proxy_NOT_physical_success",
              "author_thin_gripper_collision_fraction": float(np.mean(collision)),
              "author_between_fingers_surface_overlap_fraction": float(np.mean(overlap)),
              "author_static_feasible_pose_fraction": float(passing.mean()),
              "author_static_feasible_any_at_20": bool(passing[:20].any()) if len(passing) >= 20 else None,
              "any_at_20_actual_draws": k,
              "mean_thin_gripper_clearance_m": float(np.mean(clearance)),
              "per_pose_collision": collision, "per_pose_volume_overlap": overlap,
              "per_pose_proxy_pass": passing.tolist(),
              "physical_success_rate": None,
              "limitations": "Triangle-surface collision only; simplified fixed open gripper; no containment guarantee, closure, friction, lifting, robot or scene simulation."}
    if official_collision:
        result["official_franka_mesh_collision_fraction"] = float(np.mean(official_collision))
        result["official_franka_mesh_collision_is_physical_success"] = False
    return result


def product_distance2(r1, p1, r2, p2):
    # Squared chordal SO(3) distance / 2 plus normalized R3 distance.
    dr = ((r1[:, None]-r2[None])**2).sum((-1, -2)) / 2
    dp = ((p1[:, None]-p2[None])**2).sum(-1)
    return dr + dp


def distribution_metrics(r, p, target_r, target_p, radius):
    if min(len(r), len(target_r)) < 2:
        raise ValueError("Unbiased MMD needs >=2 draws per sample")
    xy = product_distance2(r, p, target_r, target_p)
    xx = product_distance2(r, p, r, p)
    yy = product_distance2(target_r, target_p, target_r, target_p)
    # Fixed preregistered bandwidths; never tuned on the test samples.
    values = []
    for bandwidth in (0.5, 1., 2., 4.):
        a, b, c = [np.exp(-d/(2*bandwidth**2)) for d in (xx, yy, xy)]
        values.append(float((a.sum()-np.trace(a))/(len(a)*(len(a)-1)) +
                            (b.sum()-np.trace(b))/(len(b)*(len(b)-1)) - 2*c.mean()))
    nearest = xy.argmin(1)
    angular = Rotation.from_matrix(np.transpose(r, (0, 2, 1)) @ target_r[nearest]).magnitude()
    return {"metric_kind": "distribution_only_NOT_grasp_success",
            "mmd2_unbiased_fixed_chordal_product_kernel": float(np.mean(values)),
            "kernel_bandwidths": [0.5, 1., 2., 4.],
            "mmd2_can_be_negative": True,
            "nearest_joint_reference_rotation_error_deg": float(np.degrees(angular).mean()),
            "nearest_joint_reference_translation_error_m": float((np.linalg.norm(p-target_p[nearest], axis=-1)*radius).mean()),
            "n_generated": len(r), "n_reference": len(target_r)}
