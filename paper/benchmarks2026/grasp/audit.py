"""Independent NumPy checks of held-out source rows and exported pose MMD."""

import numpy as np

from .data import sha256, validate_transforms


def independent_mmd(r, p, tr, tp):
    r, p, tr, tp = [np.asarray(x, dtype=np.float64) for x in (r, p, tr, tp)]
    def distance(a, x, b, y):
        a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
        rotation = ((a*a).sum(1)[:, None]+(b*b).sum(1)[None]-2*a@b.T)/2
        translation = (x*x).sum(1)[:, None]+(y*y).sum(1)[None]-2*x@y.T
        return rotation+translation
    xx, yy, xy = distance(r, p, r, p), distance(tr, tp, tr, tp), distance(r, p, tr, tp)
    estimates = []
    for bandwidth in (.5, 1., 2., 4.):
        kx, ky, kxy = [np.exp(-d/(2*bandwidth**2)) for d in (xx, yy, xy)]
        np.fill_diagonal(kx, 0.)
        np.fill_diagonal(ky, 0.)
        estimates.append(kx.sum()/(len(r)*(len(r)-1))+ky.sum()/(len(tr)*(len(tr)-1))-2*kxy.mean())
    return float(np.mean(estimates))


def audit_report(report, data):
    if report["physical_success_rate"] is not None or report["manifest_sha256"] != sha256(data.path):
        raise ValueError("Physical-success or manifest provenance violation")
    receipts = []
    for index in data.splits[report["split"]]:
        record, target = data.records[index], data.objects[index]
        obj = next(o for o in report["objects"] if o["object_id"] == record["object_id"])
        if sha256(obj["samples_path"]) != obj["samples_sha256"]:
            raise ValueError("Sample hash mismatch")
        with np.load(obj["samples_path"]) as f:
            r, p, tr, tp = [f[key] for key in ("rotations", "translations", "target_rotations", "target_translations")]
            source = f["reference_source_indices"]
            allowed = target["pose_indices"][report["split"]].numpy()
            rng = np.random.default_rng(report["eval_seed"]+100000+index)
            choose = rng.choice(allowed, min(report["evaluation_config"]["samples"], len(allowed)), replace=False)
            if (not np.array_equal(source, target["source_indices"][choose]) or
                    not np.array_equal(tr, target["r"][choose].numpy()) or not np.array_equal(tp, target["p"][choose].numpy())):
                raise ValueError("Held-out references differ from the fixed positive source-row draw")
            if len(r) != report["evaluation_config"]["samples"] or len(set(source.tolist())) != len(source):
                raise ValueError("Sample count or source-row uniqueness violation")
            validate_transforms(f["transforms_m"])
            validate_transforms(f["target_transforms_m"])
            np.testing.assert_array_equal(f["transforms_m"][:, :3, :3], r)
            np.testing.assert_allclose(f["transforms_m"][:, :3, 3], p*record["radius_m"]+np.asarray(record["center_m"]), atol=1e-8)
            np.testing.assert_allclose(f["target_transforms_m"][:, :3, 3], tp*record["radius_m"]+np.asarray(record["center_m"]), atol=1e-8)
            score = independent_mmd(r, p, tr, tp)
        if abs(score-obj["distribution"]["mmd2_unbiased_fixed_chordal_product_kernel"]) > 2e-6:
            raise ValueError("Independent MMD recomputation mismatch")
        proxy = obj["static_geometry"]
        passing = ~np.asarray(proxy["per_pose_collision"]) & np.asarray(proxy["per_pose_volume_overlap"])
        if (len(passing) != min(report["evaluation_config"]["collision_samples"], len(r)) or
                not np.array_equal(passing, proxy["per_pose_proxy_pass"]) or
                bool(passing[:20].any()) != proxy["author_static_feasible_any_at_20"] or
                float(passing.mean()) != proxy["author_static_feasible_pose_fraction"] or proxy["physical_success_rate"] is not None):
            raise ValueError("Static proxy bookkeeping differs from the unfiltered draw order")
        reference = obj["heldout_reference_static_geometry"]
        reference_count = min(report["evaluation_config"]["collision_samples"], len(source))
        reference_pass = ~np.asarray(reference["per_pose_collision"]) & np.asarray(reference["per_pose_volume_overlap"])
        if (not np.array_equal(obj["heldout_reference_proxy_source_indices"], source[:reference_count]) or
                len(reference_pass) != reference_count or not np.array_equal(reference_pass, reference["per_pose_proxy_pass"]) or
                bool(reference_pass[:20].any()) != reference["author_static_feasible_any_at_20"] or
                float(reference_pass.mean()) != reference["author_static_feasible_pose_fraction"] or reference["physical_success_rate"] is not None):
            raise ValueError("Held-out reference proxy baseline/order mismatch")
        receipts.append({"object_id": record["object_id"], "reference_rows": len(source), "generated": len(r),
                         "source_rows_verified": True, "mmd2_float64_recomputed": score,
                         "static_proxy_bookkeeping_verified": True, "heldout_reference_proxy_rows": reference_count,
                         "heldout_reference_proxy_order_verified": True})
    return receipts
