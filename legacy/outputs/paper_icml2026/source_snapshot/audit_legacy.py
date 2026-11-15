"""Read-only legacy/data audit; writes only the authorized supplement outputs."""
import hashlib
import json
import pathlib
import zipfile

import numpy as np
import pandas as pd
import torch

import controlled_suite as s


def main():
    torch.set_num_threads(1)
    output = {"data_archive_sha256": s.sha(s.REPO / "data/data.zip"), "archive_checks": [],
              "checkpoints": [], "data_files": [], "molecular_source_overlap": []}
    with zipfile.ZipFile(s.REPO / "data/data.zip") as archive:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            path = s.REPO / name
            if path.is_file():
                h = hashlib.sha256(archive.read(name)).hexdigest()
                output["archive_checks"].append({"path": str(path), "archive_sha256": h,
                                                 "matches_archive": s.sha(path) == h})
    files = list((s.REPO / "data/earth_data").glob("*.csv"))
    files += [s.REPO / "data" / kind / "aggregated_angles.tsv" for kind in ("top500", "rna")]
    files += list((s.REPO / "data/mesh").glob("*.npy")) + list((s.REPO / "data/mesh").glob("*.npz"))
    for path in files:
        item = {"path": str(path), "sha256": s.sha(path), "bytes": path.stat().st_size}
        if path.suffix in (".npy", ".npz"):
            a = np.load(path, mmap_mode="r")
            item["shapes"] = {k: list(a[k].shape) for k in a.files} if path.suffix == ".npz" else list(a.shape)
        output["data_files"].append(item)
    legacy = json.loads((s.ROOT / "outputs/rmgm_summary/final_reproduction_status.json").read_text())
    paths = [pathlib.Path(row["run_dir"]) / "checkpoints/last.pt" for row in legacy["runs"]]
    paths += list((s.ROOT / "outputs/rmgm_mesh_bunny_spot_pilot").glob("*/*/rmgm/seed*/checkpoints/last.pt"))
    for path in paths:
        ck = torch.load(path, map_location="cpu", weights_only=True)
        c = ck["config"]
        metadata = json.loads((path.parents[1] / "config.json").read_text())
        item = {"checkpoint": str(path), "checkpoint_sha256": s.sha(path), "step": ck["step"],
                "requested_steps": c["steps"], "seed": c["seed"],
                "effective_data_seed": c.get("data_seed") if c.get("data_seed") is not None else c["seed"],
                "config": c, "dataset_sizes": metadata["dataset_sizes"], "finite": True}
        for key in ("generator", "critic", "field"):
            if key in ck:
                item["finite"] &= all(torch.isfinite(v).all().item() for v in ck[key].values()
                                      if torch.is_tensor(v) and v.is_floating_point())
        if "generator" in ck and c["data"] in s.REAL:
            model = s.build_generator({"generator": c}, s.Sphere() if c["data"] in s.REAL[:4] else s.FlatTorus(), metadata["dim"])
            report = model.load_state_dict(ck["generator"], strict=False)
            item["current_code_missing_keys"] = report.missing_keys
            item["current_code_unexpected_keys"] = report.unexpected_keys
            item["current_code_strict_compatible"] = not report.missing_keys and not report.unexpected_keys
            item["generator_parameters"] = s.parameter_count(model)
        output["checkpoints"].append(item)
    for kind in ("top500", "rna"):
        df = pd.read_csv(s.REPO / "data" / kind / "aggregated_angles.tsv", sep="\t", header=None)
        groups = [("rna", df)] if kind == "rna" else [(name, df[df[3] == amino]) for name, amino in
                  [("general", "General"), ("glycine", "Glycine"), ("proline", "Proline"), ("prepro", "Pre-Pro")]]
        for name, group in groups:
            # The author-style benchmark is row-split; quantify shared source PDBs.
            src = group[0].astype(str).str.split(":").str[0].to_numpy()
            n = len(src)
            p = torch.randperm(n, generator=torch.Generator().manual_seed(0)).numpy()
            nv = n // 10
            train, val, test = set(src[p[:n-2*nv]]), set(src[p[n-2*nv:n-nv]]), set(src[p[n-nv:]])
            output["molecular_source_overlap"].append({"data": name, "row_count": n,
                        "unique_pdbs": len(set(src)), "train_pdbs": len(train), "val_pdbs": len(val),
                        "test_pdbs": len(test), "test_pdbs_also_in_train": len(test & train)})
    from manifm.rmgm.geometry import midpoint_path
    probe = s.Mesh.__new__(s.Mesh)
    torch.nn.Module.__init__(probe)
    probe.register_buffer("v", torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                                              [2., 0., 1.], [3., 0., 1.], [2., 1., 1.]], dtype=torch.float64))
    probe.register_buffer("f", torch.tensor([[0, 1, 2], [3, 4, 5]]))
    x0 = torch.tensor([[0.1, 0.1, 0.]], dtype=torch.float64)
    x1 = torch.tensor([[2.1, 0.1, 1.]], dtype=torch.float64)
    t = torch.tensor([[0.45]], dtype=torch.float64)
    point, velocity = midpoint_path(probe, x0, x1, t)
    lo, _ = midpoint_path(probe, x0, x1, t - 1e-5)
    hi, _ = midpoint_path(probe, x0, x1, t + 1e-5)
    fd = (hi - lo) / 2e-5
    output["mesh_edge_derivative_counterexample"] = {"mesh": "two disjoint triangles; operator-level edge-clamping counterexample",
                    "t": 0.45, "point": point.tolist(), "supplied_velocity": velocity.tolist(),
                    "finite_difference_velocity": fd.tolist(), "velocity_difference_l2": float((velocity - fd).norm())}
    s.write_json(s.OUT / "legacy_audit.json", output)
    print(json.dumps({"checkpoints": len(paths), "all_final_step": all(x["step"] == x["requested_steps"] for x in output["checkpoints"]),
                      "all_finite": all(x["finite"] for x in output["checkpoints"]),
                      "compatibility_exceptions": [{"path": x["checkpoint"], "missing": x["current_code_missing_keys"]}
                          for x in output["checkpoints"] if x.get("current_code_strict_compatible") is False],
                      "molecular_source_overlap": output["molecular_source_overlap"]}), flush=True)


if __name__ == "__main__":
    main()
