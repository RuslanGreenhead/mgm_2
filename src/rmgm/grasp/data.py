"""Pinned public data, explicit provenance, strict mesh-identity splits."""

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import h5py
import numpy as np
import requests
import torch
import trimesh

from . import OUTPUT, ROOT

ACRONYM_COMMIT = "d02f58fb3a556c3b719976e206cce0dce0103199"
AUTHOR_COMMIT = "45d96bcd09a271ea3a770b2014865dabbac9eeeb"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fetch_public(destination=None):
    destination = Path(destination or ROOT / "external/acronym-data/public")
    response = requests.get(f"https://api.github.com/repos/NVlabs/acronym/git/trees/{ACRONYM_COMMIT}?recursive=1", timeout=60)
    response.raise_for_status()
    files = [x for x in response.json()["tree"] if x["type"] == "blob" and
             (x["path"].startswith("data/") or x["path"] in ("LICENSE", "README.md", "THIRDPARTYLICENSES"))]
    receipts = []
    for item in files:
        url = f"https://raw.githubusercontent.com/NVlabs/acronym/{ACRONYM_COMMIT}/{item['path']}"
        target = destination / item["path"]
        if not target.exists():
            result = requests.get(url, timeout=120)
            result.raise_for_status()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(result.content)
        contents = target.read_bytes()
        blob = hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
        if blob != item["sha"]:
            raise ValueError(f"Upstream Git blob verification failed: {target}")
        receipts.append({"path": str(target.resolve()), "url": url, "bytes": len(contents),
                         "git_blob_sha1": blob, "sha256": sha256(target)})
    manifest = {"source": "NVIDIA ACRONYM official public examples", "commit": ACRONYM_COMMIT,
                "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_license": "CC BY-NC 4.0", "code_license": "MIT",
                "license_source": "https://github.com/NVlabs/acronym#license",
                "mesh_scope": "Only the two mesh examples explicitly distributed in the official repository; not a license for full ShapeNetSem.",
                "files": receipts}
    write_json(destination / "download_manifest.json", manifest)
    return manifest


def audit_access():
    urls = {
        "shapenet_mesh_archive": "https://huggingface.co/datasets/ShapeNet/ShapeNetSem-archive/resolve/main/ShapeNetSem.zip",
        "official_grasp_repository": f"https://api.github.com/repos/NVlabs/acronym/git/trees/{ACRONYM_COMMIT}?recursive=1",
        "official_full_grasp_archive": "https://drive.usercontent.google.com/download?id=1OjykLD9YmnFdfYpH2qO8yBo-I-22vKwu&export=download&confirm=t",
    }
    probes = {}
    for name, url in urls.items():
        try:
            response = requests.get(url, headers={"Range": "bytes=0-1023"}, stream=True, timeout=60)
            probes[name] = {"url": url, "status": response.status_code,
                            "error_code": response.headers.get("X-Error-Code"),
                            "content_type": response.headers.get("Content-Type"),
                            "content_range": response.headers.get("Content-Range"),
                            "content_length": response.headers.get("Content-Length")}
            response.close()
        except requests.RequestException as exc:
            probes[name] = {"url": url, "error": str(exc)}
    author_root = ROOT / "external/riemannian-meanflow-author"
    paths = ["README.md", "grasp/src/core/visualize.py", "grasp/src/models/lightning.py",
             "grasp/src/models/flow.py", "grasp/src/data/dataset.py", "grasp/src/data/data_manager.py",
             "rmf/rmf/models/flowmap_module.py"]
    report = {"utc": datetime.now(timezone.utc).isoformat(), "probes": probes,
              "author_commit_inspected": AUTHOR_COMMIT,
              "author_files": {p: sha256(author_root / p) for p in paths if (author_root / p).exists()},
              "paper": "https://arxiv.org/html/2603.10718v3#S5.SS2",
              "author_metric": {"physical_simulator": False,
                                "pose_rule": "No intersection with thin gripper mesh AND intersection with between-finger box, using FCL triangle meshes.",
                                "aggregation": "Fraction of objects with at least one passing pose among 20 draws (configured sample count).",
                                "evidence": ["grasp/src/core/visualize.py:174-318", "grasp/src/models/lightning.py:688-754"],
                                "limitations": "No finger closing, contact force/friction, lifting, dynamics, scene/table collision or robot reachability."},
              "data_label": "grasps/qualities/flex/object_in_gripper is an existing FleX simulation label, not an evaluator for generated poses.",
              "physical_evaluation_status": "Unavailable in released author evaluator; NVIDIA FleX generation pipeline is not included in the ACRONYM tools.",
              "full_benchmark_status": "Blocked until at least three licensed distinct matched mesh/H5 objects are supplied; two public examples are smoke-only."}
    previous_path = OUTPUT / "access_audit.json"
    history_path = OUTPUT / "access_audit_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if not history or history[-1]["utc"] != previous["utc"]:
            history.append(previous)
    history.append(report)
    write_json(history_path, history)
    write_json(previous_path, report)
    return report


def audit_author_data():
    api = "https://api.github.com/repos/haolsun/Riemannian_MeanFlow"
    tree_response = requests.get(f"{api}/git/trees/{AUTHOR_COMMIT}?recursive=1", timeout=60)
    tree_response.raise_for_status()
    tree = tree_response.json()["tree"]
    assets = [x for x in tree if x["type"] == "blob" and x["path"].startswith("grasp/") and
              x["path"].endswith((".h5", ".obj", ".npz", ".pkl", ".ckpt", ".pt", ".glb", ".zip", ".tar.gz"))]
    root = ROOT / "external/riemannian-meanflow-author"
    for asset in assets:
        local = root / asset["path"]
        asset["local_sha256"] = sha256(local) if local.is_file() else None
    release_response = requests.get(api + "/releases", timeout=30)
    release_response.raise_for_status()
    report = {"utc": datetime.now(timezone.utc).isoformat(), "author_commit": AUTHOR_COMMIT,
              "searched": [f"https://github.com/haolsun/Riemannian_MeanFlow/tree/{AUTHOR_COMMIT}/grasp",
                           "https://github.com/haolsun/Riemannian_MeanFlow/releases",
                           "https://github.com/Ferdydh/RFM-Grasp", "https://github.com/NVlabs/acronym",
                           "https://huggingface.co/datasets/ShapeNet/ShapeNetSem-archive"],
              "author_tracked_data_assets": assets,
              "author_release_count": len(release_response.json()),
              "author_license_files": [x["path"] for x in tree if "license" in x["path"].lower() and "egg-info" not in x["path"]],
              "finding": "Author tree contains one additional 1Shelves H5/OBJ pair, but no explicit repository license grant was found. It is NOT included in this pilot. No processed grasp cache or checkpoint bundle is tracked or linked by the grasp README/code.",
              "rfm_grasp_finding": "Upstream RFM-Grasp main tree has only data/.gitkeep; README requires supplying the ACRONYM and ShapeNetSem data, no extra processed bundle link.",
              "authorized_pilot_assets": "Exactly the NVIDIA-distributed public mug/table matched pairs under CC BY-NC 4.0; full meshes remain gated.",
              "action": "No massive full-H5 download, no gated-license bypass, no third-object inclusion; user confirmed no local licensed full data."}
    write_json(OUTPUT / "source_data_search.json", report)
    return report


def fetch_full_grasps(destination=None, expected_sha256=None):
    """Optional 1.6 GB official archive; no meshes or gated access included."""
    destination = Path(destination or ROOT / "external/acronym-data/full")
    destination.mkdir(parents=True, exist_ok=True)
    url = "https://drive.usercontent.google.com/download?id=1OjykLD9YmnFdfYpH2qO8yBo-I-22vKwu&export=download&confirm=t"
    archive = destination / "acronym.tar.gz"
    if not archive.exists():
        temporary = destination / "acronym.tar.gz.partial"
        offset = temporary.stat().st_size if temporary.exists() else 0
        response = requests.get(url, stream=True, headers={"Range": f"bytes={offset}-"} if offset else {}, timeout=(30, 120))
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            response.close()
            raise ValueError("Google Drive returned HTML instead of the archive. Download the official linked archive manually; no access gate is bypassed.")
        if offset and response.status_code != 206:
            offset = 0
        with open(temporary, "ab" if offset else "wb") as f:
            for chunk in response.iter_content(1024*1024):
                f.write(chunk)
        response.close()
        temporary.replace(archive)
    digest = sha256(archive)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("Full ACRONYM archive checksum mismatch")
    extracted = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if name.suffix != ".h5":
                continue
            if name.is_absolute() or ".." in name.parts or not member.isfile() or member.size > 20*1024*1024:
                raise ValueError("Unexpected or unsafe grasp archive member")
            target = destination / "grasps" / name.name
            target.parent.mkdir(parents=True, exist_ok=True)
            contents = tar.extractfile(member).read()
            if target.exists() and target.read_bytes() != contents:
                raise ValueError(f"Conflicting archive basename: {name.name}")
            target.write_bytes(contents)
            extracted.append({"archive_member": member.name, "path": str(target.resolve()), "sha256": sha256(target)})
    if not extracted:
        raise ValueError("No actual HDF5 grasp files in downloaded archive")
    receipt = {"source": "Official NVIDIA ACRONYM full grasp archive", "url": url,
               "dataset_license": "CC BY-NC 4.0", "mesh_license": "ShapeNet terms separately required",
               "archive_path": str(archive.resolve()), "archive_sha256": digest,
               "checksum_status": "verified against supplied checksum" if expected_sha256 else "observed SHA256; upstream README publishes no checksum",
               "downloaded_utc": datetime.now(timezone.utc).isoformat(), "files": extracted}
    write_json(destination / "download_manifest.json", receipt)
    return receipt


def safe_mesh_path(root, name):
    root = Path(root).resolve()
    relative = PurePosixPath(name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or ":" in name:
        raise ValueError(f"Unsafe HDF5 mesh reference: {name}")
    candidate = root.joinpath(*relative.parts).resolve()
    if root not in candidate.parents:
        raise ValueError(f"Mesh reference escapes root: {name}")
    return candidate


def load_mesh(path, scale):
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Not a nonempty triangle mesh: {path}")
    mesh.apply_scale(float(scale))
    return mesh


def sample_surface(mesh, n, rng):
    indices = rng.choice(len(mesh.faces), n, p=mesh.area_faces / mesh.area_faces.sum())
    triangle = mesh.triangles[indices]
    uv = rng.random((n, 2))
    su = np.sqrt(uv[:, :1])
    points = (1 - su) * triangle[:, 0] + su * (1 - uv[:, 1:]) * triangle[:, 1] + su * uv[:, 1:] * triangle[:, 2]
    return points, mesh.face_normals[indices]


def validate_transforms(transforms, tolerance=2e-4):
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or not np.isfinite(transforms).all():
        raise ValueError("Expected finite Nx4x4 transforms")
    r = transforms[:, :3, :3]
    orth = float(np.abs(r.transpose(0, 2, 1) @ r - np.eye(3)).max())
    det = float(np.abs(np.linalg.det(r) - 1).max())
    bottom = float(np.abs(transforms[:, 3, :] - [0, 0, 0, 1]).max())
    if max(orth, det, bottom) > tolerance:
        raise ValueError(f"Invalid SE(3) source transforms: {orth=}, {det=}, {bottom=}")
    return {"orthogonality_max": orth, "determinant_error_max": det, "bottom_row_error_max": bottom}


def make_split(identities, seed, smoke_only=False):
    unique = sorted(set(identities), key=lambda x: hashlib.sha256(f"{seed}:{x}".encode()).hexdigest())
    if len(unique) < 3:
        if not smoke_only or len(unique) < 2:
            raise ValueError("Need >=3 distinct mesh identities for nonempty train/val/test. The two official examples are smoke-only; use --smoke-only explicitly.")
        groups = {"train": unique[:1], "val": [], "test": unique[1:]}
    else:
        nv = max(1, round(len(unique) * 0.1))
        nt = max(1, round(len(unique) * 0.1))
        groups = {"train": unique[:len(unique)-nv-nt], "val": unique[len(unique)-nv-nt:len(unique)-nt], "test": unique[-nt:]}
    return groups


def make_pose_split(positive_indices, seed, object_id):
    indices = np.asarray(positive_indices, dtype=np.int64)
    if len(indices) < 10 or len(set(indices.tolist())) != len(indices):
        raise ValueError("Within-object pilot requires >=10 unique positive source indices")
    local_seed = int(hashlib.sha256(f"{seed}:{object_id}:positive_pose_split".encode()).hexdigest()[:16], 16)
    shuffled = np.random.default_rng(local_seed).permutation(indices)
    ntrain, nval = int(.8*len(indices)), int(.1*len(indices))
    result = {"train": sorted(shuffled[:ntrain].tolist()),
              "val": sorted(shuffled[ntrain:ntrain+nval].tolist()),
              "test": sorted(shuffled[ntrain+nval:].tolist())}
    validate_pose_split(result, indices)
    return result


def validate_pose_split(split, positives):
    if set(split) != {"train", "val", "test"}:
        raise ValueError("Exactly train/val/test pose partitions are required")
    if any(len(v) != len(set(v)) for v in split.values()):
        raise ValueError("Repeated source row inside a pose partition")
    sets = [set(split[s]) for s in ("train", "val", "test")]
    if any(not x for x in sets) or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Pose splits must be nonempty and pairwise disjoint")
    if set.union(*sets) != set(np.asarray(positives).tolist()):
        raise ValueError("Pose split union must equal all and only positive HDF5 source indices")


def verify_record(record, seed):
    """Reconstruct the prepared arrays from the matched raw mesh/H5 pair on CPU."""
    for kind in ("h5", "mesh", "cache"):
        path = record["cache"] if kind == "cache" else record[kind + "_path"]
        if sha256(path) != record[kind + "_sha256"]:
            raise ValueError(f"Raw/prepared {kind} hash changed: {record['object_id']}")
    with h5py.File(record["h5_path"], "r") as f:
        name = f["object/file"][()]
        name = name.decode() if isinstance(name, bytes) else str(name)
        relative = PurePosixPath(name.replace("\\", "/"))
        mesh_path = Path(record["mesh_path"]).resolve()
        if relative.is_absolute() or ".." in relative.parts or ":" in name:
            raise ValueError("Unsafe source mesh identity")
        if tuple(mesh_path.parts[-len(relative.parts):]) != relative.parts:
            raise ValueError("HDF5 object/file does not match the actual mesh path")
        scale = float(f["object/scale"][()])
        transforms = np.asarray(f["grasps/transforms"][:], dtype=np.float64)
        labels = np.asarray(f["grasps/qualities/flex/object_in_gripper"][:]).reshape(-1)
    validate_transforms(transforms)
    mesh = load_mesh(mesh_path, scale)
    center, radius = mesh.bounds.mean(axis=0), float(mesh.extents.max()/2)
    if (scale != record["mesh_scale"] or radius != record["radius_m"] or
            not np.array_equal(center, record["center_m"])):
        raise ValueError("Mesh-only metric normalization does not match the raw scaled mesh")
    if len(labels) != record["n_source"] or int(labels.sum()) != record["n_positive"] or not np.isin(labels, [0, 1]).all():
        raise ValueError("Source label/count mismatch")
    with np.load(record["cache"]) as f:
        expected_p = ((transforms[:, :3, 3]-center)/radius).astype(np.float32)
        if (not np.array_equal(f["rotations"], transforms[:, :3, :3].astype(np.float32)) or
                not np.array_equal(f["translations"], expected_p) or not np.array_equal(f["labels"], labels)):
            raise ValueError("Cache poses/labels differ from original metric HDF5 transforms")
        rng = np.random.default_rng(int(hashlib.sha256(f"{seed}:{mesh_path.stem}:{scale}".encode()).hexdigest()[:16], 16))
        xyz, normals = sample_surface(mesh, len(f["cloud"]), rng)
        cloud = np.concatenate(((xyz-center)/radius, normals), axis=-1).astype(np.float32)
        if not np.array_equal(cloud, f["cloud"]):
            raise ValueError("Conditioning cloud is not the recorded matched mesh normalization")
    return {"object_id": record["object_id"], "raw_mesh_h5_cache_verified": True,
            "h5_mesh_reference": name, "mesh_scale": scale, "center_m": center.tolist(),
            "radius_m": radius, "source_rows": len(labels), "positive_rows": int(labels.sum()),
            "translation_scale_applied": False, "conditioning_reconstructed_exactly": True}


def prepare(grasps_root, mesh_root, destination, seed=2026, points=256, smoke_only=False, license_record=None,
            protocol="object_heldout"):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records, missing = [], []
    for h5path in sorted(Path(grasps_root).rglob("*.h5")):
        with h5py.File(h5path, "r") as f:
            name = f["object/file"][()]
            name = name.decode() if isinstance(name, bytes) else str(name)
            scale = float(f["object/scale"][()])
            mesh_path = safe_mesh_path(mesh_root, name)
            if not mesh_path.is_file():
                missing.append({"h5": str(h5path.resolve()), "mesh": str(mesh_path)})
                continue
            transforms = np.asarray(f["grasps/transforms"][:], dtype=np.float64)
            labels = np.asarray(f["grasps/qualities/flex/object_in_gripper"][:]).reshape(-1)
        if len(transforms) != len(labels) or not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Invalid labels in {h5path}")
        validation = validate_transforms(transforms)
        if not labels.any():
            missing.append({"h5": str(h5path.resolve()), "reason": "No positive FleX labels"})
            continue
        mesh = load_mesh(mesh_path, scale)
        center = mesh.bounds.mean(axis=0)
        radius = float(mesh.extents.max() / 2)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(f"Invalid mesh size: {mesh_path}")
        identity = mesh_path.stem
        rng = np.random.default_rng(int(hashlib.sha256(f"{seed}:{identity}:{scale}".encode()).hexdigest()[:16], 16))
        xyz, normals = sample_surface(mesh, points, rng)
        cloud = np.concatenate(((xyz-center) / radius, normals), axis=-1).astype(np.float32)
        cache = destination / "objects" / (h5path.stem + ".npz")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, rotations=transforms[:, :3, :3].astype(np.float32),
                            translations=((transforms[:, :3, 3]-center) / radius).astype(np.float32),
                            labels=labels.astype(np.uint8), cloud=cloud)
        records.append({"object_id": h5path.stem, "mesh_identity": identity,
                        "category": h5path.stem.split("_")[0], "h5_path": str(h5path.resolve()),
                        "h5_sha256": sha256(h5path), "mesh_path": str(mesh_path), "mesh_sha256": sha256(mesh_path),
                        "mesh_scale": scale, "center_m": center.tolist(), "radius_m": radius,
                        "cache": str(cache.resolve()), "cache_sha256": sha256(cache),
                        "n_source": len(labels), "n_positive": int(labels.sum()),
                        "mesh_watertight": bool(mesh.is_watertight), "mesh_faces": len(mesh.faces),
                        "source_se3_validation": validation})
    write_json(destination / "missing_assets.json", missing)
    if protocol == "within_object_heldout_pose":
        if not smoke_only:
            raise ValueError("The within-object pilot must be explicitly marked --smoke-only")
        groups = {s: sorted({r["mesh_identity"] for r in records}) for s in ("train", "val", "test")}
        for record in records:
            with np.load(record["cache"]) as f:
                record["pose_splits_source_indices"] = make_pose_split(np.flatnonzero(f["labels"] == 1), seed, record["object_id"])
            record["pose_split_sha256"] = hashlib.sha256(json.dumps(record["pose_splits_source_indices"], sort_keys=True).encode()).hexdigest()
            record["split"] = "within_object"
    elif protocol == "object_heldout":
        groups = make_split([r["mesh_identity"] for r in records], seed, smoke_only)
        for record in records:
            record["split"] = next(k for k, v in groups.items() if record["mesh_identity"] in v)
    else:
        raise ValueError(f"Unknown split protocol: {protocol}")
    license_evidence = json.loads(Path(license_record).read_text(encoding="utf-8")) if license_record else None
    manifest = {"schema": 2, "seed": seed, "smoke_only": smoke_only, "protocol": protocol,
                "protocol_label": "within-object held-out-pose pilot, NOT unseen-object ACRONYM" if protocol == "within_object_heldout_pose" else "object-held-out",
                "benchmark_ready": not smoke_only and license_evidence is not None,
                "source": "Actual ACRONYM HDF5 grasps and matched object meshes; no synthetic targets",
                "license_record": str(Path(license_record).resolve()) if license_record else None,
                "license_record_sha256": sha256(license_record) if license_record else None,
                "normalization": "mesh bbox center and half max extent, per-object; no grasp/test-label statistics. H5 grasp translations already in meters, scale applied ONLY to mesh.",
                "training_filter": "Only original object_in_gripper == 1 poses; labels retained for source provenance, never inferred for generated poses.",
                "split_rule": "Seeded per-object permutation of positive HDF5 source indices, 80/10/10 by floor; fixed mesh geometry shared across splits, no poses shared." if protocol == "within_object_heldout_pose" else "Seeded SHA256 ordering of unscaled mesh identity; all scales of one mesh stay together.",
                "split_groups": groups, "missing_asset_count": len(missing), "objects": records}
    write_json(destination / "manifest.json", manifest)
    return manifest


class PreparedData:
    def __init__(self, manifest_path, device="cpu", allow_smoke=False):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if not self.manifest["benchmark_ready"] and not allow_smoke:
            raise ValueError("Dataset is not benchmark-ready. --allow-smoke-data is required for integration tests.")
        self.records = self.manifest["objects"]
        self.within_object = self.manifest.get("protocol") == "within_object_heldout_pose"
        license_record = self.manifest.get("license_record")
        if license_record and sha256(license_record) != self.manifest["license_record_sha256"]:
            raise ValueError("License/provenance receipt changed")
        self.source_verification = [verify_record(r, self.manifest["seed"]) for r in self.records]
        self.device = torch.device(device)
        self.objects = []
        for record in self.records:
            if sha256(record["cache"]) != record["cache_sha256"]:
                raise ValueError("Prepared object hash changed")
            with np.load(record["cache"]) as f:
                positive = f["labels"] == 1
                source_indices = np.flatnonzero(positive)
                if self.within_object:
                    split = record["pose_splits_source_indices"]
                    validate_pose_split(split, source_indices)
                    if split != make_pose_split(source_indices, self.manifest["seed"], record["object_id"]):
                        raise ValueError("Pose split does not match the fixed source-row permutation")
                    digest = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
                    if digest != record["pose_split_sha256"]:
                        raise ValueError("Pose split checksum changed")
                    pose_indices = {s: torch.tensor(np.searchsorted(source_indices, split[s]), device=device) for s in split}
                else:
                    pose_indices = {s: torch.arange(len(source_indices), device=device) for s in ("train", "val", "test")}
                self.objects.append({"r": torch.as_tensor(f["rotations"][positive], device=device),
                                     "p": torch.as_tensor(f["translations"][positive], device=device),
                                     "source_indices": source_indices, "pose_indices": pose_indices,
                                     "cloud": torch.as_tensor(f["cloud"], device=device)})
        self.clouds = torch.stack([o["cloud"] for o in self.objects])
        self.scales = torch.tensor([[np.log(r["radius_m"]), np.log(0.082/r["radius_m"])] for r in self.records], device=device, dtype=torch.float32)
        self.splits = {s: list(range(len(self.records))) if self.within_object else
                       [i for i, r in enumerate(self.records) if r["split"] == s] for s in ("train", "val", "test")}

    def sample(self, size, rng):
        pool = self.splits["train"]
        ids = torch.tensor(pool, device=self.device)[torch.randint(len(pool), (size,), device=self.device, generator=rng)]
        r = torch.empty(size, 3, 3, device=self.device)
        p = torch.empty(size, 3, device=self.device)
        for i in ids.unique().tolist():
            mask = ids == i
            obj = self.objects[i]
            allowed = obj["pose_indices"]["train"]
            selected = allowed[torch.randint(len(allowed), (int(mask.sum()),), device=self.device, generator=rng)]
            r[mask], p[mask] = obj["r"][selected], obj["p"][selected]
        return ids, r, p
