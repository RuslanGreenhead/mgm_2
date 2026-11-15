"""Read and validate dataset manifests for new experiment protocols."""

import hashlib
import json
from pathlib import Path


def read_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    arrays = path.parent / manifest["arrays_file"]
    if not arrays.is_file():
        raise FileNotFoundError(arrays)
    actual = hashlib.sha256(arrays.read_bytes()).hexdigest()
    if actual != manifest["arrays_sha256"]:
        raise ValueError(f"Dataset hash mismatch: {arrays}")
    return manifest, arrays
