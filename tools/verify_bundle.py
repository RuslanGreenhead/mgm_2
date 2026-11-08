"""Verify a source bundle with the standard library; never imports experiments."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".pkl", ".pickle",
             ".npy", ".npz", ".h5", ".hdf5", ".dll", ".exe", ".whl", ".pyc", ".zip"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syntax", action="store_true", help="Also parse Python sources; do not execute them")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    seen, parsed = set(), 0
    for item in manifest["files"]:
        name = item["path"]
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and name not in seen, name
        assert path.suffix.lower() not in FORBIDDEN, name
        assert not any(p in {".git", ".conda", "__pycache__", "site-packages", "python-deps"} for p in Path(name).parts), name
        content = path.read_bytes()
        assert len(content) == item["bytes"], name
        assert hashlib.sha256(content).hexdigest() == item["sha256"], name
        if args.syntax and path.suffix == ".py":
            ast.parse(content, filename=name)
            parsed += 1
        seen.add(name)
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()}
    assert actual == seen | {"MANIFEST.json"}, sorted(actual ^ (seen | {"MANIFEST.json"}))[:10]
    print(json.dumps({"passed": True, "files_verified": len(seen), "python_parsed": parsed,
                      "gpu_used": False, "training_launched": False}))


if __name__ == "__main__":
    main()
