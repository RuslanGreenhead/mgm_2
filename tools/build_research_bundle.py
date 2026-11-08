"""Create a code-and-manuscript archive without weights, raw datasets or CUDA."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DOCSRC = ROOT / "release" if (ROOT / "release/README.md").exists() else ROOT
DATE = "2026-11-03"
PREFIX = "MGM"
OUT = ROOT / "output/bundles" / f"MGM_Research_Package_{DATE}.zip"
CODE = {".py", ".pyi", ".ipynb", ".ps1", ".sh", ".bat", ".c", ".h", ".cpp", ".hpp", ".cc", ".cu", ".cuh", ".pyx", ".pxd", ".jl", ".m"}
CONFIG = {".yaml", ".yml", ".toml", ".ini", ".cfg", ".bib"}
PAPER = {".tex", ".bib", ".sty", ".bst", ".md", ".json"}
FORBIDDEN = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".pkl", ".pickle", ".npy", ".npz", ".h5", ".hdf5", ".dll", ".exe", ".whl", ".pyc", ".zip"}
PRUNE = {".git", ".conda", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "site-packages", "python-deps", "node_modules"}
MAX_RECORD = 2 * 1024 * 1024
PAYLOAD = {}
EXCLUDED = []


def digest(data):
    return hashlib.sha256(data).hexdigest()


def add(path, name=None, role="source"):
    path = Path(path)
    assert path.resolve().is_relative_to(ROOT), path
    name = name or path.relative_to(ROOT).as_posix()
    if name in PAYLOAD:
        assert PAYLOAD[name].get("source") == path
        return
    assert Path(name).suffix.lower() not in FORBIDDEN, name
    PAYLOAD[name] = {"source": path, "role": role}


def generated(name, value, role="documentation"):
    assert name not in PAYLOAD, name
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    PAYLOAD[name] = {"content": data, "role": role}


def json_text(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def walk(base):
    if not base.exists():
        return
    for directory, dirs, names in os.walk(base, followlinks=False):
        here = Path(directory)
        kept = []
        for name in sorted(dirs):
            path = here / name
            if name in PRUNE or not path.resolve().is_relative_to(ROOT) or path.is_symlink():
                EXCLUDED.append({"path": path.relative_to(ROOT).as_posix() + "/", "reason": "environment_vcs_cache_or_external_tree"})
            elif path.relative_to(ROOT).as_posix().startswith("paper/review"):
                EXCLUDED.append({"path": path.relative_to(ROOT).as_posix() + "/", "reason": "review_or_duplicate_smoke_tree"})
            else:
                kept.append(name)
        dirs[:] = kept
        for name in sorted(names):
            path = here / name
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                continue
            yield path


def is_notice(path):
    name = path.name.lower()
    return name.startswith(("license", "copying", "notice", "copyright", "citation", "readme", "requirements")) or name in {"thirdpartylicenses", "environment.yml", "setup.cfg", "pyproject.toml"}


def choose_sources():
    canonical = set()
    for folder in ("paper", "external", "outputs"):
        for path in walk(ROOT / folder):
            relative = path.relative_to(ROOT).as_posix()
            suffix = path.suffix.lower()
            selected, role = False, "source"
            if suffix in CODE:
                canonical.add(relative)
                selected = True
                role = "frozen_source" if folder == "outputs" else "source"
            elif folder == "paper":
                selected = suffix in PAPER or suffix in CONFIG or is_notice(path)
                if relative.startswith("paper/figures/") and suffix in {".pdf", ".png"}:
                    selected, role = True, "manuscript_figure"
                if relative == "paper/vendor/ne_110m_land.geojson":
                    selected, role = True, "public_domain_geographic_context"
            elif folder == "external":
                selected = suffix in CONFIG or is_notice(path) or (suffix == ".json" and "configs" in path.parts)
                role = "third_party_source_or_configuration"
            elif folder == "outputs":
                selected = suffix in {".json", ".csv", ".md", ".yaml", ".yml"} and path.stat().st_size <= MAX_RECORD
                role = "recorded_evidence_not_complete_artifacts"
                if relative.startswith(("outputs/comparison2026/paused_2026_10_26/", "outputs/comparison2026/visualizations/final_v1/test_002/")) and suffix in {".pdf", ".png"}:
                    selected, role = True, "comparison_visualization"
                if relative.startswith("outputs/comparison2026/grasp/publication_figures/seed") and suffix in {".pdf", ".png"}:
                    selected, role = True, "grasp_visualization"
            if selected and suffix not in FORBIDDEN:
                if suffix == ".json":
                    obj = json.loads(path.read_text(encoding="utf-8-sig"))
                    if isinstance(obj, dict) and {"state_dict", "model_state", "optimizer_state"} & obj.keys():
                        selected = False
                if selected:
                    add(path, role=role)
                    continue
            EXCLUDED.append({"path": relative, "bytes": path.stat().st_size,
                             "reason": "weights_or_raw_arrays" if suffix in FORBIDDEN else "raw_asset_log_build_product_or_nonselected_artifact"})
    assert canonical <= PAYLOAD.keys(), sorted(canonical - PAYLOAD.keys())
    return canonical


def add_article_history():
    for path in (ROOT / "output/pdf").glob("*.pdf"):
        add(path, role="article_pdf")
    base = ROOT / "paper/review/source_archive_smoke"
    if not base.is_dir():
        base = ROOT / "article_history/2026-10-24"
    if base.is_dir():
        for path in base.rglob("*"):
            if path.is_file() and (path.suffix.lower() in {".tex", ".bib", ".sty", ".bst", ".ps1", ".md"} or (path.parent.name == "figures" and path.suffix == ".pdf")):
                add(path, "article_history/2026-10-24/" + path.relative_to(base).as_posix(), "previous_manuscript_source")
    for name in ("current_snapshot_pdf_checks.json", "current_snapshot_package_checks.json", "current_snapshot_review.md"):
        path = ROOT / "paper/review" / name
        if not path.is_file():
            path = ROOT / "docs/verification" / name
        if path.is_file():
            add(path, "docs/verification/" + name, "recorded_manuscript_verification")
    path = ROOT / "output/manuscript/snapshot_2026-10-26_manifest.json"
    if path.is_file():
        add(path, role="article_snapshot_metadata")


def environment():
    observed = {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]}
    optional_dir = ROOT / "external/acronym-data/python-deps"
    optional = {d.metadata["Name"]: d.version for d in metadata.distributions(path=[str(optional_dir)]) if d.metadata["Name"]} if optional_dir.is_dir() else {}
    generated("environment/observed_versions.json", json_text({"python": sys.version, "platform": platform.platform(), "base_distributions": dict(sorted(observed.items())), "isolated_optional": dict(sorted(optional.items())), "installation_retested_from_scratch": False}))
    core = ("numpy", "scipy", "matplotlib", "pandas", "scikit-learn", "geoopt", "torchdiffeq", "trimesh", "libigl", "hydra-core", "pytorch-lightning", "torchmetrics", "biopython", "tqdm", "click", "omegaconf")
    lines = ["# Observed direct dependencies, not a universal lockfile.", "# Install the platform-appropriate torch==2.5.1 separately; see README."]
    for name in core:
        try:
            lines.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            lines.append(f"# {name}: not installed in the base environment")
    generated("environment/requirements-core-reference.txt", "\n".join(lines) + "\n")


def git_output(path, *args):
    completed = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, errors="replace", timeout=20)
    return completed.stdout.strip() if completed.returncode == 0 else None


def provenance():
    records = []
    for name in ("riemannian-fm", "riemannian-fm-official-clean", "kgdm-author", "riemannian-meanflow-author", "acronym-data/public"):
        path = ROOT / "external" / name
        records.append({"path": path.relative_to(ROOT).as_posix(), "git_head_if_available": git_output(path, "rev-parse", "HEAD"),
                        "tracked_changes_if_available": git_output(path, "diff", "--name-only", "HEAD"),
                        "license_files": [p.relative_to(path).as_posix() for p in path.iterdir() if p.is_file() and p.name.lower().startswith(("license", "copying", "notice", "thirdparty"))]})
    generated("docs/UPSTREAM_PROVENANCE.json", json_text(records))


def source_index():
    lines = ["# Static Python source index", "", "Generated by AST inspection only; no modules or launchers were executed.",
             "Frozen snapshots are included in the manifest but omitted from this human index to avoid duplicate listings.", "", "| Source | Purpose / first docstring line | Literal CLI flags |", "| --- | --- | --- |"]
    syntax_errors = []
    for name, item in sorted(PAYLOAD.items()):
        if not name.endswith(".py") or not name.startswith(("paper/", "external/riemannian-fm/", "tools/")) or "source" not in item["role"]:
            continue
        path = item.get("source")
        if not path:
            continue
        try:
            tree = ast.parse(path.read_bytes(), filename=name)
        except (SyntaxError, UnicodeDecodeError) as error:
            syntax_errors.append({"path": name, "error": str(error)})
            continue
        description = (ast.get_docstring(tree) or "").splitlines()
        description = description[0].replace("|", "/") if description else "Module / definitions"
        flags = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                flags.extend(a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str))
        lines.append(f"| `{name}` | {description[:170]} | " + ", ".join(f"`{s}`" for s in dict.fromkeys(flags)) + " |")
    generated("docs/SOURCE_INDEX.md", "\n".join(lines) + "\n")
    return syntax_errors


def notebook_copy(content):
    data = json.loads(content.decode("utf-8-sig"))
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cell.pop("attachments", None)
    data.get("metadata", {}).pop("widgets", None)
    return json_text(data).encode("utf-8")


def main():
    assert OUT.resolve().is_relative_to(ROOT / "output/bundles")
    canonical = choose_sources()
    add_article_history()
    add(DOCSRC / "README.md", "README.md", "package_documentation")
    for path in (DOCSRC / "docs").glob("*.md"):
        add(path, "docs/" + path.name, "package_documentation")
    helper_dir = DOCSRC / "tools"
    for path in helper_dir.glob("*.py"):
        add(path, "tools/" + path.name, "package_source")
    add(Path(__file__), "tools/build_research_bundle.py", "package_source")
    environment()
    provenance()
    syntax_errors = source_index()
    assert not syntax_errors, syntax_errors
    current_pdf = ROOT / "output/pdf/Midpoint_Generative_Models_for_General_Geometries_2026-10-26.pdf"
    article_sha = digest(current_pdf.read_bytes())
    original = json.loads((ROOT / "output/manuscript/snapshot_2026-10-26_manifest.json").read_text())
    assert article_sha == original["pdf_sha256"]
    partial = json.loads((ROOT / "outputs/comparison2026/paused_2026_10_26/partial_audit_manifest.json").read_text())
    assert partial["counts"] == {"completed": 18, "interrupted": 2, "not_started": 88}
    exclusions = {"skipped_top_level_trees": [".conda/", "tmp/", "output/manuscript/ (nested archives)", "outputs/logs/ text logs not selected"],
                  "max_text_record_bytes": MAX_RECORD, "excluded": EXCLUDED}
    generated("EXCLUSIONS.json", json_text(exclusions))
    generated("docs/PACKAGE_CHECKS.json", json_text({"scope": "pre-archive selection checks; final CRC/SHA checks are in the external .checks.json",
        "all_discovered_canonical_code_selected": True, "canonical_code_files": len(canonical),
        "static_index_syntax_errors": syntax_errors, "article_pdf_unchanged": True, "article_pdf_sha256": article_sha,
        "recorded_evidence_status": partial["counts"], "weights_and_raw_arrays_selected": False,
        "gpu_used": False, "training_launched": False, "fresh_dependency_installation_tested": False}))
    smoke = ROOT / "release/SMOKE_TESTS.json"
    if not smoke.is_file():
        smoke = ROOT / "docs/SMOKE_TESTS.json"
    if smoke.is_file():
        add(smoke, "docs/SMOKE_TESTS.json", "recorded_package_cpu_tests")
    test_log = ROOT / "tmp/bundle_all_cpu_tests.log"
    if test_log.is_file():
        add(test_log, "docs/verification/package_cpu_tests.log", "recorded_package_cpu_tests")
    keys = list(PAYLOAD)
    assert len({s.casefold() for s in keys}) == len(keys), "Case-insensitive path collision"
    assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in keys)
    records, credentials = [], []
    patterns = [re.compile(rb"AKIA[0-9A-Z]{16}"), re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}"), re.compile(rb"sk-proj-[A-Za-z0-9_-]{30,}")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".building.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, item in sorted(PAYLOAD.items()):
            path = item.get("source")
            content = path.read_bytes() if path else item["content"]
            source_sha = digest(content)
            transform = "none"
            if name.endswith(".ipynb"):
                content = notebook_copy(content)
                transform = "notebook_outputs_counters_attachments_widgets_removed"
            if Path(name).suffix in CODE | CONFIG | {".json", ".md", ".txt"} and any(pattern.search(content) for pattern in patterns):
                credentials.append(name)
            record = {"path": name, "bytes": len(content), "sha256": digest(content), "role": item["role"], "transformation": transform}
            if path:
                record.update(original_source=path.relative_to(ROOT).as_posix(), original_sha256=source_sha)
            records.append(record)
            archive.writestr(f"{PREFIX}/{name}", content)
        assert not credentials, {"possible_credentials_files_only": credentials}
        manifest = {"package_date": DATE, "article_snapshot_date": "2026-10-26", "archive_root": PREFIX,
                    "file_count_excluding_manifest": len(records), "uncompressed_bytes_excluding_manifest": sum(r["bytes"] for r in records),
                    "code_completeness_scope": "Canonical paper/external/outputs code; environments, caches, duplicate review/tmp trees excluded",
                    "gpu_used_to_package": False, "trained_weights_included": False, "raw_datasets_included": False,
                    "files": records}
        archive.writestr(f"{PREFIX}/MANIFEST.json", json_text(manifest))
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(records) + 1
        for record in records:
            content = archive.read(f"{PREFIX}/{record['path']}")
            assert digest(content) == record["sha256"] and len(content) == record["bytes"]
            assert Path(record["path"]).suffix.lower() not in FORBIDDEN
    temporary.replace(OUT)
    archive_sha = digest(OUT.read_bytes())
    report = {"passed": True, "archive": OUT.relative_to(ROOT).as_posix(), "sha256": archive_sha,
              "archive_bytes": OUT.stat().st_size, "files": len(records) + 1, "canonical_code_files": len(canonical),
              "roles": dict(Counter(r["role"] for r in records)), "crc_passed": True, "all_payload_hashes_passed": True,
              "possible_credential_pattern_hits": credentials, "gpu_used": False, "training_launched": False}
    OUT.with_suffix(".checks.json").write_text(json_text(report), encoding="utf-8")
    OUT.with_suffix(".sha256").write_text(archive_sha + "  " + OUT.name + "\n", encoding="ascii")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
