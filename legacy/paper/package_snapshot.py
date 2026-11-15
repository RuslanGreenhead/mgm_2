"""Package the compiled research draft and self-contained TeX assets."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TAG = "2026-10-26"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    checks = json.loads((PAPER / "review/current_snapshot_pdf_checks.json").read_text())
    assert checks["pdf_sha256"] == digest(PAPER / "main.pdf")
    assert checks["main_pages"] <= 8
    pdf = ROOT / f"output/pdf/Midpoint_Generative_Models_for_General_Geometries_{TAG}.pdf"
    archive = ROOT / f"output/manuscript/Midpoint_Generative_Models_for_General_Geometries_{TAG}_TeX.zip"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    files = [PAPER / name for name in ("main.tex", "references.bib", "build.ps1", "README.md")]
    files += list((PAPER / "sections").glob("*.tex")) + list((PAPER / "generated").glob("*.tex"))
    files += list((PAPER / "figures").glob("*.pdf")) + list(PAPER.glob("*.sty")) + list(PAPER.glob("*.bst"))
    files = sorted(set(files))
    for path in files:
        assert path.is_relative_to(PAPER) and path.is_file()
        if path.suffix in (".tex", ".bib", ".sty", ".bst", ".md", ".ps1"):
            source = path.read_text(encoding="utf-8", errors="replace")
            assert not any(term in source for term in ("C:/Users/Alexander", "C:\\Users\\Alexander", "E:/Codex Projects")), path
    for path in PAPER.glob("*.sty"):
        assert digest(path) == digest(PAPER / "vendor/icml2026" / path.name), path
    manifest = {"snapshot": TAG, "description": "ICML-format research draft, not a completed full benchmark or acceptance claim",
                "pages": checks["pages"], "main_pages": checks["main_pages"], "pdf_sha256": checks["pdf_sha256"],
                "sources": {p.relative_to(PAPER).as_posix(): digest(p) for p in files},
                "build": "powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1",
                "requires_gpu_for_compilation": False}
    prefix = f"MGM_General_Geometries_{TAG}"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in files:
            z.write(path, f"{prefix}/{path.relative_to(PAPER).as_posix()}")
        z.writestr(f"{prefix}/source_manifest.json", json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(PAPER / "main.pdf", pdf)
    record = {"pdf": str(pdf.relative_to(ROOT)), "tex_archive": str(archive.relative_to(ROOT)),
              "pdf_sha256": digest(pdf), "archive_sha256": digest(archive), "source_files": len(files),
              "pages": checks["pages"], "main_pages": checks["main_pages"]}
    (ROOT / f"output/manuscript/snapshot_{TAG}_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
