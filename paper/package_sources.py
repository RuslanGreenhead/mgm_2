"""Package the checked manuscript and editable TeX, without training data or old drafts."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

PAPER=Path(__file__).resolve().parent
ROOT=PAPER.parent
OUT=ROOT/"output"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(path,files,base):
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as z:
        for source in sorted(files):z.write(source,source.relative_to(base).as_posix())


def main():
    audit=json.loads((PAPER/"review/pdf_checks.json").read_text())
    assert audit["pdf_sha256"]==sha(PAPER/"main.pdf")
    assert audit["page_cap_passed"] and not audit["unresolved_markers"]
    verified=ROOT/"outputs/paper_icml2026/controlled_independent/verification.json"
    assert json.loads(verified.read_text())["all_checks_passed"]
    (OUT/"pdf").mkdir(parents=True,exist_ok=True)
    final=OUT/"pdf/Midpoint_Generative_Models_for_General_Geometries.pdf"
    shutil.copyfile(PAPER/"main.pdf",final)
    sources=[PAPER/name for name in ["main.tex","references.bib","icml2026.sty","icml2026.bst",
             "algorithm.sty","algorithmic.sty","fancyhdr.sty","build.ps1","README.md"] if (PAPER/name).exists()]
    sources+=list((PAPER/"sections").glob("*.tex"))+list((PAPER/"generated").glob("*.tex"))
    sources+=list((PAPER/"figures").glob("*.pdf"))
    sourcezip=OUT/"manuscript/ICML_MGM_TeX_sources.zip"
    archive(sourcezip,sources,PAPER)
    figurezip=OUT/"figures/MGM_high_resolution_figures.zip"
    archive(figurezip,list((PAPER/"figures").glob("*.png")),PAPER/"figures")
    manifest={"pdf":str(final.relative_to(ROOT)),"pdf_sha256":sha(final),
        "tex_archive":str(sourcezip.relative_to(ROOT)),"tex_archive_sha256":sha(sourcezip),
        "figure_archive":str(figurezip.relative_to(ROOT)),"figure_archive_sha256":sha(figurezip),
        "pages":audit["pages"],"main_text_pages":audit["main_text_with_floats_pages"],
        "source_file_sha256":{p.relative_to(PAPER).as_posix():sha(p) for p in sources},
        "primary_verification_sha256":sha(verified),
        "scope":"Complete ICML-format research draft; not a claim of submission readiness or demonstrated practical superiority. TeX archive does not contain datasets or full experiment code."}
    (OUT/"manuscript/delivery_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"pdf":str(final),"sources":str(sourcezip),"figures":str(figurezip),"pages":audit["pages"]}))


if __name__=="__main__":main()
