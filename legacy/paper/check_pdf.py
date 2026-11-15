"""Render the MiKTeX PDF and record reproducible layout/metadata checks."""
from pathlib import Path
import json
import hashlib
import fitz
from PIL import Image,ImageDraw

PAPER=Path(__file__).resolve().parent
pdf=PAPER/"main.pdf"
out=PAPER/"review/rendered"
out.mkdir(parents=True,exist_ok=True)
for old in out.glob("page-*"):
    if old.suffix in {".png",".txt"}:old.unlink()
doc=fitz.open(pdf)
pages=[]
thumbs=[]
reference_page=None
appendix_page=None
fonts={}
for i,page in enumerate(doc):
    text=page.get_text()
    (out/f"page-{i+1:02d}.txt").write_text(text,encoding="utf-8")
    if reference_page is None and "References" in text:reference_page=i+1
    if appendix_page is None and "A. Proofs" in text:appendix_page=i+1
    pix=page.get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False)
    pix.save(out/f"page-{i+1:02d}.png")
    thumb=Image.open(out/f"page-{i+1:02d}.png").convert("RGB")
    thumb.thumbnail((275,356))
    tile=Image.new("RGB",(295,386),"#ededed")
    tile.paste(thumb,((295-thumb.width)//2,10))
    ImageDraw.Draw(tile).text((12,369),f"Page {i+1}",fill="black")
    thumbs.append(tile)
    bad=[]
    for word in page.get_text("words"):
        if word[0]<0 or word[1]<0 or word[2]>page.rect.width+.5 or word[3]>page.rect.height+.5:bad.append(word[4])
    for f in page.get_fonts(full=True):fonts[str(f[0])]={"type":f[2],"name":f[3],"extension":f[1]}
    pages.append({"page":i+1,"width_pt":page.rect.width,"height_pt":page.rect.height,
                  "words_outside_page":bad,"text_characters":len(text)})
cols=4
sheet=Image.new("RGB",(295*cols,386*((len(thumbs)+cols-1)//cols)),"white")
for i,tile in enumerate(thumbs):sheet.paste(tile,((i%cols)*295,(i//cols)*386))
sheet.save(out/"contact_sheet.png")
report={"pages":len(doc),"bytes":pdf.stat().st_size,"metadata":doc.metadata,
        "pdf_sha256":hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "first_reference_page":reference_page,"first_appendix_page":appendix_page,
        "fonts":list(fonts.values()),"type3_fonts":[f for f in fonts.values() if f["type"]=="Type3"],
        "page_audit":pages}
report["main_text_with_floats_pages"]=reference_page-1 if reference_page else None
alltext="\n".join(p.get_text() for p in doc)
report["unresolved_markers"]=[m for m in ["AUTHORERR","??","TODO","TBD"] if m in alltext]
report["page_cap_passed"]=bool(reference_page and reference_page-1<=8)
assert report["page_cap_passed"],report["main_text_with_floats_pages"]
assert not report["unresolved_markers"],report["unresolved_markers"]
assert not report["type3_fonts"]
assert not any(p["words_outside_page"] for p in pages)
(PAPER/"review/pdf_checks.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k not in ("fonts","page_audit")},indent=2))
