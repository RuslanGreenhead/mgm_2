"""Read-only PDF checks and a contact sheet; run with bundled PDF Python."""
import hashlib
import json
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper/main.pdf"
RENDER = ROOT / "tmp/pdfs/current_2026_10_26"


def main():
    doc = fitz.open(PDF)
    log = (ROOT / "paper/main.log").read_text(errors="replace")
    starts = [i for i, page in enumerate(doc) if "Impact Statement" in page.get_text()]
    assert starts and starts[0] <= 8
    assert all(abs(p.rect.width - 612) < .1 and abs(p.rect.height - 792) < .1 for p in doc)
    assert "Overfull" not in log and "undefined" not in log.lower()
    assert "AUTHORERR" not in log
    fonts = {font[2] for p in doc for font in p.get_fonts()}
    assert "Type3" not in fonts
    outside = []
    for i, page in enumerate(doc):
        for word in page.get_text("words"):
            if word[0] < -.5 or word[1] < -.5 or word[2] > page.rect.width + .5 or word[3] > page.rect.height + .5:
                outside.append((i + 1, word[:5]))
    assert not outside, outside[:5]
    text = "\n".join(p.get_text() for p in doc)
    assert not any(s in text for s in ("C:/Users/", "C:\\Users\\", "E:\\Codex Projects", "??"))
    for term in ("13.135", "11.957", "4.265", "71.00", "18/108", "88", "unmeasured"):
        assert term in text, term
    width, height, cols = 204, 282, 5
    sheet = Image.new("RGB", (width * cols, height * ((len(doc) + cols - 1) // cols)), "#c7cdd1")
    draw = ImageDraw.Draw(sheet)
    for i in range(len(doc)):
        path = RENDER / f"page-{i + 1:02}.png"
        with Image.open(path) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((width - 12, height - 20))
            x, y = (i % cols) * width + 6, (i // cols) * height + 18
            sheet.paste(thumb, (x, y))
            draw.text((x, y - 15), str(i + 1), fill="black")
    sheet.save(RENDER / "contact_sheet.png")
    report = {"pdf_sha256": hashlib.sha256(PDF.read_bytes()).hexdigest(), "pages": len(doc),
              "main_pages": starts[0], "us_letter_all_pages": True, "type3_fonts": False,
              "overfull_boxes": False, "undefined_references": False, "out_of_page_words": outside,
              "private_paths_in_pdf": False, "file_bytes": PDF.stat().st_size, "metadata": doc.metadata,
              "contact_sheet": str((RENDER / "contact_sheet.png").relative_to(ROOT)),
              "new_result_pages": [i + 1 for i, p in enumerate(doc) if any(s in p.get_text() for s in ("Completed modern one-step", "Completed SE(3) pilot on", "Recorded validation curves", "Per-object test results"))]}
    (ROOT / "paper/review/current_snapshot_pdf_checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
