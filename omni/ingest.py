"""Turn any file into pages of text lines with normalized boxes. OCR only when a page has no usable text layer."""
from __future__ import annotations
import hashlib, io, json, os, re, zipfile
from dataclasses import dataclass, field
import fitz
import numpy as np
from . import legacy_bn, ocr_bn
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "data", "cache")
OCR_DIR = os.path.join(HERE, "..", "ocr_models")
os.makedirs(CACHE, exist_ok=True)
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
_ocr = None


def ocr_engine():
    global _ocr
    if _ocr is None:
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import LangRec
        det, rec = os.path.join(OCR_DIR, "ch_PP-OCRv5_det_mobile.onnx"), os.path.join(OCR_DIR, "PP-OCRv6_rec_small.onnx")
        if os.path.exists(det) and os.path.exists(rec):  # pinned models
            _ocr = RapidOCR(params={"Det.model_path": det, "Rec.model_path": rec, "Rec.lang_type": LangRec.LATIN})
        else:  # fresh checkout: RapidOCR fetches its default models itself
            _ocr = RapidOCR()
    return _ocr


@dataclass
class Page:
    n: int                                      # 1-based
    lines: list = field(default_factory=list)   # [(text, (x0, y0, x1, y1) normalized 0-1)]
    source: str = "none"                        # text | ocr | none (needs ocr)
    scores: list = field(default_factory=list)  # per-line OCR confidence (only for OCR'd pages); low = probably handwriting or a poor scan

    @property
    def text(self):
        return "\n\n".join(t for t, _ in self.lines)  # a blank line marks a block boundary


@dataclass
class Doc:
    id: str
    name: str
    path: str
    kind: str
    pages: list
    sha: str
    lang: str = ""   # OCR language hint: "" / "auto" (decide per page), "bn" (Bengali + English), "en" (Latin only)


def _pdf_lines(pg):
    W, H = pg.rect.width, pg.rect.height
    out = []
    for b in pg.get_text("blocks"):
        t = "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in b[4].splitlines() if ln.strip())
        if t and b[6] == 0:
            out.append((t, (b[0] / W, b[1] / H, b[2] / W, b[3] / H)))
    # Bangladeshi PDFs are often typeset with a legacy Bijoy font: the text layer is Latin-looking gibberish. Convert it to real Bengali.
    if out and os.environ.get("OMNI_BN", "1") == "1" and legacy_bn.is_legacy(" ".join(t for t, _ in out)):
        out = [(legacy_bn.convert(t), bx) for t, bx in out]
    return out


def _scan_ratio(pg):
    """Largest share of the page covered by a single embedded image (1.0 = a full-page scan)."""
    area = pg.rect.width * pg.rect.height
    best = 0.0
    try:
        for img in pg.get_images(full=True):
            for r in pg.get_image_rects(img[0]):
                best = max(best, (r.width * r.height) / area)
    except Exception:
        pass
    return best


def _paginate(lines, per=45):
    pages = [Page(i // per + 1, [(l.strip(), (0, 0, 1, 1)) for l in lines[i:i + per]], "text") for i in range(0, len(lines), per)]
    return pages or [Page(1, [], "text")]


def load(path: str) -> Doc:
    data = open(path, "rb").read()
    sha = hashlib.sha1(data).hexdigest()[:16]
    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lower()
    if ext == ".pdf":
        pdf = fitz.open(stream=data, filetype="pdf")
        pages = []
        for i, pg in enumerate(pdf):
            ln = _pdf_lines(pg)
            good = sum(len(t) for t, _ in ln) >= 40
            # a page that is mostly one big image is a scan/photo: any text layer on it is someone else's OCR, so untrusted
            src = ("scan-text" if _scan_ratio(pg) >= 0.6 else "text") if good else "none"
            pages.append(Page(i + 1, ln if good else [], src))
        kind = "pdf"
    elif ext in IMG_EXT:
        pages, kind = [Page(1, [], "none")], "image"
    elif ext == ".docx":
        xml = zipfile.ZipFile(io.BytesIO(data)).read("word/document.xml").decode("utf8", "ignore")
        paras = [re.sub(r"<[^>]+>", "", p) for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)]
        pages, kind = _paginate([p for p in paras if p.strip()]), "docx"
    else:  # txt, md, csv, json, code ...
        pages, kind = _paginate([l for l in data.decode("utf8", "ignore").splitlines() if l.strip()]), "text"
    doc = Doc(sha, name, os.path.abspath(path), kind, pages, sha)
    cf = os.path.join(CACHE, f"{sha}.json")  # reuse OCR done in earlier sessions
    if os.path.exists(cf):
        for n, lines in json.load(open(cf)).items():
            p = doc.pages[int(n) - 1]
            p.lines, p.source = [(t, tuple(b)) for t, b, *_ in lines], "ocr"
            p.scores = [(rest[0] if rest else 1.0) for _, _, *rest in lines]
    return doc


_DATE_NOISE = re.compile(r"\b(\d{1,2})/(\d{2})[12lI|](20\d\d)\b")


def _fix_dates(text):
    """OCR often reads the second slash of dd/mm/yyyy as a digit or bar ('24/0212026'): restore it."""
    return _DATE_NOISE.sub(lambda m: f"{m.group(1)}/{m.group(2)}/{m.group(3)}", text)


def _rows(boxes):
    """Group OCR boxes into text rows (handwriting sits a few pixels off the printed label's baseline),
    then read each row left to right, so 'Applicant Name:' and its handwritten value end up on one line."""
    if not boxes:
        return [], []
    hs = sorted(b[3] - b[1] for _, b, _ in boxes)
    tol = 0.6 * hs[len(hs) // 2]
    boxes = sorted(boxes, key=lambda tb: (tb[1][1] + tb[1][3]) / 2)
    rows, cur, cur_y = [], [], None
    for t, b, sc in boxes:
        yc = (b[1] + b[3]) / 2
        if cur and abs(yc - cur_y) > tol:
            rows.append(cur)
            cur = []
        cur.append((t, b, sc))
        cur_y = sum((bb[1] + bb[3]) / 2 for _, bb, _ in cur) / len(cur)
    if cur:
        rows.append(cur)
    out, scores = [], []
    for row in rows:
        row.sort(key=lambda tb: tb[1][0])
        box = (min(b[0] for _, b, _ in row), min(b[1] for _, b, _ in row), max(b[2] for _, b, _ in row), max(b[3] for _, b, _ in row))
        out.append((" ".join(t for t, _, _ in row), box))
        scores.append(min(sc for _, _, sc in row))  # a row is only as trustworthy as its weakest piece
    return out, scores


def render(doc: Doc, n: int, dpi=150) -> Image.Image:
    if doc.kind == "image":
        return Image.open(doc.path).convert("RGB")
    if doc.kind != "pdf":
        raise ValueError("only PDFs and images have a visual page")
    pix = fitz.open(doc.path)[n - 1].get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.w, pix.h), pix.samples)


def run_ocr(doc: Doc, n: int) -> Page:
    """OCR one page (the caller has already paid the ocr_pages budget) and cache the result on disk."""
    page = doc.pages[n - 1]
    if page.source == "ocr":
        return page
    pil = render(doc, n)
    img = np.array(pil)
    H, W = img.shape[:2]
    lang = (doc.lang or os.environ.get("OMNI_OCR_LANG", "auto")).lower()
    lines = scores = None
    if lang == "bn" and ocr_bn.available():  # told it is Bengali: go straight to the Bengali reader
        rows_bn = ocr_bn.repair_item_numbers(ocr_bn.read(pil)[0])
        lines, scores = [(t, b) for t, b, _ in rows_bn], [c for _, _, c in rows_bn]
    if lines is None:
        r = ocr_engine()(img)
        boxes = []
        for box, txt, sc in zip(r.boxes if r.boxes is not None else [], r.txts or [], r.scores or []):
            if txt.strip() and sc > 0.3:
                xs, ys = [p[0] for p in box], [p[1] for p in box]
                boxes.append((txt.strip(), tuple(float(v) for v in (min(xs) / W, min(ys) / H, max(xs) / W, max(ys) / H)), float(sc)))
        rows, scores = _rows(boxes)
        lines = [(_fix_dates(t), b) for t, b in rows]
        # The Latin OCR has no Bengali recogniser: on a Bengali page it is unsure of most lines (measured median confidence 0.67, against 0.99 on English pages).
        raw = [float(x) for x in (r.scores or [])]
        if lang == "auto" and len(raw) >= 6 and ocr_bn.available() and float(np.median(raw)) < 0.85 and sum(x < 0.8 for x in raw) / len(raw) >= 0.4:
            rows_bn = ocr_bn.repair_item_numbers(ocr_bn.read(pil)[0])
            if rows_bn and ocr_bn.bengali_share(" ".join(t for t, _, _ in rows_bn)) >= 0.3:
                lines, scores = [(t, b) for t, b, _ in rows_bn], [c for _, _, c in rows_bn]
    page.lines, page.source, page.scores = lines, "ocr", scores
    cf = os.path.join(CACHE, f"{doc.sha}.json")
    cur = json.load(open(cf)) if os.path.exists(cf) else {}
    cur[str(n)] = [[t, list(b), round(sc, 3)] for (t, b), sc in zip(lines, scores)]
    with open(cf + ".tmp", "w") as fh:  # write-then-rename so a crash can't leave a corrupt cache
        json.dump(cur, fh)
    os.replace(cf + ".tmp", cf)
    return page
