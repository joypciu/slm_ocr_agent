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
# detection at 960 px instead of the library's 736: on FUNSD/CORD word ground truth (dev set) exact words 84.7% -> 87.4%, numbers 87.7% -> 89.6%.
# 1280 px read a little more on those images (88.5% words) but garbled two clean typed lines of a real 150-dpi letter that 736 and 960 read
# perfectly, and costs ~20% more time; 1600 px: +47% time.
DET_DEFAULTS = {"OMNI_DET_SIDE": "960"}


def ocr_engine():
    global _ocr
    if _ocr is None:
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import LangRec
        det, rec = os.path.join(OCR_DIR, "ch_PP-OCRv5_det_mobile.onnx"), os.path.join(OCR_DIR, "PP-OCRv6_rec_small.onnx")
        if os.environ.get("OMNI_REC") == "en":  # experiment switch: the English recogniser as the primary
            rec = os.path.join(OCR_DIR, "en_PP-OCRv5_rec_mobile.onnx")
        if os.path.exists(det) and os.path.exists(rec):  # pinned models
            params = {"Det.model_path": det, "Rec.model_path": rec, "Rec.lang_type": LangRec.EN if "en_" in rec else LangRec.LATIN}
            for env, key, cast in (("OMNI_DET_SIDE", "Det.limit_side_len", int), ("OMNI_DET_UNCLIP", "Det.unclip_ratio", float),
                                   ("OMNI_DET_BOX", "Det.box_thresh", float), ("OMNI_DET_THRESH", "Det.thresh", float)):
                if os.environ.get(env, DET_DEFAULTS.get(env)):
                    params[key] = cast(os.environ.get(env, DET_DEFAULTS.get(env)))
            _ocr = RapidOCR(params=params)
        else:  # fresh checkout: RapidOCR fetches its default models itself
            _ocr = RapidOCR()
    return _ocr


_second = None
REREAD_BELOW = 0.90   # boxes the page OCR is unsure of (handwriting, poor scans) get a second reading


def second_reader():
    """A different recognition model (English PP-OCRv5) for a second opinion on unsure boxes; None when its file is not installed."""
    global _second
    if _second is None:
        path = os.path.join(OCR_DIR, "en_PP-OCRv5_rec_mobile.onnx")
        if not os.path.exists(path):
            _second = False
        else:
            from rapidocr import RapidOCR
            from rapidocr.utils.typings import LangRec
            _second = RapidOCR(params={"Det.model_path": os.path.join(OCR_DIR, "ch_PP-OCRv5_det_mobile.onnx"), "Rec.model_path": path,
                                       "Rec.lang_type": LangRec.EN, "Global.use_cls": False})
    return _second or None


def _reread(pil, boxes):
    """Re-recognise each unsure box with the second model (crop + white border) and keep whichever reading is more confident.
    Measured on 20 real handwriting crops: exact readings 11 -> 14; on FUNSD/CORD word ground truth (dev set) exact words +0.6 and numbers +0.9 points,
    numbers misread 8.6% -> 7.7%. On by default; OMNI_REREAD=0 turns it off."""
    if os.environ.get("OMNI_REREAD", "1") != "1":
        return boxes
    eng = second_reader()
    if eng is None:
        return boxes
    from PIL import ImageOps
    W, H = pil.size
    out = []
    for t, b, sc in boxes:
        if sc < REREAD_BELOW:
            x0, y0, x1, y1 = int(b[0] * W) - 2, int(b[1] * H) - 2, int(b[2] * W) + 2, int(b[3] * H) + 2
            crop = ImageOps.expand(pil.crop((max(0, x0), max(0, y0), min(W, x1), min(H, y1))), border=(12, 8), fill="white")
            r = eng(np.array(crop), use_det=False, use_cls=False, use_rec=True)
            alt = None
            if r.txts and r.txts[0].strip():
                t2, s2 = r.txts[0].strip(), float(r.scores[0])
                if _alnum(t2) != _alnum(t):
                    alt = t if s2 > sc else t2   # the readers disagree: keep the other reading so the answer can be flagged
                if s2 > sc:
                    t, sc = t2, s2
            out.append((t, b, sc, alt))
            continue
        out.append((t, b, sc, None))
    return out


def _alnum(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


@dataclass
class Page:
    n: int                                      # 1-based
    lines: list = field(default_factory=list)   # [(text, (x0, y0, x1, y1) normalized 0-1)]
    source: str = "none"                        # text | ocr | none (needs ocr)
    scores: list = field(default_factory=list)  # per-line OCR confidence (only for OCR'd pages); low = probably handwriting or a poor scan
    segs: list = field(default_factory=list)    # per line, the OCR boxes it was merged from: [[text, x0, x1], ...] (one box ~ one field); [] = unknown

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
            p.segs = [(rest[1] if len(rest) > 1 else None) for _, _, *rest in lines]
    return doc


_DATE_NOISE = re.compile(r"\b(\d{1,2})/(\d{2})[12lI|](20\d\d)\b")


def _fix_dates(text):
    """OCR often reads the second slash of dd/mm/yyyy as a digit or bar ('24/0212026'): restore it."""
    return _DATE_NOISE.sub(lambda m: f"{m.group(1)}/{m.group(2)}/{m.group(3)}", text)


def _rows(boxes):
    """Group OCR boxes into text rows (handwriting sits a few pixels off the printed label's baseline),
    then read each row left to right, so 'Applicant Name:' and its handwritten value end up on one line."""
    if not boxes:
        return [], [], []
    hs = sorted(b[3] - b[1] for _, b, *_ in boxes)
    tol = 0.6 * hs[len(hs) // 2]
    boxes = sorted(boxes, key=lambda tb: (tb[1][1] + tb[1][3]) / 2)
    rows, cur, cur_y = [], [], None
    for t, b, sc, *more in boxes:
        yc = (b[1] + b[3]) / 2
        if cur and abs(yc - cur_y) > tol:
            rows.append(cur)
            cur = []
        cur.append((t, b, sc, more[0] if more else None))
        cur_y = sum((bb[1] + bb[3]) / 2 for _, bb, *_ in cur) / len(cur)
    if cur:
        rows.append(cur)
    out, scores, segs = [], [], []
    for row in rows:
        row.sort(key=lambda tb: tb[1][0])
        box = (min(b[0] for _, b, *_ in row), min(b[1] for _, b, *_ in row), max(b[2] for _, b, *_ in row), max(b[3] for _, b, *_ in row))
        out.append((" ".join(t for t, *_ in row), box))
        scores.append(min(x[2] for x in row))  # a row is only as trustworthy as its weakest piece
        segs.append([[_fix_dates(t), round(b[0], 4), round(b[2], 4), round(sc, 3), alt] for t, b, sc, alt in row])
    return out, scores, segs


def render(doc: Doc, n: int, dpi=150) -> Image.Image:
    if doc.kind == "image":
        return Image.open(doc.path).convert("RGB")
    if doc.kind != "pdf":
        raise ValueError("only PDFs and images have a visual page")
    pix = fitz.open(doc.path)[n - 1].get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.w, pix.h), pix.samples)


def ocr_image(pil, reread=True):
    """The Latin OCR path for one page image -> (rows [(text, box)], per-row confidence, per-row boxes, bengali_like)."""
    up = float(os.environ.get("OMNI_OCR_UPSCALE", "1") or 1)
    if up != 1:
        pil = pil.resize((int(pil.width * up), int(pil.height * up)), Image.LANCZOS)
    img = np.array(pil)
    H, W = img.shape[:2]
    r = ocr_engine()(img)
    boxes = []
    for box, txt, sc in zip(r.boxes if r.boxes is not None else [], r.txts or [], r.scores or []):
        if txt.strip() and sc > 0.3:
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            boxes.append((txt.strip(), tuple(float(v) for v in (min(xs) / W, min(ys) / H, max(xs) / W, max(ys) / H)), float(sc)))
    # The Latin OCR has no Bengali recogniser: on a Bengali page it is unsure of most lines (measured median confidence 0.67, against 0.99 on English pages).
    raw = [float(x) for x in (r.scores or [])]
    bengali_like = len(raw) >= 6 and float(np.median(raw)) < 0.85 and sum(x < 0.8 for x in raw) / len(raw) >= 0.4
    if reread and not bengali_like:
        boxes = _reread(pil, boxes)
    rows, scores, segs = _rows(boxes)
    return rows, scores, segs, bengali_like


def run_ocr(doc: Doc, n: int) -> Page:
    """OCR one page (the caller has already paid the ocr_pages budget) and cache the result on disk."""
    page = doc.pages[n - 1]
    if page.source == "ocr":
        return page
    pil = render(doc, n)
    img = np.array(pil)
    H, W = img.shape[:2]
    lang = (doc.lang or os.environ.get("OMNI_OCR_LANG", "auto")).lower()
    lines = scores = segs = None
    if lang == "bn" and ocr_bn.available():  # told it is Bengali: go straight to the Bengali reader
        rows_bn = ocr_bn.repair_item_numbers(ocr_bn.read(pil)[0])
        lines, scores = [(t, b) for t, b, _ in rows_bn], [c for _, _, c in rows_bn]
        segs = [None] * len(lines)
    if lines is None:
        rows, scores, segs, bengali_like = ocr_image(pil, reread=lang != "bn")
        lines = [(_fix_dates(t), b) for t, b in rows]
        if lang == "auto" and bengali_like and ocr_bn.available():
            rows_bn = ocr_bn.repair_item_numbers(ocr_bn.read(pil)[0])
            if rows_bn and ocr_bn.bengali_share(" ".join(t for t, _, _ in rows_bn)) >= 0.3:
                lines, scores = [(t, b) for t, b, _ in rows_bn], [c for _, _, c in rows_bn]
                segs = [None] * len(lines)
    page.lines, page.source, page.scores, page.segs = lines, "ocr", scores, segs
    cf = os.path.join(CACHE, f"{doc.sha}.json")
    cur = json.load(open(cf)) if os.path.exists(cf) else {}
    cur[str(n)] = [[t, list(b), round(sc, 3), sg] for (t, b), sc, sg in zip(lines, scores, segs)]
    with open(cf + ".tmp", "w") as fh:  # write-then-rename so a crash can't leave a corrupt cache
        json.dump(cur, fh)
    os.replace(cf + ".tmp", cf)
    return page
