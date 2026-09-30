"""Specialist sub-agents the orchestrator calls on its own. Each one is small, spends from the shared budget, and returns
a reading plus how much to trust it. The vision model is used as a *focused* specialist (crops, rows, photos), never as the
general reasoner: on whole pages it is much weaker than the text model.
"""
from __future__ import annotations
import difflib, re
from . import ingest
from .budget import Budget
from .llm import LLM

DATE_Q = re.compile(r"\b(date|dob|birth|born|expiry|expire|issued|issue)\b", re.I)
MONEY_Q = re.compile(r"\b(amount|salary|income|fee|price|total|balance|limit|loan amount|tk|bdt|taka)\b", re.I)
NUM_Q = re.compile(r"\b(how (many|much|long|old)|number|age|years|months|term|phone|mobile|id|tin|code)\b", re.I)
CHECK_Q = re.compile(r"\b(gender|sex|marital|married|single|ticked|checked|selected|which option|residence status|education level|loan type|type of loan|yes or no|customer\?)\b", re.I)
DESCRIBE_Q = re.compile(r"\b(describe|what is in|what does .* (show|look)|what color|how many (people|persons|men|women|objects)|is there (a|an)|photo|picture|image)\b", re.I)


GENERIC_WORDS = {"what", "which", "applicant", "customer", "client", "form", "document", "this", "that", "the", "loan", "name"}


def kind_of(question: str) -> str:
    """What sort of value does the question ask for? Decides which format check applies."""
    if DATE_Q.search(question):
        return "date"
    if MONEY_Q.search(question):
        return "amount"
    if NUM_Q.search(question):
        return "number"
    return "word"


# ---------------------------------------------------------------- verifier: format checks
_DATE = re.compile(r"(?<!\d)(\d{1,2})\s*[/.\-]\s*(\d{1,2})\s*[/.\-]\s*(\d{4}|\d{2})(?!\d)")


def find_date(text: str):
    for m in _DATE.finditer(text):
        d, mo = int(m.group(1)), int(m.group(2))
        if 1 <= d <= 31 and 1 <= mo <= 12:
            return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return None


def find_number(text: str, min_digits=3):
    best = None
    for m in re.finditer(r"\d[\d,\.]*", text):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) >= min_digits and (best is None or len(digits) > len(re.sub(r"\D", "", best))):
            best = m.group(0).rstrip(",.")
    return best


def extract_value(kind: str, text: str):
    """The value of the asked kind found inside `text`, or None if the text does not contain a valid one."""
    if kind == "date":
        return find_date(text)
    if kind == "amount":
        return find_number(text, 3)
    if kind == "number":
        return find_number(text, 1)
    return None


def same_value(kind: str, a: str, b: str) -> bool:
    n = lambda s: re.sub(r"[\s,\.]", "", s.lower())
    return bool(a and b) and n(a) == n(b)


# ---------------------------------------------------------------- locating the row/crop to look at
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_row_idx(page, answer: str, question: str):
    """The OCR row holding the answer (or, failing that, the row sharing most distinctive words with the question): (index, text, box) or None."""
    if not page.lines:
        return None
    a = _norm(answer)
    if a and len(a) >= 3:
        for i, (t, box) in enumerate(page.lines):
            if a in _norm(t):
                return i, t, box
    qw = {w for w in re.findall(r"[a-z]{4,}", question.lower())} - GENERIC_WORDS
    best, score = None, 0
    for i, (t, box) in enumerate(page.lines):
        sc = sum(1 for w in re.findall(r"[a-z]{4,}", t.lower()) if w in qw or any(difflib.SequenceMatcher(None, w, q).ratio() > 0.85 for q in qw))
        if sc > score:
            best, score = (i, t, box), sc
    return best


def find_row(page, answer: str, question: str):
    r = find_row_idx(page, answer, question)
    return (r[1], r[2]) if r else None


def row_confidence(page, idx: int) -> float:
    """OCR confidence of a row (1.0 when unknown). Low means handwriting or a poor scan."""
    return page.scores[idx] if 0 <= idx < len(getattr(page, "scores", [])) else 1.0


def crop(doc, page_no, box, pad=0.012, dpi=200):
    img = ingest.render(doc, page_no, dpi=dpi)
    W, H = img.size
    x0, y0, x1, y1 = box
    return img.crop((int(max(x0 - pad, 0) * W), int(max(y0 - pad, 0) * H), int(min(x1 + pad, 1) * W), int(min(y1 + pad, 1) * H)))


VALUE_MARGIN = 0.12  # measured on real handwriting: a wider margin stops clipping the first letters of the value (0.02 -> 0.68, 0.12 -> 0.91 mean similarity)


def value_crop(img, row_text: str, margin=VALUE_MARGIN):
    """Cut the printed label off a 'Label: value' row so the vision model reads only the handwriting. On real forms the whole-row crop made it
    describe the image or repeat the wrong label (mean similarity 0.48); value-only reads score 0.91. The colon position is estimated from
    its character offset in the OCR text."""
    if ":" not in row_text:
        return img
    frac = (row_text.index(":") + 1) / max(len(row_text), 1)
    W, H = img.size
    return img.crop((max(int(W * (frac - margin)), 0), 0, W, H))


def value_of(row_text: str) -> str:
    return row_text.split(":", 1)[1].strip() if ":" in row_text else row_text


def _upscale(img, side=900):
    w, h = img.size
    return img.resize((side, max(int(h * side / w), 24))) if w < side else img


# ---------------------------------------------------------------- vision sub-agents
def read_crop(llm: LLM, b: Budget, img, why="vision: read crop") -> str:
    """Transcribe a small focused image. The vision model is at its best here."""
    b.spend("vlm_looks", 1, why)
    c = [llm.image_part(_upscale(img), max_side=1400), {"type": "text", "text": "Transcribe the text in this image exactly. Reply with only the text."}]
    return llm.chat([{"role": "user", "content": c}], b, max_tokens=40, why=why, vision=True).replace("\n", " ").strip()


def judge_checkbox(llm: LLM, b: Budget, row_img, row_text: str):
    """Which option in a form row is ticked? Asked twice with different wording; accepted only if both answers agree.
    -> (option or None, consistent: bool, alternatives)"""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-/]{2,}", row_text)]
    asks = ["In this form row, which option has a tick or check mark inside its box? Answer with only the option's text as printed.",
            "Look at the small boxes in this form row. Which one is filled in or ticked? Reply with only the word printed next to that box."]
    answers = []
    for q in asks:
        b.spend("vlm_looks", 1, "vision: checkbox")
        c = [llm.image_part(_upscale(row_img), max_side=1400), {"type": "text", "text": q}]
        a = llm.chat([{"role": "user", "content": c}], b, max_tokens=16, why="vision: checkbox", vision=True).replace("\n", " ").strip(" .")
        snap = max(words, key=lambda w: difflib.SequenceMatcher(None, w.lower(), a.lower()).ratio(), default=a)
        ok = difflib.SequenceMatcher(None, snap.lower(), a.lower()).ratio() >= 0.7
        answers.append(snap if ok else a)
    consistent = _norm(answers[0]) == _norm(answers[1]) and bool(answers[0])
    return (answers[0] if consistent else None), consistent, answers


def describe_image(llm: LLM, b: Budget, img, question: str) -> str:
    """A photo or picture with little or no text: answer straight from the pixels."""
    b.spend("vlm_looks", 1, "vision: photo")
    c = [llm.image_part(img, max_side=1024), {"type": "text", "text": question + "\nAnswer briefly using only what is visible."}]
    return llm.chat([{"role": "user", "content": c}], b, max_tokens=80, why="vision: photo", vision=True).replace("\n", " ").strip()


# ---------------------------------------------------------------- checkbox reader: image processing, no model
def read_checkbox_row(row_img, question: str = "", fill_margin=0.06, area_ratio=1.22):
    """Which option in a form row is ticked? OCR finds each option's label; the square next to it is an ink component; the ticked one has
    clearly more ink than its siblings, by fill ratio or by raw ink area (a tick poking out of the box hides in the fill ratio).
    Measured on 60 synthetic rows: ~83% right, 95-97% right when it commits, abstains when the margins are not decisive.
    -> dict(option=str|None, confident=bool, options=[label...], detail=str)"""
    import cv2
    import numpy as np
    rgb = np.array(row_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    r = ingest.ocr_engine()(rgb)
    boxes = []
    for box, txt in zip(r.boxes if r.boxes is not None else [], r.txts or []):
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        boxes.append((txt.strip(), (min(xs), min(ys), max(xs), max(ys))))
    boxes.sort(key=lambda t: t[1][0])
    if len(boxes) < 2:
        return {"option": None, "confident": False, "options": [], "detail": "no option labels found in this row"}
    kws = [w for w in re.findall(r"[a-z]{4,}", question.lower()) if w not in GENERIC_WORDS]
    start = None
    for i, (t, _) in enumerate(boxes):
        if any(w in t.lower() for w in kws):
            start = i
            break
    opts = []
    if start is not None:
        t, bx = boxes[start]
        tail = t.split(":", 1)[1].strip() if ":" in t else ""
        if len(tail) >= 2:  # the row label and its first option were read as one box ('Gender:Male')
            opts.append((re.sub(r"^[^A-Za-z]+", "", tail), bx))
        rest = boxes[start + 1:]
    else:
        rest = boxes[1:]
    for t, bx in rest:
        if t.endswith(":") and opts:  # the next field's label: stop
            break
        opts.append((re.sub(r"^[^A-Za-z]+", "", t), bx))
    if len(opts) < 2:
        return {"option": None, "confident": False, "options": [o[0] for o in opts], "detail": "fewer than two options"}
    n, lab, st, cen = cv2.connectedComponentsWithStats(binary, connectivity=8)
    found = []
    for text, (x0, y0, x1, y1) in opts:
        h = max(y1 - y0, 12)
        rx0, rx1, ry0, ry1 = x0 - 0.7 * h, x0 + 1.7 * h, y0 - 0.35 * h, y1 + 0.35 * h
        best = None
        for k in range(1, n):
            x, y, w, hh, area = st[k]
            cx, cy = cen[k]
            if rx0 <= cx <= rx1 and ry0 <= cy <= ry1 and 0.45 * h <= hh <= 1.9 * h and 0.45 * h <= w <= 2.0 * h and 0.55 <= w / hh <= 1.8:
                if best is None or w * hh > best[0]:
                    best = (w * hh, area / float(w * hh), float(area))
        found.append((text, best))
    have = [(t, b) for t, b in found if b]
    labels = [t for t, _ in found]
    if len(have) < 2 or len(have) < len(found) - 1:
        return {"option": None, "confident": False, "options": labels, "detail": "could not find the boxes next to the options"}
    fills = np.array([b[1] for _, b in have])
    areas = np.array([b[2] for _, b in have])
    fo, ao = np.argsort(fills)[::-1], np.argsort(areas)[::-1]
    by_fill = fills[fo[0]] - fills[fo[1]] >= fill_margin
    by_area = areas[ao[0]] >= area_ratio * areas[ao[1]]
    cand = None
    if by_area and by_fill and fo[0] == ao[0]:
        cand = fo[0]
    elif by_area and not by_fill:
        cand = ao[0]
    elif by_fill and not by_area:
        cand = fo[0]
    if cand is None:
        return {"option": None, "confident": False, "options": labels, "detail": "no box has clearly more ink than the others"}
    return {"option": have[cand][0], "confident": True, "options": labels, "detail": "the box with clearly the most ink"}
