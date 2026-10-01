"""Reads receipt / invoice style rows ('SUBTTL 28.182', 'PB-1 10% 2.818', 'J.STB PROMO 17500') without a model.
A row is a label followed by its amount; labels are matched with a tolerance for OCR typos ('SUB TOTAI', 'SUS_TOTAL').
Only simple, canonical questions are handled (total / subtotal / tax / price of ITEM); anything ambiguous returns None so the model path decides."""
from __future__ import annotations
import difflib, re

MONEY = re.compile(r"(?<![\w.,-])(?:[$€£]|rp\.?|tk\.?|bdt|usd)?\s?\d{1,3}(?:[.,] ?\d{3})+(?:[.,]\d{1,2})?(?![\w%]|[.,]\d)|(?<![\w.,-])\d+(?:[.,]\d{1,2})?(?![\w%]|[.,]\d)", re.I)
SIMPLE_Q = re.compile(r"^\s*(?:what|how much)\s+(?:is|was|are)\s+the\s+((?:grand |net |final |sub ?)?total|subtotal|sub-total|tax|vat|gst|service charge|amount due|balance due)(?:\s+(?:amount|due|payable|price|value))?\s*\??\s*$", re.I)
PRICE_Q = re.compile(r"^\s*(?:what|how much)\s+(?:is|was)\s+the\s+(?:price|cost|amount)\s+(?:of|for)\s+(.+?)\s*\??\s*$", re.I)

TOTAL_LABELS = ("grand total", "total", "total sales", "net total", "total payable", "amount due", "balance due", "total due", "jumlah", "total amount")
SUB_LABELS = ("sub total", "subtotal", "subttl", "sub ttl", "sub-total", "total before tax", "net amount")
TAX_LABELS = ("tax", "vat", "gst", "ppn", "pb1", "pb-1", "pajak", "sales tax", "service charge")


def _norm(s):
    return re.sub(r"[^a-z0-9% ]", " ", s.lower().replace("_", " ").replace("-", "")).strip()


def _words(s):
    return _norm(s).split()


def number_in(text):
    """All money-looking amounts in a row, as written, left to right (percentages and bare year-like/short counters left out)."""
    return [m.group(0).strip() for m in MONEY.finditer(text)]


def digits(n):
    return re.sub(r"\D", "", n)


def _label_part(row):
    """The text before the first amount."""
    m = MONEY.search(row)
    label = row[:m.start()] if m else row
    return re.sub(r"[\d.,]+\s*%", " ", label)  # 'PB-1 10%' is the label 'PB-1' with its rate


def _matches(label_text, labels):
    lt = _norm(label_text)
    lw = lt.split()
    if not lw:
        return False
    for lab in labels:
        ln = _norm(lab)
        if lt == ln or lt.replace(" ", "") == ln.replace(" ", ""):
            return True
        # typo-tolerant, but only for labels long enough to be distinctive
        if len(ln.replace(" ", "")) >= 5 and difflib.SequenceMatcher(None, lt.replace(" ", ""), ln.replace(" ", "")).ratio() >= 0.82:
            return True
        if len(lw) >= 1 and ln in lt and len(lw) <= len(ln.split()) + 1:
            return True
    return False


def _clean(n):
    """An amount that can really be one: no zero-padded junk ('000800'), no lone digit counters."""
    d = digits(n)
    return bool(d) and not (len(d) > 1 and d.startswith("0") and not re.match(r"^0[.,]\d", n.strip()))


def _value(rows, i, first=False):
    """The amount that belongs to row i. Total-like rows may carry what was paid after the amount ('TOTAL 80,500  100,000'), so those take the first;
    item rows with two amounts (quantity x price) are ambiguous. With no amount on the row, the lone amount alone on the next row."""
    nums = [n for n in number_in(rows[i]) if _clean(n)]
    if nums:
        if len(nums) == 1 or first:
            return nums[0]
        return None
    if i + 1 < len(rows):
        nxt = [n for n in number_in(rows[i + 1]) if _clean(n)]
        if len(nxt) == 1 and not _label_part(rows[i + 1]).strip(" :"):
            return nxt[0]
    return None


def _as_int(v):
    return int(digits(v) or 0)


SUBWORD = re.compile(r"\bsub|sub ?t|discount|before|items?|qty")


def answer(question, text):
    """-> (value, row) or None.  `text` is the page text, one row per line."""
    rows = [r.strip() for r in text.split("\n") if r.strip()]
    m = SIMPLE_Q.match(question)
    if m:
        kind = re.sub(r"[\s-]", "", m.group(1).lower())
        if kind == "subtotal":
            labels, mode = SUB_LABELS, "sub"
        elif kind in ("tax", "vat", "gst", "servicecharge"):
            labels, mode = TAX_LABELS, "tax"
        else:
            labels, mode = TOTAL_LABELS, "total"
        hits = []
        for i, r in enumerate(rows):
            lab = _label_part(r)
            if not _matches(lab, labels):
                continue
            n = _norm(lab)
            if mode in ("total", "tax") and SUBWORD.search(n):
                continue  # 'Subtotal: Total:' / 'PAJAK Subtotal': a row that names two things is not this one
            if mode == "sub" and re.search(r"\btotal\b.*\btotal\b|sub ?total.*\btotal\b|\btax\b|\bpajak\b", n):
                continue
            v = _value(rows, i, first=True)
            if v:
                hits.append((v, r))
        if not hits:
            return None
        if mode == "total":
            hit = hits[-1]  # the grand total is the last total-like row
        else:
            hit = hits[0]
        if mode == "tax":
            base = [_as_int(x[0]) for x in (answer("What is the subtotal?", text), answer("What is the total amount?", text)) if x]
            if base and _as_int(hit[0]) >= min(base):
                return None  # a tax larger than the amount it is levied on: a misread row
        return hit
    m = PRICE_Q.match(question)
    if m:
        item = [w for w in _words(m.group(1)) if len(w) >= 2]
        if not item:
            return None
        best, best_sc = None, 0.0
        for i, r in enumerate(rows):
            lab = _words(_label_part(r))
            if not lab:
                continue
            sc = sum(max((difflib.SequenceMatcher(None, w, x).ratio() for x in lab), default=0) >= 0.8 for w in item) / len(item)
            if sc > best_sc:
                best, best_sc = i, sc
        if best is None or best_sc < 0.6:
            return None
        v = _value(rows, best)
        return (v, rows[best]) if v else None
    return None
