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
TAX_LABELS = ("tax", "vat", "gst", "ppn", "pb1", "pb-1", "pb", "pb 1", "pajak", "pajak resto", "sales tax", "service charge")


def _norm(s):
    return re.sub(r"[^a-z0-9% ]", " ", s.lower().replace("_", " ").replace("-", "")).strip()


def _words(s):
    return _norm(s).split()


SMALL = re.compile(r"\d{1,2}")


def _amounts(text):
    """Money-looking matches in a row, left to right. A bare 1-2 digit number is a quantity or part of a name ('1 FUTAMI 17 GREEN TEA', 'Pb 1'), not an amount."""
    return [m for m in MONEY.finditer(text) if not SMALL.fullmatch(m.group(0).strip())]


def number_in(text):
    """All money-looking amounts in a row, as written, left to right (percentages and quantities left out)."""
    return [m.group(0).strip() for m in _amounts(text)]


def tidy(v):
    """'16, 500' (an OCR space inside the number) -> '16,500'."""
    v = re.sub(r"(?<=[.,]) (?=\d)", "", v.strip())
    v = re.sub(r"(?<=\d) (?=\d{3}(?!\d))", "", v)        # '$ 48 801,10': a space used as the thousands separator
    return re.sub(r"^([$€£]) (?=\d)", r"\1", v)


def digits(n):
    return re.sub(r"\D", "", n)


def _label_part(row):
    """The text before the first amount."""
    ms = _amounts(row)
    label = row[:ms[0].start()] if ms else row
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


def _value(rows, i, mode):
    """-> (amount, on_same_row) for row i, or None.
    Total-like rows may carry what was paid after the amount ('TOTAL 80,500  100,000'): the first amount. Item rows end with the line total
    ('2x @12.000 24.000', '16,363 16363'): the last. With no amount on the row, the next row when it holds only amounts / quantity marks."""
    nums = [n for n in number_in(rows[i]) if _clean(n)]
    if nums:
        return (nums[-1] if mode == "price" else nums[0]), True
    if i + 1 < len(rows):
        nxt = [n for n in number_in(rows[i + 1]) if _clean(n)]
        rest = re.sub(r"[\d\s.,@x:]|rp", "", rows[i + 1].lower())
        if nxt and not rest and (len(nxt) == 1 or mode == "price"):
            return nxt[-1], False
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
            got = _value(rows, i, mode)
            if got:
                hits.append((tidy(got[0]), r, got[1]))
        if not hits:
            return None
        same = [h for h in hits if h[2]] or hits  # a value on the label's own row beats one borrowed from the next row
        hit = same[-1] if mode == "total" else same[0]  # the grand total is the last total-like row
        if mode == "tax":
            base = [_as_int(x[0]) for x in (answer("What is the subtotal?", text), answer("What is the total amount?", text)) if x]
            if base and _as_int(hit[0]) >= min(base):
                return None  # a tax larger than the amount it is levied on: a misread row
        return hit[0], hit[1]
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
        got = _value(rows, best, "price")
        return (tidy(got[0]), rows[best]) if got else None
    return None


COLUMN_Q = re.compile(r"^\s*(?:what|how much)\s+(?:is|was)\s+the\s+total\s+([a-z][a-z %\[\]]{1,30}?)\s*\??\s*$", re.I)


def column_total(question, lines, segs):
    """'What is the total gross worth?' on an invoice summary table: the 'Total' row's box that sits under the 'Gross worth' header.
    Needs the OCR boxes (segs); -> (value, row) or None."""
    m = COLUMN_Q.match(question)
    if not m or not segs:
        return None
    col = _norm(m.group(1)).replace(" ", "")
    if col in ("amount", "price", "value", "due", "sum", ""):
        return None  # 'total amount' is the receipt question, handled by answer()
    for i, parts in enumerate(segs):
        if not parts:
            continue
        heads = [p for p in parts if _norm(p[0]).replace(" ", "") == col]
        if not heads:
            continue
        hx = (heads[0][1] + heads[0][2]) / 2
        for k in range(i + 1, min(i + 4, len(segs))):
            row = segs[k] or []
            if not row or not _matches(row[0][0], ("total",)):
                continue
            cands = [p for p in row[1:] if re.search(r"\d", p[0]) and abs((p[1] + p[2]) / 2 - hx) < 0.06]
            if len(cands) == 1:
                return tidy(cands[0][0]), lines[k][0]
    return None
