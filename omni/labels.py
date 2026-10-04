"""Reads the value of a label the question names ("What is the value of 'FAX'?", "What is the booking branch?") straight from the form, 0 tokens.

A form field is 'Label: value'. The label is found by its colon (with a tolerance for OCR typos: 'SSf:' for 'SS#:'); the value is what follows the
colon in the same OCR box, else the next box on the row, else the box right under the label. One OCR box is roughly one field, so box boundaries
tell where a value ends ('cc: S. Willinger (3)' | 'K. A. Hutchison'). Anything uncertain (no colon, several different values, tick boxes) returns None
and the question goes to the model as before."""
from __future__ import annotations
import difflib, re

QUOTED = re.compile(r"['\"‘“](.{1,60})['\"’”]")  # outermost quotes: the label may hold an apostrophe (SUBMITTER'S)
VALUE_OF = re.compile(r"\bvalue (?:of|for)\b", re.I)
PLAIN = re.compile(r"^\s*what(?:'s| is| was)\s+(?:the\s+)?([a-z][a-z0-9 '#&/.()-]{1,48}?)\s*\??\s*$", re.I)
WHO = re.compile(r"^\s*who(?:'s| is| was)\s+the\s+([a-z][a-z0-9 '#&/.()-]{1,48}?)\s*\??\s*$", re.I)
BOXES = re.compile("[■-◿☐-☒✓✔✗✘]")
NEXT_LABEL = re.compile(r"(?<=[\s0-9a-z])[A-Z][A-Za-z'#&/.]*(?: [A-Za-z'#&/.]+){0,3} ?:(?!\d)")  # a following 'Label:' ends the value
GENERIC = {"name", "date", "value", "number", "type", "amount", "total", "address", "title", "status"}  # too vague to name a field without quotes


def _option_list(v):
    """Several options sharing a word ('Personal Loan Doctors Loan Auto loan', 'Male Female Other' is not caught): a list to choose from."""
    words = [w.lower() for w in re.findall(r"[A-Za-z]{3,}", v)]
    return len(words) >= 4 and any(words.count(w) >= 2 for w in set(words))


def _key(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _same(label, cand, strict):
    a, b = _key(label), _key(cand)
    if not a or not b:
        return False
    if strict or len(a) <= 3:
        return a == b
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.84


def _cut(v):
    m = NEXT_LABEL.search(v)
    if m and m.start() > 0:
        v = v[:m.start()]
    return re.sub(r"\s+", " ", v).strip(" :;-|,_")


def _label_of(question):
    """-> (label, strict) or None. A quoted label may be matched loosely (OCR typos); an unquoted one must match a page label exactly."""
    m = QUOTED.search(question)
    if m and (VALUE_OF.search(question) or question.strip().lower().startswith(("what is", "what's"))):
        return m.group(1).strip().rstrip(":").strip(), False
    m = PLAIN.match(question) or WHO.match(question)
    if m:
        lab = m.group(1).strip()
        lab = re.sub(r"^name of (?:the )?", "", lab, flags=re.I)  # 'the name of the client' is what the 'Client:' field holds
        if lab.lower() in GENERIC or len(_key(lab)) < 4:
            return None
        return lab, True
    return None


def read(question, lines, segs=None):
    """lines: [(text, box)] of one page; segs: per line [[text, x0, x1], ...] or None. -> (value, row_text) or None."""
    got = _label_of(question)
    if not got:
        return None
    label, strict = got
    n_lab = len(label.split())
    found = {}
    for i, (row, box) in enumerate(lines):
        parts = (segs[i] if segs and i < len(segs) and segs[i] else None) or [[row, box[0], box[2]]]
        for j, (seg, x0, x1, *_) in enumerate(parts):
            for cm in re.finditer(":", seg):
                before = seg[seg.rfind(":", 0, cm.start()) + 1:cm.start()]
                words = before.split()
                if words:
                    words[-1] = re.sub(r"^[\d.,/-]+(?=[A-Za-z])", "", words[-1])  # '1993DEPARTMENT': a value glued to the next label
                if strict:
                    if not _same(label, " ".join(words), True):
                        continue  # unquoted: the whole label must be the question's words ('Booking Branch' is not 'branch')
                elif not any(1 <= n <= len(words) and _same(label, " ".join(words[-n:]), False) for n in (n_lab, n_lab - 1, n_lab + 1)):
                    continue
                src = seg[cm.end():]
                val = _cut(src)
                if not val and j + 1 < len(parts) and not parts[j + 1][0].strip().endswith(":"):  # the value is its own box on the row (not the next label)
                    src = parts[j + 1][0]
                    val = _cut(src)
                if not val and i + 1 < len(lines):        # ...or written under the label
                    nparts = (segs[i + 1] if segs and i + 1 < len(segs) and segs[i + 1] else None) or [[lines[i + 1][0], lines[i + 1][1][0], lines[i + 1][1][2]]]
                    under = [p for p in nparts if x0 - 0.05 <= p[1] <= x1 + 0.05 and ":" not in p[0]]
                    if under:
                        src = under[0][0]
                        val = _cut(src)
                if val and re.match(re.escape(val) + r"\s*:", src.strip()):
                    val = ""  # 'Subtotal: Total: 20,000': what follows is the next label, not a value
                if val and _option_list(val):
                    val = ""  # 'Loan Type: Personal Loan  Doctors Loan  Auto loan': the printed options, not the chosen one (tick marks are not text)
                if val and len(_key(val)) >= 2 and not BOXES.search(val):
                    found.setdefault(_key(val), (val, row))
                break
    return next(iter(found.values())) if len(found) == 1 else None
