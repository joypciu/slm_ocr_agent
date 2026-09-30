"""Bengali (+ English) OCR through Tesseract, used for scanned pages the Latin OCR cannot read.

RapidOCR has no Bengali recogniser, so a scanned Bangla page comes back as junk with low confidence. Tesseract's `ben` model reads printed Bengali
(measured on a degraded scan of a real form: mean line similarity 0.86 with the accurate model, ~6 s a page on this CPU). It is optional: everything falls
back to the Latin OCR when Tesseract or the Bengali data are not installed.

Layout expected (see download_models.py):  runtime/tesseract/tesseract.exe  and  runtime/tesseract/tessdata_best/{ben,eng}.traineddata (+ configs/)
Override with env TESSERACT_CMD and TESSDATA_DIR.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "runtime", "tesseract")


def _cmd():
    for c in (os.environ.get("TESSERACT_CMD"), os.path.join(ROOT, "tesseract.exe"), os.path.join(ROOT, "tesseract"), shutil.which("tesseract")):
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None


def _tessdata():
    for d in (os.environ.get("TESSDATA_DIR"), os.path.join(ROOT, "tessdata_best"), os.path.join(ROOT, "tessdata")):
        if d and os.path.exists(os.path.join(d, "ben.traineddata")):
            return os.path.abspath(d)
    return None


def available() -> bool:
    return _cmd() is not None and _tessdata() is not None


def bengali_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if "ঀ" <= c <= "৿") / max(len(letters), 1)


def read(img, lang="ben+eng", psm=4, timeout=180):
    """Recognise a PIL image. -> (rows, mean_conf) with rows = [(text, (x0, y0, x1, y1) normalised, confidence 0-1)] top to bottom.
    Page-segmentation mode 4 (a single column of variable-size text) measured far better than mode 6 on forms."""
    W, H = img.size
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "page.png")
        img.convert("RGB").save(path)
        env = dict(os.environ, TESSDATA_PREFIX=_tessdata())
        r = subprocess.run([_cmd(), path, "stdout", "-l", lang, "--psm", str(psm), "tsv"], capture_output=True, env=env, timeout=timeout)
    lines = {}
    for ln in r.stdout.decode("utf8", "ignore").splitlines()[1:]:
        c = ln.split("\t")
        if len(c) < 12 or not c[11].strip():
            continue
        try:
            conf = float(c[10])
        except ValueError:
            continue
        if conf < 0:
            continue
        x, y, w, h = int(c[6]), int(c[7]), int(c[8]), int(c[9])
        lines.setdefault((int(c[2]), int(c[3]), int(c[4])), []).append((c[11], x, y, x + w, y + h, conf))
    rows = []
    for words in lines.values():
        words.sort(key=lambda t: t[1])
        text = unicodedata.normalize("NFC", " ".join(t[0] for t in words))
        box = (min(t[1] for t in words) / W, min(t[2] for t in words) / H, max(t[3] for t in words) / W, max(t[4] for t in words) / H)
        rows.append((text, box, sum(t[5] for t in words) / len(words) / 100.0))
    rows.sort(key=lambda r: (round(r[1][1] * 60), r[1][0]))
    conf = sum(r[2] for r in rows) / len(rows) if rows else 0.0
    return rows, conf


# ---------------------------------------------------------------- item-number repair
_TO_ASCII = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_TO_BN = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
# digits Tesseract confuses in Bengali print (measured on a scan of a real form: ১->৯ was the common one, ৫->0/6, ৪->8)
_CONFUSED = {"9": "1", "1": "9", "6": "5", "5": "60", "0": "5", "8": "4", "4": "8", "3": "1"}


def _fixable(observed: str, expected: str) -> bool:
    """True when `observed` differs from `expected` only in digits the OCR is known to confuse."""
    if len(observed) != len(expected):
        return False
    return all(o == e or e in _CONFUSED.get(o, "") for o, e in zip(observed, expected))


def repair_item_numbers(rows):
    """Numbered items on a form run in sequence: 1.1, 1.2, 1.3 ... then 2.1 (numbering starts at 1 and steps to the next item or the next section).
    If a line's number is not one of those two expected numbers but differs from one only in digits the OCR confuses, restore it.
    Only lines that already look like 'N.M text' are touched, and only towards the sequence."""
    import re
    pat = re.compile(r"^\s*([০-৯0-9]{1,2})\s*[.।]\s*([০-৯0-9]{1,2})(?!\d)")
    out = list(rows)
    cur_major, prev_minor, changed, seen = None, 0, 0, 0
    for i, (text, box, conf) in enumerate(rows):
        m = pat.match(text)
        if not m:
            continue
        seen += 1
        major, minor = m.group(1).translate(_TO_ASCII), m.group(2).translate(_TO_ASCII)
        if cur_major is None:
            candidates = [("1", "1")]
        else:
            candidates = [(str(cur_major), str(prev_minor + 1)), (str(cur_major + 1), "1")]
        if (major, minor) in candidates:
            chosen = (major, minor)
        else:
            chosen = next(((cm, cn) for cm, cn in candidates if _fixable(major, cm) and _fixable(minor, cn)), (major, minor))
        if chosen != (major, minor):
            digits_bn = any("০" <= c <= "৯" for c in m.group(0))
            fixed = f"{chosen[0]}.{chosen[1]}"
            out[i] = ((fixed.translate(_TO_BN) if digits_bn else fixed) + text[m.end():], box, conf)
            changed += 1
        if chosen[0].isdigit() and chosen[1].isdigit():
            cur_major, prev_minor = int(chosen[0]), int(chosen[1])
    return out
