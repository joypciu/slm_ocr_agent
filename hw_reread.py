"""Local experiment: can a second recognition pass on a low-confidence row beat the page OCR on real handwriting? (data/real_hw_truth.json, git-ignored)
Variants re-recognise the row crop only (no detection), at different scales / preprocessing / recognition models; 'best-conf' keeps the most confident."""
import difflib, json, re, time
import numpy as np
from PIL import Image, ImageOps
from rapidocr import RapidOCR
from rapidocr.utils.typings import LangRec

items = json.load(open("data/real_hw_truth.json"))
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
sim = lambda a, b: difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
base = {"Det.model_path": "ocr_models/ch_PP-OCRv5_det_mobile.onnx", "Global.use_cls": False}
engines = {
    "v6small": RapidOCR(params={**base, "Rec.model_path": "ocr_models/PP-OCRv6_rec_small.onnx", "Rec.lang_type": LangRec.LATIN}),
    "v5en": RapidOCR(params={**base, "Rec.model_path": "ocr_models/en_PP-OCRv5_rec_mobile.onnx", "Rec.lang_type": LangRec.EN}),
}


def rec(eng, img):
    r = eng(np.array(img), use_det=False, use_cls=False, use_rec=True)
    t = (r.txts or [""])[0] if r.txts else ""
    s = float((r.scores or [0])[0]) if r.scores else 0.0
    return t, s


def variants(img):
    g = ImageOps.grayscale(img)
    yield "1x", img
    yield "2x", img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    yield "pad", ImageOps.expand(img, border=(12, 8), fill="white")
    yield "bin", g.point(lambda p: 255 if p > 150 else 0).convert("RGB")
    yield "autocontrast", ImageOps.autocontrast(g, cutoff=2).convert("RGB")


res = {}
for it in items:
    img = Image.open(it["file"]).convert("RGB")
    cands = [("page", it["ppocr"], float(it["score"]))]
    for en, eng in engines.items():
        for vn, im in variants(img):
            t, s = rec(eng, im)
            cands.append((f"{en}-{vn}", t, s))
    for name, t, s in cands:
        res.setdefault(name, []).append((t, it["truth"]))
    best = max(cands, key=lambda c: c[2])
    res.setdefault("best-conf", []).append((best[1], it["truth"]))
    # agreement vote: the reading most variants agree on (ties -> page)
    votes = {}
    for _, t, s in cands:
        votes[norm(t)] = votes.get(norm(t), 0) + 1
    top = max(cands, key=lambda c: (votes[norm(c[1])], c[0] == "page"))
    res.setdefault("vote", []).append((top[1], it["truth"]))

for name, pairs in sorted(res.items(), key=lambda kv: -np.mean([sim(a, b) for a, b in kv[1]])):
    ex = sum(norm(a) == norm(b) for a, b in pairs)
    print(f"{name:20s} exact {ex:2d}/20  mean similarity {np.mean([sim(a, b) for a, b in pairs]):.3f}")

# which variants carry the gain: greedy subsets, always starting from the page reading
import itertools
names = [n for n in res if n not in ("best-conf", "vote", "page")]
per = {}  # name -> list of (text, conf) per item, recomputed
