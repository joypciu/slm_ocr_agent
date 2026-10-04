"""How often does our OCR misread? Word-level check against FUNSD / CORD ground truth (data/ocr_truth.json, made by build_ocr_truth.py).
For each true word: find the OCR box over its centre; 'exact' if the word is among that box's words (case and edge punctuation ignored),
'missed' if no box covers it, else 'misread'. Numbers are reported separately (amounts, dates, IDs drive most answers).
   python ocr_eval.py [set ...]        sets: dev (data/varied), tune (data/varied_fresh), final (data/varied_final)
   OCR settings come from the environment (e.g. OMNI_REREAD=1, OMNI_OCR_UPSCALE=1.5, OMNI_DET_SIDE=1280)."""
import difflib, json, os, re, sys, time
from collections import defaultdict
import numpy as np
from PIL import Image
from omni import ingest

SETS = {"dev": "data/varied/", "tune": "data/varied_fresh/", "final": "data/varied_final/"}
truth = json.load(open("data/ocr_truth.json", encoding="utf-8"))


def norm(w):
    return re.sub(r"^[^\w]+|[^\w]+$", "", w.lower())


def read(path):
    """Run the production OCR path on one image, without the disk cache. -> list of (text, x0, y0, x1, y1) boxes."""
    pil = Image.open(path).convert("RGB")
    rows, scores, segs, _ = ingest.ocr_image(pil)
    out = []
    for (t, b), sg in zip(rows, segs):
        for s in (sg or [[t, b[0], b[2]]]):
            out.append((s[0], s[1], b[1], s[2], b[3], s[3] if len(s) > 3 else 1.0, s[4] if len(s) > 4 else None))
    return out


FLAGS = defaultdict(lambda: [0, 0])   # signal -> [words flagged, of which misread]
MISREADS = defaultdict(int)            # signal -> misreads it caught (plus 'all')


def score(words, boxes):
    st = defaultdict(int)
    sims = []
    for w in words:
        g = norm(w["text"])
        if not re.search(r"[a-z0-9]", g):
            continue
        cx, cy = (w["box"][0] + w["box"][2]) / 2, (w["box"][1] + w["box"][3]) / 2
        cover = [b for b in boxes if b[1] - 0.005 <= cx <= b[3] + 0.005 and b[2] - 0.004 <= cy <= b[4] + 0.004]
        kind = "num" if re.search(r"\d", g) else "word"
        st[kind + "_n"] += 1
        if not cover:
            st[kind + "_missed"] += 1
            sims.append(0.0)
            continue
        toks = [norm(t) for b in cover for t in b[0].split()]
        text = " ".join(b[0].lower() for b in cover)
        core = re.sub(r"[^a-z0-9]", "", g)
        pieces = re.split(r"[^a-z0-9]+", text)
        # the datasets split '614-466-5087' into '614' '-466' '-5087' and dates into '12' '/10' '/98': compare letters and digits only
        ok = core in pieces or (len(core) >= 3 and core in re.sub(r"[^a-z0-9]", "", text))
        low = min(b[5] for b in cover)
        signals = {"conf<0.80": low < 0.80, "conf<0.90": low < 0.90, "conf<0.95": low < 0.95, "readers disagree": any(b[6] for b in cover)}
        signals["disagree or conf<0.80"] = signals["readers disagree"] or signals["conf<0.80"]
        for name, on in signals.items():
            if on:
                FLAGS[kind + " " + name][0] += 1
                FLAGS[kind + " " + name][1] += not ok
                MISREADS[kind + " " + name] += not ok
        MISREADS[kind + " all"] += not ok
        if ok:
            st[kind + "_exact"] += 1
            sims.append(1.0)
        else:
            st[kind + "_misread"] += 1
            sims.append(max((difflib.SequenceMatcher(None, g, t).ratio() for t in toks), default=0.0))
    return st, sims


if __name__ == "__main__":
    pick = sys.argv[1:] or ["dev", "tune"]
    t0 = time.time()
    for name in pick:
        tot, sims, by = defaultdict(int), [], defaultdict(lambda: defaultdict(int))
        files = sorted(f for f in truth if f.startswith(SETS[name]))
        for f in files:
            st, s = score(truth[f], read(f))
            sims += s
            for k, v in st.items():
                tot[k] += v
                by["funsd" if "/funsd/" in f else "cord"][k] += v
        line = f"{name:5s} {len(files)} images"
        for kind in ("word", "num"):
            n = tot[kind + "_n"] or 1
            line += f" | {kind}s n={tot[kind + '_n']} exact {tot[kind + '_exact'] / n:.3f} misread {tot[kind + '_misread'] / n:.3f} missed {tot[kind + '_missed'] / n:.3f}"
        print(line + f" | char-sim {np.mean(sims):.3f}", flush=True)
        for cat, st in by.items():
            print(f"      {cat:5s} words exact {st['word_exact'] / max(st['word_n'], 1):.3f}  numbers exact {st['num_exact'] / max(st['num_n'], 1):.3f}"
                  f"  (misread {st['num_misread'] / max(st['num_n'], 1):.3f}, missed {st['num_missed'] / max(st['num_n'], 1):.3f})")
    print("flag signal                     flagged  misread-when-flagged  misreads caught")
    for kind in ("num", "word"):
        n_all = sum(1 for k in FLAGS if k.startswith(kind))
        for name in ("conf<0.80", "conf<0.90", "conf<0.95", "readers disagree", "disagree or conf<0.80"):
            f, m = FLAGS[kind + " " + name]
            print(f"  {kind:4s} {name:24s} {f:6d}   {m / max(f, 1):6.2f}               {MISREADS[kind + ' ' + name] / max(MISREADS[kind + ' all'], 1):.2f}")
    print(f"{time.time() - t0:.0f}s")
