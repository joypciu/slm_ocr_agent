"""Score the vision sub-agents on the synthetic benchmark.  python vision_eval.py <task> [variant]   (needs the vision server on :8082)"""
import json, os, re, sys, time
os.environ.setdefault("OMNI_VISION_URL", "http://127.0.0.1:8082")
os.environ["OMNI_VISION"] = "0"
import numpy as np
from PIL import Image
from omni import ingest, subagents as sa
from omni.budget import Budget
from omni.llm import LLM
from vision_bench import OUT, split

llm = LLM()
CASES = json.load(open(f"{OUT}/cases.json"))


def B():
    return Budget.make("thorough", llm_tokens=10 ** 6, vlm_looks=10 ** 4, seconds=10 ** 4, tool_calls=10 ** 4, ocr_pages=10 ** 4)


def norm(s):
    return re.sub(r"[\s,\.]", "", str(s).lower())


def report(name, rows):
    dev, test = split(rows)
    for lab, part in (("dev", dev), ("test", test)):
        n = len(part)
        ok = sum(r["ok"] for r in part)
        extra = ""
        if "abstain" in part[0]:
            ab = sum(r["abstain"] for r in part)
            wrong = sum((not r["ok"]) and not r["abstain"] for r in part)
            extra = f" | right {ok}/{n}  abstained {ab}  confidently-wrong {wrong}"
        print(f"{name:34} {lab:4} n={n:3} acc {ok / n:.3f}{extra}", flush=True)


# ---------------------------------------------------------------- checkbox variants
def cb_baseline(c):
    img = Image.open(c["file"]).convert("RGB")
    opt, consistent, answers = sa.judge_checkbox(llm, B(), img, c["row_text"])
    return {"ok": bool(consistent and norm(opt) == norm(c["truth"])), "abstain": not consistent, "answers": answers}


# ---------------------------------------------------------------- handwriting variants
def hw_ocr(c):
    r = ingest.ocr_engine()(np.array(Image.open(c["file"]).convert("RGB")))
    return " ".join(r.txts or [])


def hw_vlm(c):
    return sa.read_crop(llm, B(), Image.open(c["file"]).convert("RGB"))


# ---------------------------------------------------------------- photo variants
def photo_baseline(c):
    a = sa.describe_image(llm, B(), Image.open(c["file"]).convert("RGB"), c["q"])
    return {"ok": re.search(r"(?<![a-z0-9])" + re.escape(c["truth"]) + r"(?![a-z])", a.lower()) is not None, "answer": a}


# ================================================================= new checkbox designs (experiments) =================================================================
import cv2

_OCR_CACHE = {}


def row_boxes(c):
    """PP-OCR text boxes of a row image, left to right: [(text, (x0, y0, x1, y1) in pixels)]. The first box is the row's label."""
    if c["file"] not in _OCR_CACHE:
        img = np.array(Image.open(c["file"]).convert("RGB"))
        r = ingest.ocr_engine()(img)
        out = []
        for box, txt in zip(r.boxes if r.boxes is not None else [], r.txts or []):
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            out.append((txt.strip(), (min(xs), min(ys), max(xs), max(ys))))
        _OCR_CACHE[c["file"]] = sorted(out, key=lambda t: t[1][0])
    return _OCR_CACHE[c["file"]]


def option_regions(c):
    """For each option label: (label text, region holding its checkbox, crop box for a per-option look).
    The OCR box of an option already contains its square at the left edge, so the square is the first ~1.3 text heights of the box."""
    boxes = row_boxes(c)
    res = []
    for txt, (x0, y0, x1, y1) in boxes[1:]:  # boxes[0] is the row label, e.g. 'Gender:'
        h = max(y1 - y0, 10)
        res.append((re.sub(r"^[^A-Za-z]+", "", txt), (x0 - 0.6 * h, y0 - 0.2 * h, x0 + 1.3 * h, y1 + 0.2 * h), (x0 - 0.8 * h, y0 - 0.5 * h, x1 + 0.3 * h, y1 + 0.5 * h)))
    return res


def ink(binary, region):
    x0, y0, x1, y1 = [int(max(v, 0)) for v in region]
    part = binary[y0:y1, x0:x1]
    return float((part > 0).mean()) if part.size else 0.0


def binarize(c):
    g = cv2.cvtColor(np.array(Image.open(c["file"]).convert("RGB")), cv2.COLOR_RGB2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b


def cb_pixel(c, margin=0.05):
    b = binarize(c)
    regs = option_regions(c)
    if len(regs) < 2:
        return {"ok": False, "abstain": True, "answers": []}
    inks = [ink(b, r[1]) for r in regs]
    order = np.argsort(inks)[::-1]
    top, second = inks[order[0]], inks[order[1]]
    if top - second < margin:
        return {"ok": False, "abstain": True, "answers": inks}
    pick = regs[order[0]][0]
    return {"ok": norm(pick) == norm(c["truth"]), "abstain": False, "answers": inks}


def cb_pervlm(c):
    img = Image.open(c["file"]).convert("RGB")
    regs = option_regions(c)
    yes = []
    for label, _, crop_box in regs:
        x0, y0, x1, y1 = [int(max(v, 0)) for v in crop_box]
        piece = img.crop((x0, y0, x1, y1))
        q = "Look at the small square box on the left. Is it ticked, crossed or filled in? Answer yes or no."
        b = B()
        b.spend("vlm_looks", 1)
        ans = llm.chat([{"role": "user", "content": [llm.image_part(sa._upscale(piece, 500), max_side=900), {"type": "text", "text": q}]}], b, max_tokens=4, vision=True).lower()
        yes.append(ans.startswith("yes"))
    if sum(yes) != 1:
        return {"ok": False, "abstain": True, "answers": yes}
    pick = regs[yes.index(True)][0]
    return {"ok": norm(pick) == norm(c["truth"]), "abstain": False, "answers": yes}


# ---- transcription-style checkbox: the model is asked to COPY the row, marking ticked boxes (plays to its OCR strength)
def cb_transcribe(c, prompt_id=0):
    img = Image.open(c["file"]).convert("RGB")
    prompts = ["Copy this form row exactly. Write [x] in front of an option if its box is ticked or crossed out, and [ ] in front of an option if its box is empty.",
               "Transcribe this line. For every small square box write [x] if it has a tick, cross or fill inside, otherwise write [ ]. Then the word next to it."]
    b = B()
    b.spend("vlm_looks", 1)
    out = llm.chat([{"role": "user", "content": [llm.image_part(sa._upscale(img, 1000), max_side=1400), {"type": "text", "text": prompts[prompt_id]}]}], b, max_tokens=80, vision=True)
    ticked = []
    for opt in c["options"]:
        m = re.search(r"\[\s*([^\]\s]{0,2})\s*\]\s*" + re.escape(opt.split()[0]), out, re.I)
        if m and m.group(1).lower() in ("x", "v", "✓", "✔", "√", "t"):
            ticked.append(opt)
    if len(ticked) != 1:
        return {"ok": False, "abstain": True, "answers": [out[:80]]}
    return {"ok": norm(ticked[0]) == norm(c["truth"]), "abstain": False, "answers": [out[:80]]}


def cb_transcribe2(c):
    return cb_transcribe(c, 1)


# ================================================================= picture questions with test-time augmentation
from PIL import ImageOps
from collections import Counter

NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6}


def _num(a):
    for w in re.findall(r"[a-z0-9]+", a.lower()):
        if w in NUMW:
            return str(NUMW[w])
    return None


def _side(a):
    a = a.lower()
    if "left" in a and "right" not in a:
        return "left"
    if "right" in a and "left" not in a:
        return "right"
    return None


def photo_tta(c):
    img = Image.open(c["file"]).convert("RGB")
    q = c["q"]
    if c["kind"] == "count":
        votes = []
        for side in (384, 640, 1024):
            im = img.copy()
            im.thumbnail((side, side))
            votes.append(_num(sa.describe_image(llm, B(), im, q + " Answer with a number.")))
        votes = [v for v in votes if v]
        ans = Counter(votes).most_common(1)[0][0] if votes else ""
    elif c["kind"] == "side":
        a1 = _side(sa.describe_image(llm, B(), img, q + " Answer left or right."))
        a2 = _side(sa.describe_image(llm, B(), ImageOps.mirror(img), q + " Answer left or right."))
        flip = {"left": "right", "right": "left"}.get(a2)
        ans = a1 if (a1 and a1 == flip) else (a1 or flip or "")
        ans = ans or ""
    else:
        ans = sa.describe_image(llm, B(), img, q + " Answer with one word.")
    return {"ok": re.search(r"(?<![a-z0-9])" + re.escape(c["truth"]) + r"(?![a-z])", ans.lower()) is not None, "answer": ans}


# ================================================================= checkbox by connected components (no model)
def cb_cc_detail(c):
    """Per option: the checkbox-like ink component near its label and how filled it is."""
    img = np.array(Image.open(c["file"]).convert("L"))
    _, b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, st, cen = cv2.connectedComponentsWithStats(b, connectivity=8)
    out = []
    for text, (x0, y0, x1, y1) in row_boxes(c)[1:]:
        h = max(y1 - y0, 12)
        rx0, rx1, ry0, ry1 = x0 - 0.7 * h, x0 + 1.7 * h, y0 - 0.35 * h, y1 + 0.35 * h
        best = None
        for k in range(1, n):
            x, y, w, hh, area = st[k]
            cx, cy = cen[k]
            if not (rx0 <= cx <= rx1 and ry0 <= cy <= ry1):
                continue
            if not (0.45 * h <= hh <= 1.9 * h and 0.45 * h <= w <= 2.0 * h and 0.55 <= w / hh <= 1.8):
                continue
            if best is None or w * hh > best[0]:
                best = (w * hh, area / float(w * hh), area)
        out.append((re.sub(r"^[^A-Za-z]+", "", text), best))
    return out


def cb_cc(c, margin=0.06):
    d = cb_cc_detail(c)
    have = [(l, b) for l, b in d if b]
    if len(have) < 2 or len(have) < len(d) - 0:
        return {"ok": False, "abstain": True, "answers": [(l, round(b[1], 2) if b else None) for l, b in d]}
    fills = [b[1] for _, b in have]
    order = np.argsort(fills)[::-1]
    if fills[order[0]] - fills[order[1]] < margin:
        return {"ok": False, "abstain": True, "answers": fills}
    pick = have[order[0]][0]
    return {"ok": norm(pick) == norm(c["truth"]), "abstain": False, "answers": fills}


def cb_cc2(c, fill_margin=0.06, area_ratio=1.22):
    """Ticked box = clearly more ink than its siblings, judged by fill ratio OR raw ink area (a tick poking out of the box hides in the fill ratio)."""
    d = cb_cc_detail(c)
    have = [(l, b) for l, b in d if b]
    if len(have) < 2 or len(have) < len(d) - 1:
        return {"ok": False, "abstain": True, "answers": []}
    fills = np.array([b[1] for _, b in have])
    areas = np.array([b[2] for _, b in have], dtype=float)
    fo, ao = np.argsort(fills)[::-1], np.argsort(areas)[::-1]
    by_fill = fills[fo[0]] - fills[fo[1]] >= fill_margin
    by_area = areas[ao[0]] >= area_ratio * areas[ao[1]]
    cand = None
    if by_area and by_fill and fo[0] == ao[0]:
        cand = fo[0]
    elif by_area and (not by_fill):
        cand = ao[0]
    elif by_fill and (not by_area):
        cand = fo[0]
    if cand is None:
        return {"ok": False, "abstain": True, "answers": [fills.round(2).tolist(), areas.tolist()]}
    return {"ok": norm(have[cand][0]) == norm(c["truth"]), "abstain": False, "answers": [fills.round(2).tolist(), areas.tolist()]}


def cb_reader(c):
    """The production reader from omni/subagents.py, on the benchmark rows."""
    r = sa.read_checkbox_row(Image.open(c["file"]), c["row_text"].split(":")[0])
    return {"ok": bool(r["confident"] and norm(r["option"]) == norm(c["truth"])), "abstain": not r["confident"], "answers": [r["option"], r["detail"]]}


if __name__ == "__main__":
    task = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    t0 = time.time()
    if task == "checkbox":
        fn = globals()["cb_" + variant]
        rows = [fn(c) for c in CASES["checkbox"]]
        report("checkbox/" + variant, rows)
        for c, r in list(zip(CASES["checkbox"], rows))[:0]:
            print(c["truth"], r)
    elif task == "hand":
        rows = []
        for c in CASES["hand"]:
            o, v = hw_ocr(c), hw_vlm(c)
            rows.append({"ok": norm(o) == norm(c["truth"]), "vlm_ok": norm(v) == norm(c["truth"]), "ocr": o, "vlm": v, "kind": c["kind"], "truth": c["truth"]})
        report("hand/ocr exact", rows)
        report("hand/vlm exact", [{"ok": r["vlm_ok"]} for r in rows])
        both = [{"ok": r["ok"] or r["vlm_ok"]} for r in rows]
        report("hand/either (oracle upper bound)", both)
        for kind in ("date", "amount", "name", "place"):
            sub = [r for r in rows if r["kind"] == kind]
            print(f"  {kind:7} ocr {sum(r['ok'] for r in sub)}/{len(sub)}  vlm {sum(r['vlm_ok'] for r in sub)}/{len(sub)}")
        for r in rows[:6]:
            print("   truth", repr(r["truth"]), "| ocr", repr(r["ocr"]), "| vlm", repr(r["vlm"]))
    elif task == "photo":
        fn = globals()["photo_" + variant]
        rows = [fn(c) for c in CASES["photo"]]
        report("photo/" + variant, rows)
        for kind in ("color", "count", "side"):
            sub = [r for r, c in zip(rows, CASES["photo"]) if c["kind"] == kind]
            print(f"  {kind:6} {sum(r['ok'] for r in sub)}/{len(sub)}")
    print(f"({time.time() - t0:.0f}s)")
