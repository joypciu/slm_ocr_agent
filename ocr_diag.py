"""What does the OCR miss or misread? Samples and counts on a benchmark set (data/ocr_truth.json).  python ocr_diag.py dev [n_images]"""
import re, sys
from collections import Counter
import ocr_eval as E

name = sys.argv[1] if len(sys.argv) > 1 else "dev"
files = sorted(f for f in E.truth if f.startswith(E.SETS[name]))[: int(sys.argv[2]) if len(sys.argv) > 2 else None]
missed, misread, why = [], [], Counter()
for f in files:
    boxes = E.read(f)
    for w in E.truth[f]:
        g = E.norm(w["text"])
        if not re.search(r"[a-z0-9]", g):
            continue
        cx, cy = (w["box"][0] + w["box"][2]) / 2, (w["box"][1] + w["box"][3]) / 2
        cover = [b for b in boxes if b[1] - 0.005 <= cx <= b[3] + 0.005 and b[2] - 0.004 <= cy <= b[4] + 0.004]
        h = w["box"][3] - w["box"][1]
        if not cover:
            near = [b for b in boxes if abs((b[2] + b[4]) / 2 - cy) < 0.02 and abs((b[1] + b[3]) / 2 - cx) < 0.1]
            why["missed: " + ("tiny (<1.2% high)" if h < 0.012 else "single char" if len(g) == 1 else "beside other text" if near else "isolated")] += 1
            missed.append((f[-14:], w["text"], round(h, 4), [b[0][:25] for b in near][:2]))
            continue
        text = " ".join(b[0].lower() for b in cover)
        core = re.sub(r"[^a-z0-9]", "", g)
        if re.search(r"\d", g) and not (core in re.split(r"[^a-z0-9]+", text) or (len(core) >= 3 and core in re.sub(r"[^a-z0-9]", "", text))):
            misread.append((f[-14:], w["text"], [b[0][:30] for b in cover], round(min(b[5] for b in cover), 2)))
print(why.most_common())
print(f"\nmissed ({len(missed)}), sample:")
for m in missed[:: max(1, len(missed) // 25)][:25]:
    print("  ", m)
print(f"\nnumbers misread ({len(misread)}), sample:")
for m in misread[:: max(1, len(misread) // 30)][:30]:
    print("  ", m)
