"""Word-level OCR ground truth for the benchmark images (run with an environment that has `datasets`).
FUNSD forms and CORD receipts ship the true text and box of every word; our saved images are matched back to their dataset records by pixel hash.
Output: data/ocr_truth.json  {image_path: [{"text": word, "box": [x0, y0, x1, y1] (0-1)}]}"""
import glob, hashlib, json
from datasets import load_dataset
from PIL import Image


def h(img):
    return hashlib.md5(img.convert("L").resize((32, 32)).tobytes()).hexdigest()


want = {}
for sub in ("varied", "varied_fresh", "varied_final"):
    for kind in ("funsd", "cord"):
        for f in glob.glob(f"data/{sub}/{kind}/*.png"):
            want.setdefault(kind, {})[h(Image.open(f))] = f.replace("\\", "/")
truth = {}

for split in ("test", "train"):
    for ex in load_dataset("nielsr/funsd", split=split, streaming=True):
        f = want["funsd"].get(h(ex["image"]))
        if not f:
            continue
        W, H = ex["image"].size
        scale = (1000, 1000)  # this FUNSD copy gives boxes on a 0-1000 scale
        truth[f] = [{"text": w, "box": [b[0] / scale[0], b[1] / scale[1], b[2] / scale[0], b[3] / scale[1]]} for w, b in zip(ex["words"], ex["bboxes"]) if w.strip()]
print("funsd", sum(k.find("/funsd/") > 0 for k in truth), "of", len(want["funsd"]))

left = set(want["cord"])
for split in ("test", "validation", "train"):
    for ex in load_dataset("naver-clova-ix/cord-v2", split=split, streaming=True):
        k = h(ex["image"])
        if k not in left:
            continue
        left.discard(k)
        W, H = ex["image"].size
        gt = json.loads(ex["ground_truth"])
        words = []
        for line in gt.get("valid_line", []):
            for w in line.get("words", []):
                q = w["quad"]
                words.append({"text": w["text"], "box": [q["x1"] / W, q["y1"] / H, q["x3"] / W, q["y3"] / H]})
        truth[want["cord"][k]] = words
        if not left:
            break
    if not left:
        break
print("cord", sum(k.find("/cord/") > 0 for k in truth), "of", len(want["cord"]))
json.dump(truth, open("data/ocr_truth.json", "w", encoding="utf-8"), ensure_ascii=False)
