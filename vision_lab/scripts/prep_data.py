import json, os, itertools
from datasets import load_dataset
os.makedirs("data/img", exist_ok=True)
SRC = {"docvqa": 190, "chartqa": 150, "textvqa": 170}   # train per source
EVAL_PER = 20
train, evals = [], []
for name, n in SRC.items():
    ds = load_dataset("HuggingFaceM4/the_cauldron", name, split="train", streaming=True)
    k = 0
    for ex in ds:
        if len(ex["images"]) != 1: continue
        im = ex["images"][0].convert("RGB")
        if max(im.size) > 1400: im.thumbnail((1400, 1400))
        for t in ex["texts"][:1]:                      # first QA pair per image
            q = t["user"].split("\n")[0].strip(); a = t["assistant"].strip()
            if not q or not a or len(a) > 80: continue
            p = f"data/img/{name}_{k}.jpg"; im.save(p, quality=88)
            (evals if k < EVAL_PER else train).append({"image": p, "q": q, "a": a, "src": name}); k += 1
        if k >= n + EVAL_PER: break
    print(name, k, flush=True)
json.dump(train, open("data/train.json", "w")); json.dump(evals, open("data/eval.json", "w"))
print("train", len(train), "eval", len(evals))
