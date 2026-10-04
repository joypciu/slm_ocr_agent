import json, sys, time, torch
from PIL import Image
from transformers import AutoModelForImageTextToText
from common import *
torch.set_num_threads(6)
path = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
proc = load_proc(path); model = AutoModelForImageTextToText.from_pretrained(path, torch_dtype=torch.float32).eval()
data = json.load(open("data/eval.json"))[:N]; by = {}; t0 = time.time(); ntok = 0
for i, ex in enumerate(data):
    inp = proc(text=build(proc, ex["q"]), images=[Image.open(ex["image"])], return_tensors="pt")
    ntok = inp["input_ids"].shape[1]
    with torch.no_grad(): out = model.generate(**inp, max_new_tokens=24, do_sample=False)
    pred = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    s = score(pred, ex["a"]); by.setdefault(ex["src"], []).append(s)
    if i < 6: print(f"  Q:{ex['q'][:60]} | gold:{ex['a']} | pred:{pred[:40]} | {s}", flush=True)
allv = [x for v in by.values() for x in v]
print(path, "| prompt tokens:", ntok, "| sec/example: %.1f" % ((time.time() - t0) / len(data)))
print("ACC overall %.3f" % (sum(allv) / len(allv)), {k: round(sum(v) / len(v), 2) for k, v in by.items()})
