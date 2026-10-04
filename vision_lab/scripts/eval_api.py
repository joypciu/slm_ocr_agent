import json, sys, time, base64, urllib.request
from common import score
port, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60
data = json.load(open("data/eval.json"))[:N]; by = {}; t0 = time.time()
for i, ex in enumerate(data):
    uri = "data:image/jpeg;base64," + base64.b64encode(open(ex["image"], "rb").read()).decode()
    body = json.dumps({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": ex["q"] + "\nAnswer briefly."}]}], "max_tokens": 24, "temperature": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=120))
    pred = r["choices"][0]["message"]["content"].strip(); by.setdefault(ex["src"], []).append(score(pred, ex["a"]))
allv = [x for v in by.values() for x in v]
print("sec/example %.1f" % ((time.time() - t0) / len(data)), "| ACC overall %.3f" % (sum(allv) / len(allv)), {k: round(sum(v) / len(v), 2) for k, v in by.items()})
