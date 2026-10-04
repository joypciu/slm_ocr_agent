import base64, io, time, json, threading, urllib.request, sys
from PIL import Image, ImageDraw
img = Image.new("RGB", (512, 256), "white"); ImageDraw.Draw(img).text((20, 40), "INVOICE #4821  Total due: $1,250.00  Date: 2026-03-14", fill="black")
b = io.BytesIO(); img.save(b, "PNG"); uri = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
PORT = sys.argv[1] if len(sys.argv) > 1 else "8081"
def ask(res, i):
    body = json.dumps({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": "What is the total due?"}]}], "max_tokens": 48, "temperature": 0}).encode()
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=300))
        res[i] = (time.time() - t, r["choices"][0]["message"]["content"], r["usage"]["completion_tokens"])
    except Exception as e:
        res[i] = (time.time() - t, f"ERR {e}", 0)
for n in (1, 4):
    res = {}; ts = [threading.Thread(target=ask, args=(res, i)) for i in range(n)]
    t = time.time(); [x.start() for x in ts]; [x.join() for x in ts]; wall = time.time() - t
    print(f"{n} concurrent: wall {wall:.1f}s, per-request {[round(v[0],1) for v in res.values()]}, tokens {sum(v[2] for v in res.values())}, replies: {[v[1][:40] for v in res.values()]}")
