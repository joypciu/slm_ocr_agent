import json, threading, time, urllib.request, base64, io, random, sys
from PIL import Image, ImageDraw
PORT = sys.argv[1] if len(sys.argv) > 1 else "8081"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 4; ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
def mk(k):
    img = Image.new("RGB", (512, 256), (random.randint(200,255),)*3); ImageDraw.Draw(img).text((20, 40), f"Order {k} Total due: ${k*137%9000}.00", fill="black")
    b = io.BytesIO(); img.save(b, "PNG"); return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
for r in range(ROUNDS):
    out = {}
    def f(i):
        body = json.dumps({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": mk(r*10+i+1)}}, {"type": "text", "text": "What is the total due?"}]}], "max_tokens": 32, "temperature": 0}).encode()
        t = time.time()
        try:
            j = json.load(urllib.request.urlopen(urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=90)); out[i] = (round(time.time() - t, 1), j["choices"][0]["message"]["content"].strip()[:30])
        except Exception as e: out[i] = f"ERR {type(e).__name__}"
    ts = [threading.Thread(target=f, args=(i,)) for i in range(N)]; t0 = time.time(); [x.start() for x in ts]; [x.join() for x in ts]
    print(f"round {r}: wall {time.time()-t0:.1f}s", out, flush=True)
