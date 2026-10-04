import json, threading, time, urllib.request, sys, base64, io
from PIL import Image, ImageDraw
img = Image.new("RGB", (512, 256), "white"); ImageDraw.Draw(img).text((20, 40), "Total due: $1,250.00", fill="black")
b = io.BytesIO(); img.save(b, "PNG"); uri = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
def go(kind, n):
    out = {}
    def f(i):
        c = [{"type": "text", "text": "Say hello in five words."}] if kind == "text" else [{"type": "image_url", "image_url": {"url": uri}}, {"type": "text", "text": "What is the total?"}]
        body = json.dumps({"messages": [{"role": "user", "content": c}], "max_tokens": 24, "temperature": 0}).encode()
        t = time.time()
        try:
            urllib.request.urlopen(urllib.request.Request("http://localhost:8081/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=40).read(); out[i] = round(time.time() - t, 1)
        except Exception as e: out[i] = f"ERR {type(e).__name__}"
    ts = [threading.Thread(target=f, args=(i,)) for i in range(n)]; [x.start() for x in ts]; [x.join() for x in ts]
    print(kind, n, out, flush=True)
go("image", 3); go("image", 4)
