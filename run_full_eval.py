"""Full profile: text reasoner (Qwen3-0.6B) on :8081 + vision specialist (SmolVLM-256M) on :8082. Evaluates the whole system."""
import os, re, subprocess, sys, time
import httpx

os.environ["OMNI_VISION_URL"] = "http://127.0.0.1:8082"
os.environ["OMNI_VISION"] = "0"
from omni import improve
from omni.agent import Agent, Workspace
from omni.budget import Budget
from omni.evalreal import evaluate_real
from omni.llm import LLM

EXE = os.path.abspath("runtime/llama/llama-server.exe")
procs = [
    subprocess.Popen([EXE, "-m", "models/Qwen3-0.6B-Q4_K_M.gguf", "-t", "6", "-np", "2", "-c", "4096", "--port", "8081", "--chat-template-kwargs", '{"enable_thinking":false}'], stdout=open("server_text.log", "w"), stderr=subprocess.STDOUT),
    subprocess.Popen([EXE, "-m", "models/vision/SmolVLM-256M-Instruct-Q8_0.gguf", "--mmproj", "models/vision/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf", "-t", "6", "-np", "2", "-c", "8192", "--port", "8082"], stdout=open("server_vision.log", "w"), stderr=subprocess.STDOUT),
]
try:
    for port in (8081, 8082):
        for _ in range(90):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2)
    live = improve.load_live()
    stages = os.environ.get("STAGES", "single,holdout,photo")
    if "single" in stages:
        r = evaluate_real(live, evalset="evalset_real.json")
        print(f"REAL single-page  ACC {r['overall']:.3f} near={r['near_misses']} by_cat { {k: round(v, 2) for k, v in r['by_cat'].items()} } {r['avg_seconds']:.1f}s/q", flush=True)
        for row in r["rows"]:
            if row["refined"] or row["conf"] or row["alt"]:
                print(f"   sub-agent: {'OK ' if row['ok'] else 'X  '} {row['q'][:46]:46} | {row['ans'][:34]!r} | {row['refined']} {row['conf']} {row['alt']}", flush=True)
    if "holdout" in stages:
        r = evaluate_real(live, evalset="evalset_real_holdout.json")
        print(f"REAL held-out     ACC {r['overall']:.3f} near={r['near_misses']} by_cat { {k: round(v, 2) for k, v in r['by_cat'].items()} } {r['avg_seconds']:.1f}s/q", flush=True)
    if "photo" in stages:
        cases = [("data/photos/red_circle.png", "What color is the circle?", ["red"]), ("data/photos/blue_square.png", "What color is the square?", ["blue"]),
                 ("data/photos/two_portraits.png", "How many people are in this picture?", ["2", "two"]),
                 ("data/photos/two_portraits.png", "Describe this picture.", ["men", "man", "people", "person", "portrait", "photo"])]
        ok = 0
        for path, q, gold in cases:
            ws = Workspace()
            d = ws.add(path)
            res = Agent(ws, LLM()).ask(q, Budget.make("balanced"), [d.id])
            hit = any(re.search(r"(?<![a-z])" + g + r"(?![a-z])", res["answer"].lower()) for g in gold)
            ok += hit
            print(f"   PHOTO {'OK ' if hit else 'X  '} {q:38} -> {res['answer'][:70]!r} [{res['strategy']}]", flush=True)
        print(f"PHOTO {ok}/{len(cases)}", flush=True)
finally:
    for p in procs:
        p.terminate()
