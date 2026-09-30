"""Benchmark candidate text models (text-only servers) on the real-document eval. One server at a time on port 8082."""
import json, os, subprocess, sys, time
import httpx

os.environ["OMNI_LLM_URL"] = "http://127.0.0.1:8082"
os.environ["OMNI_VISION"] = "0"
from omni import improve
from omni.evalreal import evaluate_real
from omni.llm import LLM

SERVER = os.path.abspath("runtime/llama/llama-server.exe")
CANDS = [  # (label, file, extra server args)
    ("qwen3-0.6b-q4km", "Qwen3-0.6B-Q4_K_M.gguf", ["--chat-template-kwargs", '{"enable_thinking":false}']),
    ("qwen2.5-0.5b-q4_0", "qwen2.5-0.5b-instruct-q4_0.gguf", []),
    ("smollm2-360m-q8", "SmolLM2-360M-Instruct-Q8_0.gguf", []),
    ("gemma3-270m-q8", "gemma-3-270m-it-Q8_0.gguf", []),
    ("lfm2-350m-q8", "LFM2-350M-Q8_0.gguf", []),
]
only = sys.argv[1:] or [c[0] for c in CANDS]
results = {}
for label, fname, extra in CANDS:
    if label not in only:
        continue
    path = os.path.join("models", "candidates", fname)
    while "done " + fname not in open("dl.log").read():
        time.sleep(5)
    size = os.path.getsize(path) / 1048576
    srv = subprocess.Popen([SERVER, "-m", path, "-t", "6", "-np", "1", "-c", "4096", "--port", "8082", *extra], stdout=open("server_cand.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            try:
                if httpx.get("http://127.0.0.1:8082/health", timeout=2).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2)
        t = time.time()
        r = evaluate_real(improve.load_live(), llm=LLM(), verbose=False)
        results[label] = {"size_mb": round(size), "acc": round(r["overall"], 3), "by_cat": {k: round(v, 2) for k, v in r["by_cat"].items()}, "sec": round((time.time() - t) / r["n"], 2), "strategies": r["strategies"]}
        json.dump(r["rows"], open(f"data/bench_{label}.json", "w"), indent=1)
        print(label, json.dumps(results[label]), flush=True)
    finally:
        srv.terminate()
        srv.wait(timeout=20)
json.dump(results, open("data/bench_models.json", "w"), indent=1)
print("ALL DONE", flush=True)
