"""One-shot held-out real-document run with the shipped configuration (Qwen3-0.6B text model). Do not tune against this."""
import os, subprocess, time
import httpx
os.environ["OMNI_VISION"] = "0"
from omni import improve
from omni.evalreal import evaluate_real

srv = subprocess.Popen([os.path.abspath("runtime/llama/llama-server.exe"), "-m", "models/Qwen3-0.6B-Q4_K_M.gguf", "-t", "6", "-np", "1", "-c", "4096",
                        "--port", "8081", "--chat-template-kwargs", '{"enable_thinking":false}'], stdout=open("server_holdout.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(60):
        try:
            if httpx.get("http://127.0.0.1:8081/health", timeout=2).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(2)
    r = evaluate_real(improve.load_live(), evalset="evalset_real_holdout.json", verbose=True)
    print(f"\nHELD-OUT REAL n={r['n']} ACC {r['overall']:.3f} near-misses {r['near_misses']} by_cat { {k: round(v, 2) for k, v in r['by_cat'].items()} } {r['avg_seconds']:.1f}s/q", flush=True)
finally:
    srv.terminate()
