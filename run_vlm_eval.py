"""Score the CURRENT pipeline with whatever model server is already running on :8081 (used to compare SmolVLM vs Qwen3)."""
import os
os.environ.setdefault("OMNI_VISION", "1")
from omni import improve
from omni.evalreal import evaluate_real

live = improve.load_live()
for label, es in (("single-page (tuned-on)", "evalset_real.json"), ("held-out (seen once)", "evalset_real_holdout.json")):
    r = evaluate_real(live, evalset=es)
    print(f"{label:24} ACC {r['overall']:.3f} n={r['n']} near={r['near_misses']} by_cat { {k: round(v, 2) for k, v in r['by_cat'].items()} } strategies={r['strategies']} {r['avg_seconds']:.1f}s/q", flush=True)
