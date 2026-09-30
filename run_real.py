import sys
from omni import improve
from omni.evalreal import evaluate_real

preset = sys.argv[1] if len(sys.argv) > 1 else "balanced"
router = improve.load_live()
r = evaluate_real(router, preset=preset, verbose=True)
print(f"\nREAL n={r['n']}  ACC {r['overall']:.3f}  by_cat { {k: round(v, 2) for k, v in r['by_cat'].items()} }")
print(f"strategies {r['strategies']}  avg seconds {r['avg_seconds']:.1f}")
