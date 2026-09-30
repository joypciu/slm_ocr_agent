import sys
from omni import improve
from omni.evalrun import evaluate

pdf = sys.argv[1] if len(sys.argv) > 1 else "synth_text.pdf"
which = sys.argv[2] if len(sys.argv) > 2 else "live"
n = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "all" else None
preset = sys.argv[4] if len(sys.argv) > 4 else "balanced"
ocr = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "-" else None
evalset = sys.argv[6] if len(sys.argv) > 6 else "evalset.json"
router = improve.load_live() if which == "live" else improve.load_candidate()
r = evaluate(router, pdf, n, evalset=evalset, preset=preset, verbose=True, ocr_pages=ocr)
print(f"\n{pdf} [{which} router v{router.version}, {preset}, ocr_pages={ocr}] n={r['n']}  ACC {r['overall']:.3f}  by_cat { {k: round(v, 2) for k, v in r['by_cat'].items()} }")
print(f"strategies {r['strategies']}  avg tokens {r['avg_tokens']:.0f}  avg seconds {r['avg_seconds']:.1f}")
