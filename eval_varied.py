"""Evaluate the pipeline on the varied-forms benchmark (data/evalset_varied.json, made by build_varied.py).
   python eval_varied.py [out.json] [cat ...]   needs the text model on :8081"""
import difflib, json, os, re, sys, time
from collections import defaultdict
os.environ["OMNI_VISION"] = "0"
from omni import improve
from omni.agent import Agent, Workspace, NOT_FOUND
from omni.budget import Budget
from omni.evalrun import _NoMemory
from omni.llm import LLM


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def correct(gold, ans):
    if "digits" in gold:
        toks = re.findall(r"\d[\d.,/\-]*", ans)
        return any(re.sub(r"\D", "", t) == gold["digits"] for t in toks)
    if "words" in gold:
        low = ans.lower()
        return all(w.lower() in low for w in gold["words"])
    g, a = norm(gold["text"]), norm(ans)
    return bool(g) and (g in a or difflib.SequenceMatcher(None, g, a).ratio() >= 0.8)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/varied_results.json"
    only = set(sys.argv[2:])
    items = [i for i in json.load(open(os.environ.get("EVALSET", "data/evalset_varied.json"))) if not only or i["cat"] in only]
    ws = Workspace()
    ag = Agent(ws, LLM())
    ag.live, ag.memory = improve.load_live(), _NoMemory()
    docs, by, rows, t0 = {}, defaultdict(list), [], time.time()
    for k, it in enumerate(items):
        if it["doc"] not in docs:
            docs[it["doc"]] = ws.add(it["doc"])
        d = docs[it["doc"]]
        r = ag.ask(it["q"], Budget.make("balanced"), [d.id])
        ok = correct(it["gold"], r["answer"])
        by[it["cat"]].append(ok)
        rows.append({"doc": it["doc"], "cat": it["cat"], "q": it["q"], "gold": it["gold"], "ans": r["answer"], "ok": ok, "strategy": r["strategy"], "support": r["support"], "steps": r.get("steps", [])[:3]})
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(items)} done, running accuracy {sum(x['ok'] for x in rows) / len(rows):.3f}", flush=True)
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    allv = [x for v in by.values() for x in v]
    print(f"\nVARIED n={len(allv)} ACC {sum(allv) / len(allv):.3f}  " + "  ".join(f"{c} {sum(v) / len(v):.3f} ({sum(v)}/{len(v)})" for c, v in by.items()) + f"  | {(time.time() - t0) / len(items):.1f}s/q")
    strat = defaultdict(lambda: [0, 0])
    for r in rows:
        strat[r["strategy"]][0] += r["ok"]
        strat[r["strategy"]][1] += 1
    print("by strategy:", {k: f"{v[0]}/{v[1]}" for k, v in strat.items()})
