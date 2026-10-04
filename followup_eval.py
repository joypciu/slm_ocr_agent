"""Chat follow-ups on the varied benchmark: for each document, ask its first question, then one of its other questions phrased as a follow-up
('and the subtotal?', "what about 'DATE'?"), scored against that question's gold.  OMNI_FOLLOWUP=0 switches the resolver off for comparison.
   EVALSET=data/evalset_varied_fresh.json python followup_eval.py"""
import json, os, re, time
from collections import defaultdict
os.environ["OMNI_VISION"] = "0"
from omni import improve
from omni.agent import Agent, Workspace
from omni.budget import Budget
from omni.evalrun import _NoMemory
from omni.llm import LLM
from eval_varied import correct


def as_followup(q, k):
    m = re.match(r"^what is (the value of )?(.+?)\?$", q, re.I)
    if not m:
        return None
    rest = m.group(2)
    if m.group(1):
        return f"and {rest}?" if k % 2 else f"what about {rest}?"
    return f"and {rest}?" if k % 2 else f"what about {rest}?"


items = json.load(open(os.environ.get("EVALSET", "data/evalset_varied_fresh.json")))
by_doc = defaultdict(list)
for it in items:
    by_doc[it["doc"]].append(it)
ws = Workspace()
ag = Agent(ws, LLM())
ag.live, ag.memory = improve.load_live(), _NoMemory()
res, t0, k = defaultdict(list), time.time(), 0
for doc, its in by_doc.items():
    d = ws.add(doc)
    for second in its[1:]:
        f = as_followup(second["q"], k)
        if not f:
            continue
        k += 1
        ag.ask(its[0]["q"], Budget.make("balanced"), [d.id])
        r = ag.ask(f, Budget.make("balanced"), [d.id])
        ok = correct(second["gold"], r["answer"])
        res[second["cat"]].append(ok)
        if not ok and k % 7 == 0:
            print(f"   miss: {f!r} -> understood {r.get('understood_as')!r} -> {r['answer'][:50]!r}")
allv = [x for v in res.values() for x in v]
print(f"FOLLOWUP resolver={os.environ.get('OMNI_FOLLOWUP', '1')} n={len(allv)} ACC {sum(allv) / len(allv):.3f}  " +
      "  ".join(f"{c} {sum(v) / len(v):.3f}" for c, v in res.items()) + f" | {time.time() - t0:.0f}s")
