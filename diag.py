"""python diag.py <real doc name e.g. card_p3> "<question>" -- show what the agent saw and why it answered as it did."""
import re, sys
from omni.agent import Agent, Workspace, META, entities
from omni.budget import Budget
from omni.llm import LLM

name, q = sys.argv[1], sys.argv[2]
ws = Workspace()
d = ws.add(f"data/real/{name}.pdf")
ag = Agent(ws, LLM())
ag.memory = type("M", (), {"recall": lambda *a, **k: None, "remember": lambda *a, **k: None})()
b = Budget.make("balanced")
r = ag.ask(q, b, [d.id])
print("PAGE SOURCE:", d.pages[0].source, "| lines:", len(d.pages[0].lines))
print("ROWS:")
for t, box in d.pages[0].lines[:40]:
    print("   ", t[:110])
print("\nQ:", q)
print("unseen terms :", ws.px.unseen(q), "| absent_share %.2f" % ws.absent_share(q), "| best_coverage %.2f" % ws.px.best_coverage(q))
print("evidence     :", [(round(s, 2), c["text"][:120]) for s, c in ws.evidence(q, [d.id])])
print("ANSWER       :", r["answer"][:200])
print("strategy     :", r["strategy"], "| tried:", r.get("tried"), "| support:", r["support"])
print("steps        :", r["steps"])
