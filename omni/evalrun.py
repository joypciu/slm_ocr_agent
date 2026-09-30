"""Frozen-eval runner shared by the CLI and the promotion gate."""
from __future__ import annotations
import json, os, re, time
from collections import defaultdict
from .agent import Agent, Workspace, NOT_FOUND
from .budget import Budget
from .llm import LLM
from . import improve

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def norm(s):
    return re.sub(r"[\s,]", "", str(s).lower())


def correct(item, answer):
    if item["gold"] is None:
        return bool(NOT_FOUND.search(answer))
    return norm(item["gold"]) in norm(answer)


class _NoMemory:
    def recall(self, *a, **k): return None
    def remember(self, *a, **k): pass


def evaluate(router, pdf="synth_text.pdf", n=None, evalset="evalset.json", preset="balanced", llm=None, verbose=False, **budget_kw):
    ws = Workspace()
    doc = ws.add(os.path.join(DATA, pdf))
    ag = Agent(ws, llm or LLM())
    ag.live, ag.memory = router, _NoMemory()
    items = json.load(open(os.path.join(DATA, evalset)))[:n]
    by, strat, tokens, secs, rows = defaultdict(list), defaultdict(int), 0, 0.0, []
    for it in items:
        b = Budget.make(preset, **budget_kw)
        r = ag.ask(it["q"], b, [doc.id])
        ok = correct(it, r["answer"])
        by[it["cat"]].append(ok)
        strat[r["strategy"]] += 1
        tokens += r["budget"]["llm_tokens"]["used"]
        secs += r["seconds"]
        rows.append({"q": it["q"], "gold": it["gold"], "ans": r["answer"], "ok": ok, "strategy": r["strategy"], "support": r["support"]})
        if verbose:
            print(("OK  " if ok else "FAIL"), r["strategy"], "|", it["q"][:60], "| gold:", it["gold"], "| ans:", r["answer"][:50])
    allv = [x for v in by.values() for x in v]
    return {"overall": sum(allv) / len(allv), "by_cat": {k: sum(v) / len(v) for k, v in by.items()}, "n": len(allv),
            "strategies": dict(strat), "avg_tokens": tokens / len(items), "avg_seconds": secs / len(items), "rows": rows}
