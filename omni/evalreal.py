"""Eval on real documents with hand-verified answers (each item names its own document)."""
from __future__ import annotations
import difflib, json, os, re, time
from collections import defaultdict
from .agent import Agent, Workspace, NOT_FOUND
from .budget import Budget
from .llm import LLM
from .evalrun import _NoMemory

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def _norm(s):
    return re.sub(r"\s+", " ", str(s).lower().replace(",", "")).strip()


def _has(a, g):
    g = _norm(g)
    if re.fullmatch(r"[a-z]+", g):
        return re.search(r"(?<![a-z])" + re.escape(g) + r"(?![a-z])", a) is not None
    return g in a


def correct(gold, answer):
    a = _norm(answer)
    if gold.get("none"):
        return bool(NOT_FOUND.search(answer))
    ok = True
    if "any" in gold:
        ok &= any(_has(a, g) for g in gold["any"])
    if "all" in gold:
        ok &= all(_has(a, g) for g in gold["all"])
    if "words" in gold:
        ok &= all(_has(a, g) for g in gold["words"])
    if "not_words" in gold:
        ok &= not any(_has(a, g) for g in gold["not_words"])
    if "not_any" in gold:
        ok &= not any(_has(a, g) for g in gold["not_any"])
    return ok


def near_miss(gold, answer):
    """Wrong by a letter or two (OCR misread of handwriting): 'Gcelshan' for 'Gulshan'. Reported separately, never counted as correct."""
    words = re.findall(r"[a-z]+", str(answer).lower())
    for g in gold.get("any", []) + gold.get("all", []):
        g = _norm(g)
        if re.fullmatch(r"[a-z]{5,}", g) and any(difflib.SequenceMatcher(None, g, w).ratio() >= 0.78 for w in words):
            return True
    return False


def evaluate_real(router, evalset="evalset_real.json", preset="balanced", llm=None, verbose=False, doc_override=None, **budget_kw):
    ws = Workspace()
    ag = Agent(ws, llm or LLM())
    ag.live, ag.memory = router, _NoMemory()
    items = json.load(open(os.path.join(DATA, evalset)))
    if doc_override:  # full-document mode: the unanswerable items are only valid for the single cut-out page
        items = [i for i in items if i["cat"] != "unanswerable"]
    docs, by, strat, rows = {}, defaultdict(list), defaultdict(int), []
    t0 = time.time()
    for it in items:
        path = os.path.join(DATA, "..", doc_override(it) if doc_override else it["doc"])
        if path not in docs:
            docs[path] = ws.add(path)
        d = docs[path]
        b = Budget.make(preset, **budget_kw)
        r = ag.ask(it["q"], b, [d.id])
        ok = correct(it["gold"], r["answer"])
        near = (not ok) and near_miss(it["gold"], r["answer"])
        by[it["cat"]].append(ok)
        strat[r["strategy"]] += 1
        rows.append({"q": it["q"], "cat": it["cat"], "ans": r["answer"], "ok": ok, "near": near, "refined": r.get("refined_by") or ("agree" if r.get("agreement") else None), "conf": r.get("confidence"), "alt": r.get("alternatives"), "strategy": r["strategy"], "support": r["support"], "doc": it["doc"]})
        if verbose:
            print(("OK  " if ok else "FAIL"), it["cat"][:11].ljust(11), r["strategy"].ljust(7), "|", it["q"][:52].ljust(52), "|", r["answer"][:60].replace("\n", " "))
    allv = [x for v in by.values() for x in v]
    return {"near_misses": sum(1 for r in rows if r["near"]), "overall": sum(allv) / len(allv), "by_cat": {k: sum(v) / len(v) for k, v in by.items()}, "n": len(allv),
            "strategies": dict(strat), "avg_seconds": (time.time() - t0) / len(items), "rows": rows}
