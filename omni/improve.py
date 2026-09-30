"""Self-improvement layer. Nothing here retrains the model; it improves *how the agent works*:

* Router      - linear contextual bandit choosing the answer strategy (text / ocr / look) from cheap features.
* Memory      - answers the user confirmed or corrected, replayed at zero token cost on repeat questions.
* Gate        - the router that learns from feedback is a CANDIDATE. It only replaces the LIVE router after it
                beats it on a frozen eval set with no per-category regression. Bad feedback cannot silently degrade serving.
Every file is plain JSON so the user can read, edit, or roll back a policy.
"""
from __future__ import annotations
import json, os, random, re, shutil, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
POL = os.path.join(HERE, "..", "policy")
os.makedirs(POL, exist_ok=True)
ACTIONS = ["fields", "text", "ocr", "look"]
FEATURES = ["bias", "need_ocr", "visual_kw", "extract_kw", "q_len", "top_score", "vlm_left", "tokens_left", "label_match"]

# hand-set priors so the very first answer is sensible; learning then moves them
PRIOR = {
    "fields": [0.55, -0.50, -0.50, 0.00, 0.00, 0.20, 0.0, 0.00, 0.90],
    "text":   [0.30, -0.60, -0.30, 0.00, 0.00, 0.60, 0.0, 0.10, 0.00],
    "ocr":    [0.10, 0.70, 0.00, 0.00, 0.00, -0.50, 0.0, 0.00, 0.00],
    "look":   [-0.20, 0.10, 0.90, 0.00, 0.00, -0.20, 0.30, 0.00, 0.00],
}


def _load(name, default):
    p = os.path.join(POL, name)
    return json.load(open(p)) if os.path.exists(p) else default


_SAVE_LOCK = threading.Lock()


def _save(name, obj):
    """Atomic, thread-safe write. Persistence problems must never break the request that triggered them."""
    p = os.path.join(POL, name)
    with _SAVE_LOCK:
        tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump(obj, fh, indent=1)
            os.replace(tmp, p)
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass


class Router:
    def __init__(self, theta=None, version=0):
        self.theta = theta or {a: list(v) for a, v in PRIOR.items()}
        self.version, self.n_updates = version, 0

    def scores(self, x):
        return {a: sum(w * xi for w, xi in zip(self.theta[a], x)) for a in ACTIONS}

    def rank(self, x, explore=0.0, rng=random):
        s = self.scores(x)
        order = sorted(ACTIONS, key=lambda a: -s[a])
        if explore and rng.random() < explore:
            order.insert(0, order.pop(rng.randrange(len(order))))
        return order, s

    def update(self, action, x, reward, lr=0.15):
        pred = sum(w * xi for w, xi in zip(self.theta[action], x))
        err = reward - pred
        self.theta[action] = [0.995 * w + lr * err * xi for w, xi in zip(self.theta[action], x)]
        self.n_updates += 1

    def to_json(self):
        return {"version": self.version, "theta": self.theta, "n_updates": self.n_updates}

    @classmethod
    def from_json(cls, j):
        r = cls(j["theta"], j.get("version", 0))
        r.n_updates = j.get("n_updates", 0)
        return r


def load_live():
    j = _load("live.json", None)
    return Router.from_json(j) if j else Router()


def load_candidate():
    j = _load("candidate.json", None)
    return Router.from_json(j) if j else load_live()


def save_candidate(r: Router):
    _save("candidate.json", r.to_json())


def promote(candidate_score: dict, live_score: dict, min_gain=0.0):
    """Gate: promote only if overall >= live + min_gain and no category drops. Returns (bool, reason)."""
    if candidate_score["overall"] < live_score["overall"] + min_gain:
        return False, f"overall {candidate_score['overall']:.3f} < live {live_score['overall']:.3f}"
    for cat, v in live_score["by_cat"].items():
        if candidate_score["by_cat"].get(cat, 0) < v - 1e-9:
            return False, f"regression in '{cat}': {candidate_score['by_cat'].get(cat, 0):.2f} < {v:.2f}"
    live = load_live()
    if os.path.exists(os.path.join(POL, "live.json")):
        shutil.copy(os.path.join(POL, "live.json"), os.path.join(POL, f"live.v{live.version}.bak.json"))
    cand = load_candidate()
    cand.version = live.version + 1
    _save("live.json", cand.to_json())
    return True, f"promoted v{cand.version}"


def rollback():
    baks = sorted(f for f in os.listdir(POL) if f.startswith("live.v") and f.endswith(".bak.json"))
    if not baks:
        return False
    shutil.copy(os.path.join(POL, baks[-1]), os.path.join(POL, "live.json"))
    os.remove(os.path.join(POL, baks[-1]))
    return True


# ---------------- memory of confirmed / corrected answers ----------------
def _norm(q):
    return set(re.findall(r"[a-z0-9]+", q.lower())) - {"the", "a", "an", "of", "is", "what", "please"}


class Memory:
    def __init__(self):
        self.items = _load("memory.json", [])

    def recall(self, question, doc_shas):
        qn = _norm(question)
        best = None
        for it in self.items:
            if set(it["docs"]) != set(doc_shas) or not it.get("verified"):
                continue
            n2 = _norm(it["q"])
            j = len(qn & n2) / max(len(qn | n2), 1)
            if j >= 0.85 and (best is None or j > best[0]):
                best = (j, it)
        return best[1] if best else None

    def remember(self, question, answer, doc_shas, verified=True, corrected=False):
        self.items = [i for i in self.items if not (i["q"] == question and set(i["docs"]) == set(doc_shas))]
        self.items.append({"q": question, "a": answer, "docs": sorted(doc_shas), "verified": verified, "corrected": corrected, "ts": time.time()})
        self.items = self.items[-500:]
        _save("memory.json", self.items)


def log_trace(rec: dict):
    os.makedirs(os.path.join(HERE, "..", "data"), exist_ok=True)
    with open(os.path.join(HERE, "..", "data", "traces.jsonl"), "a", encoding="utf8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------- learned trust in each reading source ----------------
class Trust:
    """When OCR and the vision sub-agent read the same field differently, whom to believe?
    Beta-style counts per (field kind, source), updated from user feedback. Starts neutral (a weak prior favouring OCR)."""

    def __init__(self):
        self.c = _load("trust.json", {})

    def _cell(self, kind, src):
        if kind.endswith("_hw"):  # handwriting-looking rows: OCR is weak there, so lean (moderately) on the vision reading until feedback says otherwise
            return self.c.setdefault(kind, {}).setdefault(src, [3.0, 6.0] if src == "ocr" else [3.6, 6.0])
        return self.c.setdefault(kind, {}).setdefault(src, [1.0 if src == "ocr" else 0.5, 1.0])  # [right + prior, total + prior]

    def score(self, kind, src):
        r, t = self._cell(kind, src)
        return r / t

    def prefer(self, kind, a="ocr", b="vision"):
        return a if self.score(kind, a) >= self.score(kind, b) else b

    def update(self, kind, src, right: bool):
        cell = self._cell(kind, src)
        cell[0] += 1.0 if right else 0.0
        cell[1] += 1.0
        _save("trust.json", self.c)


# ---------------- how many tokens each kind of question really needs ----------------
class TokenStats:
    """Running average of tokens actually spent per question class; the agent sets its own allowance from it."""
    DEFAULT = {"lookup": 1600, "list": 2000, "open": 2600, "visual": 2000}

    def __init__(self):
        self.s = _load("token_stats.json", {})

    def allowance(self, cls, user_limit):
        d = self.s.get(cls)
        base = self.DEFAULT[cls] if not d or d["n"] < 5 else max(0.6 * self.DEFAULT[cls], 2.0 * d["ema"])  # never collapse below 60% of the default
        return int(min(user_limit, base))

    def record(self, cls, used):
        d = self.s.setdefault(cls, {"n": 0, "ema": float(used)})
        d["n"] += 1
        d["ema"] = 0.8 * d["ema"] + 0.2 * float(used)
        _save("token_stats.json", self.s)
