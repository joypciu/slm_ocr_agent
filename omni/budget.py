"""Two-tier resource budget shared by the user and the agent.

* user limits  - hard ceilings set by the user; nobody can exceed them without the user granting more.
* agent allowance - a soft cap the agent sets for itself (<= user limits); it can spend less, throttle
  itself, or ask for an extension, but only the user (or an auto-grant rule the user set) can raise the ceiling.

Resources: llm_tokens (prompt+completion), seconds, tool_calls, vlm_looks (image encodes), ocr_pages.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field

RESOURCES = ("llm_tokens", "seconds", "tool_calls", "vlm_looks", "ocr_pages")

PRESETS = {  # user picks a preset or overrides individual numbers
    "frugal":   {"llm_tokens": 1500,  "seconds": 20,  "tool_calls": 4,  "vlm_looks": 0, "ocr_pages": 2},
    "balanced": {"llm_tokens": 4000,  "seconds": 60,  "tool_calls": 8,  "vlm_looks": 4, "ocr_pages": 6},
    "thorough": {"llm_tokens": 12000, "seconds": 180, "tool_calls": 20, "vlm_looks": 10, "ocr_pages": 20},
}

class BudgetExceeded(Exception):
    def __init__(self, resource: str, need: float, left: float):
        super().__init__(f"{resource}: need {need:g}, only {left:g} left")
        self.resource, self.need, self.left = resource, need, left

@dataclass
class Budget:
    user_limits: dict
    auto_grant: dict = field(default_factory=dict)   # resource -> max extra the agent may take WITHOUT asking (user-set)
    allowance: dict = field(default_factory=dict)    # agent's own soft caps (<= user limits)
    spent: dict = field(default_factory=lambda: {r: 0.0 for r in RESOURCES})
    log: list = field(default_factory=list)          # audit trail of every spend / grant / request
    pending: list = field(default_factory=list)      # extension requests waiting for the user
    _t0: float = field(default_factory=time.time)

    @classmethod
    def make(cls, preset="balanced", **overrides):
        limits = dict(PRESETS[preset]); limits.update({k: v for k, v in overrides.items() if k in RESOURCES and v is not None})
        return cls(user_limits=limits)

    def new_turn(self):
        """Wall-clock time is per request; every other resource keeps accumulating across a session."""
        self._t0 = time.time()

    # ---- queries
    def cap(self, r):  # effective cap = agent allowance if set, else user limit; never above the user limit
        return min(self.allowance.get(r, self.user_limits[r]), self.user_limits[r])
    def used(self, r):
        return (time.time() - self._t0) if r == "seconds" else self.spent[r]
    def left(self, r): return max(0.0, self.cap(r) - self.used(r))
    def can(self, r, n=1): return self.left(r) >= n - 1e-9
    def snapshot(self):
        return {r: {"used": round(self.used(r), 1), "cap": self.cap(r), "limit": self.user_limits[r]} for r in RESOURCES}

    # ---- spending
    def spend(self, r, n, why=""):
        if r != "seconds" and self.left(r) < n:
            raise BudgetExceeded(r, n, self.left(r))
        if r == "seconds" and self.left(r) <= 0:
            raise BudgetExceeded(r, n, 0)
        if r != "seconds": self.spent[r] += n
        self.log.append({"t": round(time.time() - self._t0, 2), "op": "spend", "r": r, "n": n, "why": why})

    def charge(self, r, n, why=""):
        """Record what was actually used (after the fact). Never raises; an overshoot is logged, not hidden."""
        over = max(0.0, self.used(r) + n - self.cap(r)) if r != "seconds" else 0.0
        if r != "seconds":
            self.spent[r] += n
        self.log.append({"t": round(time.time() - self._t0, 2), "op": "charge", "r": r, "n": n, "why": why, **({"overshoot": round(over, 1)} if over else {})})

    # ---- agent powers
    def throttle(self, r, new_cap, why=""):
        """Agent lowers its own allowance (e.g. a simple question needs few tokens). Can never raise above the user limit."""
        new_cap = min(new_cap, self.user_limits[r]); self.allowance[r] = new_cap
        self.log.append({"t": round(time.time() - self._t0, 2), "op": "throttle", "r": r, "cap": new_cap, "why": why})

    def dispense(self, r, share: float, why=""):
        """Reserve a fraction of what is left for one step; returns the amount so the step can self-limit."""
        amt = round(self.left(r) * share, 1); self.log.append({"op": "dispense", "r": r, "n": amt, "why": why}); return amt

    def request_extension(self, r, extra, reason):
        """Agent asks for more. Auto-granted only inside the user's auto_grant rule; otherwise queued for the user."""
        room = self.auto_grant.get(r, 0)
        if extra <= room:
            self.auto_grant[r] = room - extra; self.user_limits[r] += extra
            if r in self.allowance: self.allowance[r] += extra
            self.log.append({"op": "auto_grant", "r": r, "n": extra, "why": reason}); return True
        self.pending.append({"id": len(self.pending), "r": r, "extra": extra, "reason": reason, "status": "waiting"})
        self.log.append({"op": "request", "r": r, "n": extra, "why": reason}); return False

    # ---- user powers
    def resolve(self, req_id, approve: bool):
        req = self.pending[req_id]; req["status"] = "approved" if approve else "denied"
        if approve:
            self.user_limits[req["r"]] += req["extra"]
            if req["r"] in self.allowance: self.allowance[req["r"]] += req["extra"]
        self.log.append({"op": "user_" + req["status"], "r": req["r"], "n": req["extra"]})
