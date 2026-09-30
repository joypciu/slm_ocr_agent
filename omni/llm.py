"""Thin client for the local llama.cpp servers. Every call is charged to the Budget.

Two possible servers: the text reasoner (OMNI_LLM_URL) and a vision specialist (OMNI_VISION_URL). With only one server that has an
image projector, set OMNI_VISION=1 and it is used for both. With a text-only server and no vision URL, vision is unavailable.
"""
import base64, io, os, time, httpx
from .budget import Budget, BudgetExceeded


class LLM:
    def __init__(self, base=None, timeout=120):
        self.base = base or os.environ.get("OMNI_LLM_URL", "http://127.0.0.1:8081")
        self.vbase = os.environ.get("OMNI_VISION_URL") or None            # separate vision specialist, if any
        self._same_server_vision = os.environ.get("OMNI_VISION", "0") == "1"   # the one server has an image projector
        self.timeout = timeout
        self._vok = (0.0, False)

    @property
    def vision(self) -> bool:
        if self.vbase:
            t, ok = self._vok
            if time.time() - t > 20:  # re-check now and then: the specialist may be started or stopped at any time
                try:
                    ok = httpx.get(f"{self.vbase}/health", timeout=2).status_code == 200
                except Exception:
                    ok = False
                self._vok = (time.time(), ok)
            return ok
        return self._same_server_vision

    def alive(self):
        try:
            return httpx.get(f"{self.base}/health", timeout=3).status_code == 200
        except Exception:
            return False

    @staticmethod
    def estimate(messages):
        """Rough prompt size in tokens: ~3.2 characters per token, plus a flat cost per image."""
        n = 0
        for m in messages:
            c = m["content"]
            if isinstance(c, str):
                n += len(c) / 3.2 + 4
            else:
                for part in c:
                    n += (len(part.get("text", "")) / 3.2 + 4) if part.get("type") == "text" else 350
        return int(n)

    def chat(self, messages, budget: Budget, max_tokens=256, schema=None, temperature=0.0, why="llm", vision=False):
        left = int(budget.left("llm_tokens"))
        est = self.estimate(messages)
        if est + 32 > left:  # refuse before spending anything; the caller may ask for an extension
            raise BudgetExceeded("llm_tokens", est + 48, left)
        base = self.vbase if (vision and self.vbase) else self.base
        body = {"messages": messages, "max_tokens": max(16, min(max_tokens, left - est)), "temperature": temperature}
        if schema:
            body["response_format"] = {"type": "json_object", "schema": schema}
        r = httpx.post(f"{base}/v1/chat/completions", json=body, timeout=self.timeout)
        r.raise_for_status()
        j = r.json()
        budget.charge("llm_tokens", j.get("usage", {}).get("total_tokens", est), why)
        return j["choices"][0]["message"]["content"].strip()

    @staticmethod
    def image_part(pil_image, max_side=1024):
        im = pil_image.convert("RGB")
        im.thumbnail((max_side, max_side))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=88)
        return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()}}
