"""The session answer cache: only exact repeats, only verified answers, never after the documents or their reading changed, never after 'bad' feedback.
Uses a stub model, no servers. Run:  python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omni import improve  # noqa: E402
from omni.agent import Agent, Workspace  # noqa: E402
from omni.budget import Budget  # noqa: E402
from omni.evalrun import _NoMemory  # noqa: E402


class StubLLM:
    vision = False

    def __init__(self):
        self.calls = 0

    def chat(self, msgs, b, max_tokens=96, why="", vision=False, **kw):
        self.calls += 1
        b.spend("llm_tokens", 50, why)
        text = msgs[-1]["content"]
        if "deadline" in text.lower():
            return "The deadline is 14 March 2027."
        return "Not found in the documents."

    def estimate(self, *a, **k):
        return 50


def make(text):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "note.txt")
    open(p, "w", encoding="utf-8").write(text)
    ws = Workspace()
    llm = StubLLM()
    ag = Agent(ws, llm)
    ag.live, ag.memory = improve.Router(), _NoMemory()
    return ws, ag, llm, ws.add(p), p


class AnswerCache(unittest.TestCase):
    TEXT = "Project Falcon status report.\nThe submission deadline is 14 March 2027 for all teams.\nBudget owner: Karim Rahman."

    def test_exact_repeat_is_served_without_a_model_call(self):
        ws, ag, llm, d, _ = make(self.TEXT)
        r1 = ag.ask("What is the submission deadline?", Budget.make("balanced"), [d.id])
        n = llm.calls
        r2 = ag.ask("what is the submission deadline", Budget.make("balanced"), [d.id])
        self.assertTrue(r1["verified"])
        self.assertEqual(r2["source"], "session cache")
        self.assertEqual(r2["answer"], r1["answer"])
        self.assertEqual(llm.calls, n)

    def test_a_different_question_is_never_served_from_the_cache(self):
        ws, ag, llm, d, _ = make(self.TEXT)
        ag.ask("What is the submission deadline?", Budget.make("balanced"), [d.id])
        r = ag.ask("What is the submission deadline for team 2?", Budget.make("balanced"), [d.id])
        self.assertNotEqual(r["source"], "session cache")

    def test_bad_feedback_drops_the_cached_answer(self):
        ws, ag, llm, d, _ = make(self.TEXT)
        r1 = ag.ask("What is the submission deadline?", Budget.make("balanced"), [d.id])
        ag.feedback(r1["id"], "bad")
        r2 = ag.ask("What is the submission deadline?", Budget.make("balanced"), [d.id])
        self.assertNotEqual(r2["source"], "session cache")

    def test_changed_evidence_is_not_reused(self):
        ws, ag, llm, d, _ = make(self.TEXT)
        r1 = ag.ask("What is the submission deadline?", Budget.make("balanced"), [d.id])
        self.assertTrue(r1["verified"])
        page = ws.docs[d.id].pages[0]
        page.lines = [(t.replace("14 March 2027", "21 March 2027"), b) for t, b in page.lines]   # the reading of the page changed
        r2 = ag.ask("What is the submission deadline?", Budget.make("balanced"), [d.id])
        self.assertNotEqual(r2["source"], "session cache")


if __name__ == "__main__":
    unittest.main()
