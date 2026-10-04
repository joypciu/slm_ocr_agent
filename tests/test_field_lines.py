from omni.fields import line_fields
from omni.agent import Agent, Workspace
from omni.budget import Budget
from types import SimpleNamespace


def test_unique_fields_keep_values_on_their_own_lines():
    fields = line_fields("Owner: Maya\nTotal: 1200\nDate: 2026-10-04\nhttps://example.com\n")
    assert [(f["label"], f["value"]) for f in fields] == [("Owner", "Maya"), ("Total", "1200"), ("Date", "2026-10-04")]


def test_repeated_records_remain_separate():
    assert [f["value"] for f in line_fields("Total: 100\nTotal: 200")] == ["100", "200"]


def test_scan_extraction_stops_at_budget_and_keeps_ocr_unverified(monkeypatch):
    ws = Workspace()
    pages = [SimpleNamespace(n=n, source="none", text="") for n in (1, 2)]
    document = SimpleNamespace(id="scan", name="scan.pdf", kind="pdf", pages=pages)
    ws.docs[document.id] = document
    reads = []

    def ocr(doc, number):
        reads.append(number)
        doc.pages[number - 1].source = "ocr"
        doc.pages[number - 1].text = f"Total: {number * 100}"

    monkeypatch.setattr(ws, "ocr", ocr)
    agent = Agent.__new__(Agent)
    agent.ws = ws
    budget = Budget.make("frugal", ocr_pages=1)
    result = agent.extract("Extract fields", budget, ["scan"], "test")
    assert reads == [1]
    assert result["fields"][0]["value"] == "100"
    assert result["fields"][0]["verified"] is False
    assert "budget_note" in result
    assert budget.pending[0]["r"] == "ocr_pages"


def test_no_ocr_pages_charged_when_tool_budget_is_empty(monkeypatch):
    ws = Workspace()
    ws.docs["scan"] = SimpleNamespace(id="scan", name="scan.pdf", kind="pdf", pages=[SimpleNamespace(n=1, source="none", text="")])
    agent = Agent.__new__(Agent)
    agent.ws = ws
    budget = Budget.make("frugal", tool_calls=0)
    result = agent.extract("Extract fields", budget, ["scan"], "test")
    assert result["fields"] == []
    assert budget.spent["ocr_pages"] == 0
    assert budget.pending[0]["r"] == "tool_calls"
