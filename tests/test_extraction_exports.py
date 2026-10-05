import csv
import io

from fastapi.testclient import TestClient

import server
from omni import improve


def test_extraction_csv_matches_rows_and_is_owner_scoped(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "API_KEYS", {"csv-owner-secret", "other"})
    monkeypatch.setattr(server, "SESSIONS", {})
    monkeypatch.setattr(server, "UPLOADS", str(tmp_path))
    monkeypatch.setattr(improve, "POL", str(tmp_path / "policy"))
    monkeypatch.setattr(improve, "_OWNER_STATES", {})
    auth = {"Authorization": "Bearer csv-owner-secret"}
    with TestClient(server.app) as client:
        sid = client.post("/v1/sessions", headers=auth, json={}).json()["session"]
        client.post(f"/v1/sessions/{sid}/documents", headers=auth, files={"file": ("invoice.txt", "Owner: মায়া\nTotal: 1200\nFormula: =1+2\n".encode())})
        result = client.post(f"/v1/sessions/{sid}/ask", headers=auth, json={"question": "Extract fields", "mode": "extract"}).json()
        path = f"/v1/sessions/{sid}/extractions/{result['id']}.csv"
        response = client.get(path, headers=auth)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert "attachment" in response.headers["content-disposition"]
        rows = list(csv.DictReader(io.StringIO(response.text.lstrip("\ufeff"))))
        assert next(row for row in rows if row["field"] == "Owner")["value"] == "মায়া"
        total = next(row for row in rows if row["field"] == "Total")
        assert total == {"field": "Total", "value": "1200", "document": "invoice.txt", "page": "1", "verified": "true"}
        assert next(row for row in rows if row["field"] == "Formula")["value"] == "'=1+2"
        assert "csv-owner-secret" not in response.text
        assert client.get(path, headers={"Authorization": "Bearer other"}).status_code == 404
        assert client.get(f"/v1/sessions/{sid}/extractions/missing.csv", headers=auth).status_code == 404
        document = next(iter(server.SESSIONS[sid]["ws"].docs.values()))
        document.pages[0].source = "ocr"
        ocr_result = client.post(f"/v1/sessions/{sid}/ask", headers=auth, json={"question": "Extract cached OCR fields", "mode": "extract"}).json()
        ocr_csv = client.get(f"/v1/sessions/{sid}/extractions/{ocr_result['id']}.csv", headers=auth)
        assert all(row["verified"] == "false" for row in csv.DictReader(io.StringIO(ocr_csv.text.lstrip("\ufeff"))))
        server.SESSIONS[sid]["history"] = []
        assert client.get(path, headers=auth).status_code == 404


def test_csv_preserves_quoting_and_guards_formula_cells():
    output = io.StringIO(newline="")
    csv.writer(output).writerow([server.csv_cell(v) for v in [None, "A, B", 'A "quote"', "  @SUM(1,2)", "\tunsafe", "-20", "বাংলা"]])
    assert next(csv.reader(io.StringIO(output.getvalue()))) == ["", "A, B", 'A "quote"', "'  @SUM(1,2)", "'\tunsafe", "'-20", "বাংলা"]


def test_targeted_extraction_keeps_ocr_matches_unverified():
    from types import SimpleNamespace
    from omni.agent import Agent, Workspace
    from omni.budget import Budget
    from omni.ingest import Doc, Page

    for source in ("ocr", "text"):
        ws = Workspace()
        page = Page(1, [("Total: 1200", (0, 0, 1, 1))], source)
        document = Doc("invoice", "invoice.pdf", "unused", "pdf", [page], "invoice")
        ws.docs[document.id] = document
        ws._index_page(document, page)
        agent = Agent.__new__(Agent)
        agent.ws = ws
        agent.llm = SimpleNamespace(chat=lambda *args, **kwargs: '{"Total":"1200"}')
        result = agent.extract("Extract", Budget.make(), ["invoice"], "test", targets=["Total"])
        assert result["fields"][0]["value"] == "1200"
        assert result["fields"][0]["verified"] is (source == "text")
