from fastapi.testclient import TestClient
import server
from omni import improve


def test_workspace_lists_documents_only_for_owner(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "API_KEYS", {"owner", "other"})
    monkeypatch.setattr(server, "SESSIONS", {})
    monkeypatch.setattr(server, "UPLOADS", str(tmp_path))
    monkeypatch.setattr(improve, "POL", str(tmp_path / "policy"))
    monkeypatch.setattr(improve, "_OWNER_STATES", {})
    auth = {"Authorization": "Bearer owner"}
    other = {"Authorization": "Bearer other"}
    with TestClient(server.app) as client:
        sid = client.post("/v1/sessions", headers=auth, json={}).json()["session"]
        uploaded = client.post(f"/v1/sessions/{sid}/documents", headers=auth, files={"file": ("invoice.txt", b"Owner: Maya\nTotal: 1200\n")})
        assert uploaded.status_code == 200
        sessions = client.get("/v1/sessions", headers=auth).json()
        assert sessions == [{"session": sid, "created_at": server.SESSIONS[sid]["created_at"], "documents": 1}]
        assert client.get("/v1/sessions", headers=other).json() == []
        assert client.get(f"/v1/sessions/{sid}/documents", headers=other).status_code == 404
        documents = client.get(f"/v1/sessions/{sid}/documents", headers=auth).json()
        assert documents[0]["doc_id"] == uploaded.json()["doc_id"]
        assert documents[0]["name"] == "invoice.txt"
        assert "path" not in documents[0]
        extracted = client.post(f"/v1/sessions/{sid}/ask", headers=auth, json={"question": "Extract the fields", "mode": "extract"})
        assert extracted.status_code == 200
        assert any(f["value"] == "1200" for f in extracted.json()["fields"])


def test_budget_approval_is_not_replayable(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "SESSIONS", {})
    monkeypatch.setattr(improve, "POL", str(tmp_path))
    monkeypatch.setattr(improve, "_OWNER_STATES", {})
    auth = {"Authorization": "Bearer dev-key"}
    with TestClient(server.app) as client:
        sid = client.post("/v1/sessions", headers=auth, json={}).json()["session"]
        budget = server.SESSIONS[sid]["budget"]
        budget.request_extension("llm_tokens", 100, "Need more tokens")
        initial = budget.user_limits["llm_tokens"]
        path = f"/v1/sessions/{sid}/budget/resolve"
        assert client.post(path, headers=auth, json={"request_id": 0, "approve": True}).status_code == 200
        assert client.post(path, headers=auth, json={"request_id": 0, "approve": True}).status_code == 409
        assert client.post(path, headers=auth, json={"request_id": -1, "approve": True}).status_code == 422
        assert budget.user_limits["llm_tokens"] == initial + 100
