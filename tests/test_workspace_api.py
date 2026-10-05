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
        assert sessions == [{"session": sid, "name": "Untitled workspace", "created_at": server.SESSIONS[sid]["created_at"], "documents": 1}]
        assert client.get("/v1/sessions", headers=other).json() == []
        assert client.get(f"/v1/sessions/{sid}/documents", headers=other).status_code == 404
        documents = client.get(f"/v1/sessions/{sid}/documents", headers=auth).json()
        assert documents[0]["doc_id"] == uploaded.json()["doc_id"]
        assert documents[0]["name"] == "invoice.txt"
        assert "path" not in documents[0]
        extracted = client.post(f"/v1/sessions/{sid}/ask", headers=auth, json={"question": "Extract the fields", "mode": "extract"})
        assert extracted.status_code == 200
        assert any(f["value"] == "1200" for f in extracted.json()["fields"])
        history_path = f"/v1/sessions/{sid}/history"
        history = client.get(history_path, headers=auth).json()["messages"]
        assert history[0] == {"role": "user", "content": "Extract the fields"}
        assert "Total: 1200" in history[1]["content"]
        assert history[1]["result"] == extracted.json()
        assert client.get(history_path, headers=other).status_code == 404
        assert client.get(history_path + "?mode=invalid", headers=auth).status_code == 422
        rename_path = f"/v1/sessions/{sid}"
        assert client.patch(rename_path, headers=other, json={"name": "Stolen"}).status_code == 404
        assert client.patch(rename_path, headers=auth, json={"name": "  "}).status_code == 422
        assert client.patch(rename_path, headers=auth, json={"name": "x" * 81}).status_code == 422
        assert client.patch(rename_path, headers=auth, json={"name": "  Invoices  "}).json()["name"] == "Invoices"
        assert client.get("/v1/sessions", headers=auth).json()[0]["name"] == "Invoices"


def test_chat_history_does_not_return_image_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "SESSIONS", {})
    monkeypatch.setattr(server, "API_KEYS", {"owner", "other"})
    monkeypatch.setattr(improve, "POL", str(tmp_path))
    monkeypatch.setattr(improve, "_OWNER_STATES", {})
    monkeypatch.setattr(server.vision_gateway, "SESSIONS", {})
    auth = {"Authorization": "Bearer owner"}
    with TestClient(server.app) as client:
        sid = client.post("/v1/sessions", headers=auth, json={}).json()["session"]
        server.vision_gateway.SESSIONS[("owner", sid)] = [
            {"role": "user", "content": [{"type": "text", "text": "Look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,PRIVATE"}}]},
            {"role": "assistant", "content": "A picture"},
        ]
        response = client.get(f"/v1/sessions/{sid}/history?mode=chat", headers=auth)
        assert response.json()["messages"] == [
            {"role": "user", "content": "Look", "attachment": True},
            {"role": "assistant", "content": "A picture", "attachment": False},
        ]
        assert "PRIVATE" not in response.text
        assert client.get(f"/v1/sessions/{sid}/history?mode=chat", headers={"Authorization": "Bearer other"}).status_code == 404


def test_document_history_keeps_complete_pairs_and_excludes_failed_requests(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "SESSIONS", {})
    monkeypatch.setattr(improve, "POL", str(tmp_path))
    monkeypatch.setattr(improve, "_OWNER_STATES", {})
    auth = {"Authorization": "Bearer dev-key"}
    with TestClient(server.app) as client:
        sid = client.post("/v1/sessions", headers=auth, json={}).json()["session"]
        monkeypatch.setattr(server.SESSIONS[sid]["agent"], "ask", lambda *args: {"answer": "Synthetic answer", "evidence": []})
        for n in range(51):
            assert client.post(f"/v1/sessions/{sid}/ask", headers=auth, json={"question": f"Question {n}"}).status_code == 200
        response = client.get(f"/v1/sessions/{sid}/history", headers=auth).json()
        assert response["limit"] == 100
        assert len(response["messages"]) == 100
        assert response["messages"][0] == {"role": "user", "content": "Question 1"}
        assert response["messages"][-2]["content"] == "Question 50"
        failed = client.post(f"/v1/sessions/{sid}/ask", headers=auth, json={"question": "Missing document", "doc_ids": ["missing"]})
        assert failed.status_code == 404
        assert client.get(f"/v1/sessions/{sid}/history", headers=auth).json() == response


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
