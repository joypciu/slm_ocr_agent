from fastapi.testclient import TestClient

import server
from omni import improve


def test_limit_updates_are_atomic_owned_and_preserve_spending(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "API_KEYS", {"owner", "other"})
    monkeypatch.setattr(server, "SESSIONS", {})
    monkeypatch.setattr(improve, "POL", str(tmp_path))
    monkeypatch.setattr(improve, "_OWNER_STATES", {})
    auth = {"Authorization": "Bearer owner"}
    with TestClient(server.app) as client:
        sid = client.post("/v1/sessions", headers=auth, json={}).json()["session"]
        b = server.SESSIONS[sid]["budget"]
        b.spend("llm_tokens", 100, "synthetic spend")
        b.request_extension("ocr_pages", 1, "synthetic pending request")
        initial = dict(b.user_limits)
        path = f"/v1/sessions/{sid}/budget/limits"
        assert client.post(path, headers={"Authorization": "Bearer other"}, json={"limits": {"llm_tokens": 0}}).status_code == 404
        for values in ({}, {"llm_tokens": 50, "ocr_pages": -1}, {"llm_tokens": 50, "ocr_pages": 1.5}, {"llm_tokens": 50, "unknown": 1}):
            assert client.post(path, headers=auth, json={"limits": values}).status_code == 422
            assert b.user_limits == initial
        response = client.post(path, headers=auth, json={"limits": {"llm_tokens": 50, "ocr_pages": 0, "seconds": 1.5}})
        assert response.status_code == 200
        assert response.json()["budget"]["llm_tokens"]["used"] == 100
        assert b.left("llm_tokens") == 0
        assert b.user_limits["seconds"] == 1.5
        assert len(response.json()["pending_requests"]) == 1
        assert b.log[-1]["op"] == "user_limits"
        assert b.log[-1]["before"] == initial
        response = client.post(path, headers={**auth, "Content-Type": "application/json"}, content='{"limits":{"seconds":1e400}}')
        assert response.status_code == 422
        assert b.user_limits["seconds"] == 1.5


def test_negative_budget_and_invalid_auto_grants_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "SESSIONS", {})
    monkeypatch.setattr(improve, "POL", str(tmp_path))
    monkeypatch.setattr(improve, "_OWNER_STATES", {})
    auth = {"Authorization": "Bearer dev-key"}
    with TestClient(server.app) as client:
        for body in ({"llm_tokens": -1}, {"seconds": -1}, {"auto_grant": {"ocr_pages": -1}}, {"auto_grant": {"unknown": 1}}):
            assert client.post("/v1/sessions", headers=auth, json=body).status_code == 422
        assert server.SESSIONS == {}
        sid = client.post("/v1/sessions", headers=auth, json={}).json()["session"]
        assert client.post(f"/v1/sessions/{sid}/budget/limit", headers=auth, json={"resource": "ocr_pages", "cap": 0.5}).status_code == 422
        assert client.post(f"/v1/sessions/{sid}/budget/limit", headers=auth, json={"resource": "seconds", "cap": 0.5}).status_code == 200
