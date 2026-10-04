import hashlib

import httpx
from fastapi.testclient import TestClient

import server
from omni import improve
from vision_lab.app import gateway


def headers(owner):
    return {"Authorization": f"Bearer {owner}"}


def test_session_and_policy_isolation(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "API_KEYS", {"alice", "bob"})
    monkeypatch.setattr(improve, "POL", str(tmp_path))
    monkeypatch.setattr(improve, "_OWNER_STATES", {})
    with TestClient(server.app) as client:
        sid = client.post("/v1/sessions", headers=headers("alice"), json={}).json()["session"]
        assert client.get(f"/v1/sessions/{sid}/budget", headers=headers("alice")).status_code == 200
        assert client.get(f"/v1/sessions/{sid}/budget", headers=headers("bob")).status_code == 404
        assert client.post(f"/v1/sessions/{sid}/ask", headers=headers("bob"), json={"question": "hello"}).status_code == 404
    token = improve.POLICY_OWNER.set(hashlib.sha256(b"alice").hexdigest())
    try:
        improve.owner_state("alice")[2].remember("total", "123", ["same-doc"])
    finally:
        improve.POLICY_OWNER.reset(token)
    assert improve.owner_state("bob")[2].recall("total", ["same-doc"]) is None
    assert improve.owner_state("alice")[2].recall("total", ["same-doc"])["a"] == "123"


def test_chat_document_and_history_isolation(monkeypatch):
    monkeypatch.setattr(gateway, "API_KEYS", {"alice", "bob"})
    monkeypatch.setattr(gateway, "DOCS", {"private-doc": {"owner": "alice"}})
    monkeypatch.setattr(gateway, "SESSIONS", {})
    payloads = []

    def backend(request):
        import json
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    original = httpx.AsyncClient
    monkeypatch.setattr(gateway.httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(backend), **kwargs))
    with TestClient(server.app) as client:
        response = client.post("/v1/chat/completions", headers=headers("bob"), json={"messages": [{"role": "user", "content": "total"}], "doc_ids": ["private-doc"]})
        assert response.status_code == 404
        for owner, text in [("alice", "private question"), ("bob", "hello")]:
            response = client.post("/v1/chat/completions", headers=headers(owner), json={"messages": [{"role": "user", "content": text}], "session_id": "same-session"})
            assert response.status_code == 200
    assert len(payloads[1]["messages"]) == 1
    assert payloads[1]["messages"][0]["content"] == "hello"


def test_chat_rejects_empty_messages():
    with TestClient(server.app) as client:
        assert client.post("/v1/chat/completions", headers=headers("dev-key"), json={"messages": []}).status_code == 422
