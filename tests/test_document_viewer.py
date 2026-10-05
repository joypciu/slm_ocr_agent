import io

from fastapi.testclient import TestClient
from PIL import Image

import server
from omni import improve
from omni.ingest import Doc, Page


def test_document_pages_search_and_owner_isolation(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "API_KEYS", {"owner", "other"})
    monkeypatch.setattr(server, "SESSIONS", {})
    monkeypatch.setattr(server, "UPLOADS", str(tmp_path))
    monkeypatch.setattr(improve, "POL", str(tmp_path / "policy"))
    monkeypatch.setattr(improve, "_OWNER_STATES", {})
    auth = {"Authorization": "Bearer owner"}
    other = {"Authorization": "Bearer other"}
    with TestClient(server.app) as client:
        sid = client.post("/v1/sessions", headers=auth, json={}).json()["session"]
        uploaded = client.post(f"/v1/sessions/{sid}/documents?read=lazy", headers=auth,
                               files={"file": ("invoice.txt", b"Owner: Maya\nTotal: 1200\n<script>literal</script>\n")}).json()
        doc_id = uploaded["doc_id"]
        prefix = f"/v1/sessions/{sid}/documents/{doc_id}"
        page = client.get(prefix + "/pages/1", headers=auth)
        assert page.status_code == 200
        assert page.json()["name"] == "invoice.txt"
        assert page.json()["source"] == "text"
        assert page.json()["needs_ocr"] is False
        assert "<script>literal</script>" in page.json()["text"]
        assert "path" not in page.json()
        assert client.get(prefix + "/pages/0", headers=auth).status_code == 404
        assert client.get(prefix + "/pages/2", headers=auth).status_code == 404
        assert client.get(prefix + "/pages/1", headers=other).status_code == 404
        assert client.get(prefix + "/search?q=MAYA", headers=other).status_code == 404
        result = client.get(prefix + "/search", headers=auth, params={"q": "MAYA"}).json()
        assert result["total_matching_pages"] == 1
        assert "Maya" in result["matches"][0]["excerpt"]
        assert client.get(prefix + "/search?q=missing", headers=auth).json()["matches"] == []
        assert client.get(prefix + "/search?q=x", headers=auth).status_code == 422
        assert client.get(prefix + "/search?q=%20%20", headers=auth).status_code == 422
        assert client.get(f"/v1/sessions/{sid}/documents/missing/pages/1", headers=auth).status_code == 404
        extracted = client.post(f"/v1/sessions/{sid}/ask", headers=auth, json={"question": "Extract", "mode": "extract"}).json()
        assert all(f["doc_id"] == doc_id for f in extracted["fields"])


def test_unread_page_preview_does_not_start_ocr_and_search_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "SESSIONS", {})
    monkeypatch.setattr(server, "UPLOADS", str(tmp_path))
    monkeypatch.setattr(improve, "POL", str(tmp_path / "policy"))
    monkeypatch.setattr(improve, "_OWNER_STATES", {})
    auth = {"Authorization": "Bearer dev-key"}
    image = io.BytesIO()
    Image.new("RGB", (10, 10), "white").save(image, format="PNG")
    with TestClient(server.app) as client:
        sid = client.post("/v1/sessions", headers=auth, json={}).json()["session"]
        doc_id = client.post(f"/v1/sessions/{sid}/documents?read=lazy", headers=auth,
                             files={"file": ("scan.png", image.getvalue())}).json()["doc_id"]
        monkeypatch.setattr(server.SESSIONS[sid]["ws"], "ocr", lambda *args: (_ for _ in ()).throw(AssertionError("Viewer must not OCR")))
        prefix = f"/v1/sessions/{sid}/documents/{doc_id}"
        assert client.get(prefix + "/pages/1", headers=auth).json()["needs_ocr"] is True
        assert client.get(prefix + "/search?q=hello", headers=auth).json() == {"matches": [], "total_matching_pages": 0, "unread_pages": 1}
        ws = server.SESSIONS[sid]["ws"]
        ws.docs["large"] = Doc("large", "long.txt", "unused", "text", [Page(n, [("Target text", (0, 0, 1, 1))], "text") for n in range(1, 56)], "large")
        result = client.get(f"/v1/sessions/{sid}/documents/large/search?q=target", headers=auth).json()
        assert result["total_matching_pages"] == 55
        assert len(result["matches"]) == 50
