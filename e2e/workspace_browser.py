"""User flows through a real browser and API; no paid model calls.

Run: python e2e/workspace_browser.py
Requires: pip install playwright uvicorn; playwright install chromium
"""
import json
import csv
import io
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import pymupdf
from PIL import Image
from playwright.sync_api import sync_playwright, expect

root = Path(__file__).resolve().parents[1]
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
base = f"http://127.0.0.1:{port}"
process = subprocess.Popen([sys.executable, str(root / "e2e/workspace_server.py"), str(port)], cwd=root)
try:
    for attempt in range(100):
        if process.poll() is not None:
            raise RuntimeError("Browser test server exited before startup")
        try:
            if httpx.get(base + "/", timeout=1).status_code == 200:
                break
        except httpx.TransportError:
            pass
        time.sleep(0.1)
    else:
        raise RuntimeError("Browser test server did not start")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        def assert_total():
            table = page.get_by_role("table", name="Extracted fields", exact=True).last
            row = table.get_by_role("row").filter(has=page.get_by_role("cell", name="Total", exact=True))
            expect(row).to_contain_text("1200")
        # Check the actual generated Blob payload without depending on OS download saving.
        page.add_init_script("""window.exportPayload = null;
          const original = URL.createObjectURL;
          URL.createObjectURL = function(blob) {
            blob.text().then(text => window.exportPayload = text);
            return original.call(URL, blob);
          };""")
        page.goto(base)
        page.get_by_role("button", name="Connect", exact=True).click()
        page.get_by_label("API key", exact=True).fill("invalid")
        page.locator("#auth-form").get_by_role("button", name="Connect", exact=True).click()
        expect(page.locator("#auth-error")).to_have_text("invalid API key")
        page.get_by_label("API key", exact=True).fill("browser-owner")
        page.locator("#auth-form").get_by_role("button", name="Connect", exact=True).click()
        expect(page.locator("#auth-dialog")).not_to_be_visible()
        first = page.locator("#sessions").input_value()

        def rename(name):
            page.get_by_role("button", name="Rename workspace").click()
            page.get_by_label("Workspace name", exact=True).fill(name)
            page.get_by_role("button", name="Save name", exact=True).click()
            expect(page.locator("#name-dialog")).not_to_be_visible()
            expect(page.locator("#sessions option:checked")).to_contain_text(name)

        rename("Invoices <script>literal</script>")
        page.locator("#files").set_input_files({"name": "invoice.txt", "mimeType": "text/plain", "buffer": b"Owner: Maya\nTotal: 1200\nDate: 2026-10-05\n"})
        expect(page.locator("#documents")).to_contain_text("invoice.txt")
        page.locator("#extract").check()
        page.get_by_label("Your question", exact=True).fill("Extract invoice fields")
        page.get_by_role("button", name="Send", exact=False).click()
        assert_total()
        expect(page.get_by_role("table", name="Extracted fields").get_by_role("row")).to_have_count(4)
        fields = page.get_by_role("table", name="Extracted fields")
        page.get_by_label("Search extracted fields", exact=True).fill("1200")
        expect(fields.get_by_role("row")).to_have_count(2)
        expect(fields).to_contain_text("Total")
        page.get_by_label("Field evidence status", exact=True).select_option("review")
        expect(fields.get_by_role("row")).to_have_count(1)
        expect(page.get_by_text("No fields match these filters.", exact=True)).to_be_visible()
        page.get_by_label("Field evidence status", exact=True).select_option("matched")
        expect(fields.get_by_role("row")).to_have_count(2)
        with page.expect_download() as download:
            page.get_by_role("button", name="Export all fields CSV", exact=True).click()
        assert download.value.suggested_filename == "omni-extracted-fields.csv"
        assert download.value.failure() is None
        page.wait_for_function("window.exportPayload !== null")
        csv_payload = page.evaluate("window.exportPayload")
        rows = list(csv.DictReader(io.StringIO(csv_payload.lstrip("\ufeff"))))
        assert len(rows) == 3  # UI filters do not silently narrow the full export.
        assert next(row for row in rows if row["field"] == "Total") == {"field": "Total", "value": "1200", "document": "invoice.txt", "page": "1", "verified": "true"}
        assert "browser-owner" not in csv_payload
        page.get_by_role("button", name="Clear field filters", exact=True).click()
        expect(fields.get_by_role("row")).to_have_count(4)
        page.get_by_label("Search extracted fields", exact=True).fill("INVOICE.TXT")
        expect(fields.get_by_role("row")).to_have_count(4)
        page.get_by_label("Search extracted fields", exact=True).fill("<script>")
        expect(fields.get_by_role("row")).to_have_count(1)
        page.get_by_role("button", name="Clear field filters", exact=True).click()
        extraction_id = page.request.get(base + f"/v1/sessions/{first}/history", headers={"Authorization": "Bearer browser-owner"}).json()["messages"][-1]["result"]["id"]
        page.get_by_role("table", name="Extracted fields").get_by_role("button", name="invoice.txt · Page 1", exact=True).first.click()
        expect(page.locator("#page-text")).to_contain_text("Total: 1200")
        page.get_by_role("button", name="Close", exact=True).click()
        expect(page.locator("#evidence")).to_contain_text("Owner: Maya")
        page.locator("#evidence").get_by_role("button", name="Read page text").first.click()
        expect(page.locator("#page-text")).to_contain_text("Total: 1200")
        page.get_by_role("button", name="Close", exact=True).click()
        pdf = pymupdf.open()
        pdf.new_page().insert_text((72, 72), "First page of the synthetic report. A total of 1200 is recorded for Maya.")
        pdf.new_page().insert_text((72, 72), "Second page of the synthetic report. Reference: ALPHA-42. <script>literal</script>")
        page.locator("#files").set_input_files({"name": "report.pdf", "mimeType": "application/pdf", "buffer": pdf.tobytes()})
        pdf.close()
        expect(page.get_by_role("button", name="Read report.pdf", exact=True)).to_be_visible()
        page.get_by_role("button", name="Read report.pdf", exact=True).click()
        expect(page.locator("#page-position")).to_have_text("Page 1 of 2")
        expect(page.get_by_role("button", name="Previous page")).to_be_disabled()
        page.get_by_role("button", name="Next page").click()
        expect(page.locator("#page-position")).to_have_text("Page 2 of 2")
        expect(page.get_by_role("button", name="Next page")).to_be_disabled()
        expect(page.locator("#page-text")).to_contain_text("<script>literal</script>")
        assert page.locator("#page-text script").count() == 0
        page.get_by_label("Search document text").fill("maya")
        page.get_by_role("button", name="Search document", exact=True).click()
        expect(page.locator("#page-matches")).to_contain_text("1 matching page")
        page.locator("#page-matches button").click()
        expect(page.locator("#page-position")).to_have_text("Page 1 of 2")
        expect(page.locator("#page-text mark")).to_have_text("Maya")
        page.get_by_label("Search document text").fill("no-such-text")
        page.get_by_role("button", name="Search document", exact=True).click()
        expect(page.locator("#page-matches")).to_contain_text("0 matching pages")
        page.get_by_role("button", name="Close", exact=True).click()
        page.get_by_label("Read new scans").select_option("lazy")
        scan = io.BytesIO()
        Image.new("RGB", (20, 20), "white").save(scan, format="PNG")
        page.locator("#files").set_input_files({"name": "unread.png", "mimeType": "image/png", "buffer": scan.getvalue()})
        expect(page.get_by_role("button", name="Read unread.png", exact=True)).to_be_visible()
        page.get_by_role("button", name="Read unread.png", exact=True).click()
        expect(page.locator("#page-source")).to_contain_text("still needs OCR")
        expect(page.locator("#page-text")).to_contain_text("No readable text")
        page.get_by_label("Search document text").fill("anything")
        page.get_by_role("button", name="Search document", exact=True).click()
        expect(page.locator("#page-matches")).to_contain_text("1 unread page(s)")
        page.get_by_role("button", name="Close", exact=True).click()
        page.get_by_role("button", name="Edit workspace limits", exact=True).click()
        page.get_by_label("Model tokens limit", exact=True).fill("5000")
        page.get_by_label("OCR pages limit", exact=True).fill("0")
        page.get_by_label("Time (seconds) limit", exact=True).fill("600")
        page.get_by_label("Tool calls limit", exact=True).fill("-1")
        page.get_by_role("button", name="Save limits", exact=True).click()
        expect(page.locator("#limits-dialog")).to_be_visible()
        unchanged = page.request.get(base + f"/v1/sessions/{first}/budget", headers={"Authorization": "Bearer browser-owner"}).json()
        assert unchanged["budget"]["llm_tokens"]["limit"] == 4000
        page.get_by_label("Tool calls limit", exact=True).fill("8")
        page.get_by_role("button", name="Save limits", exact=True).click()
        expect(page.locator("#limits-dialog")).not_to_be_visible()
        updated = page.request.get(base + f"/v1/sessions/{first}/budget", headers={"Authorization": "Bearer browser-owner"}).json()
        assert updated["budget"]["llm_tokens"]["limit"] == 5000
        assert updated["budget"]["ocr_pages"]["limit"] == 0
        for decision in ("Decline", "Approve"):
            before_count = page.locator("#messages .assistant").count()
            page.get_by_label("Your question", exact=True).fill("Extract the remaining fields")
            page.get_by_role("button", name="Send", exact=False).click()
            expect(page.locator("#messages .assistant")).to_have_count(before_count + 1)
            expect(page.locator("#messages .assistant").last).to_contain_text("Some pages remain unread")
            page.locator("#extensions").get_by_role("button", name=decision, exact=True).click()
            expect(page.locator("#extensions button")).to_have_count(0)
        assert page.request.get(base + f"/v1/sessions/{first}/budget", headers={"Authorization": "Bearer browser-owner"}).json()["budget"]["ocr_pages"]["limit"] == 1

        page.get_by_role("button", name="New workspace").click()
        expect(page.locator("#sessions")).not_to_have_value(first)
        second = page.locator("#sessions").input_value()
        rename("Empty workspace")
        expect(page.locator("#messages .message")).to_have_count(0)
        page.locator("#sessions").select_option(first)
        assert_total()
        expect(page.locator("#evidence")).to_contain_text("Owner: Maya")
        page.reload()
        expect(page.locator("#sessions")).to_have_value(first)
        assert_total()

        page.get_by_role("button", name="Vision chat", exact=True).click()
        expect(page.locator("#messages .message")).to_have_count(0)
        page.get_by_label("Your question", exact=True).fill("Hello from vision mode")
        page.get_by_role("button", name="Send", exact=False).click()
        expect(page.locator("#messages .assistant").last).to_have_text("OmniSynthetic chat reply")
        page.get_by_role("button", name="Document Q&A", exact=True).click()
        assert_total()
        expect(page.locator("#messages")).not_to_contain_text("Synthetic chat reply")
        page.get_by_role("button", name="Vision chat", exact=True).click()
        expect(page.locator("#messages .assistant").last).to_contain_text("Synthetic chat reply")
        page.reload()
        # Document mode is the default after reload; both histories are retained.
        assert_total()
        page.get_by_role("button", name="Vision chat", exact=True).click()
        expect(page.locator("#messages .assistant").last).to_contain_text("Synthetic chat reply")
        page.get_by_role("button", name="Export chat", exact=True).click()
        page.wait_for_function("window.exportPayload !== null")
        payload = page.evaluate("window.exportPayload")
        assert "browser-owner" not in payload
        assert json.loads(payload)["messages"][0]["content"] == "Hello from vision mode"

        other = browser.new_context()
        other_page = other.new_page()
        other_page.goto(base)
        other_page.get_by_role("button", name="Connect", exact=True).click()
        other_page.get_by_label("API key", exact=True).fill("browser-other")
        other_page.locator("#auth-form").get_by_role("button", name="Connect", exact=True).click()
        expect(other_page.locator("#auth-dialog")).not_to_be_visible()
        expect(other_page.locator("#sessions option")).to_have_count(1)
        expect(other_page.locator("#messages .message")).to_have_count(0)
        denied = other_page.request.get(base + f"/v1/sessions/{first}/history", headers={"Authorization": "Bearer browser-other"})
        assert denied.status == 404
        documents = page.request.get(base + f"/v1/sessions/{first}/documents", headers={"Authorization": "Bearer browser-owner"}).json()
        denied_page = other_page.request.get(base + f"/v1/sessions/{first}/documents/{documents[0]['doc_id']}/pages/1", headers={"Authorization": "Bearer browser-other"})
        assert denied_page.status == 404
        denied_limits = other_page.request.post(base + f"/v1/sessions/{first}/budget/limits", headers={"Authorization": "Bearer browser-other"}, data={"limits": {"ocr_pages": 100}})
        assert denied_limits.status == 404
        denied_export = other_page.request.get(base + f"/v1/sessions/{first}/extractions/{extraction_id}.csv", headers={"Authorization": "Bearer browser-other"})
        assert denied_export.status == 404

        page.set_viewport_size({"width": 390, "height": 844})
        page.get_by_label("Toggle color theme").click()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.get_by_role("button", name="Read report.pdf", exact=True).click()
        expect(page.locator("#page-position")).to_have_text("Page 1 of 2")
        assert page.evaluate("document.querySelector('#page-dialog').scrollWidth <= innerWidth")
        if os.environ.get("OMNI_VIEWER_SCREENSHOT"):
            page.screenshot(path=os.environ["OMNI_VIEWER_SCREENSHOT"], full_page=True)
        page.get_by_role("button", name="Close", exact=True).click()
        page.get_by_role("button", name="Document Q&A", exact=True).click()
        expect(page.get_by_role("table", name="Extracted fields").last).to_be_visible()
        mobile_fields = page.get_by_role("table", name="Extracted fields").last
        mobile_search = page.get_by_label("Search extracted fields", exact=True).last
        mobile_search.fill("maya")
        expect(mobile_fields.get_by_role("row")).to_have_count(2)
        page.get_by_role("button", name="Clear field filters", exact=True).last.click()
        expect(mobile_fields.get_by_role("row")).to_have_count(4)
        mobile_search.scroll_into_view_if_needed()
        assert page.evaluate("document.querySelector('#messages').scrollWidth <= document.querySelector('#messages').clientWidth")
        if os.environ.get("OMNI_FIELDS_SCREENSHOT"):
            page.screenshot(path=os.environ["OMNI_FIELDS_SCREENSHOT"], full_page=True)
        page.get_by_role("button", name="Edit workspace limits", exact=True).click()
        expect(page.get_by_label("Model tokens limit", exact=True)).to_have_value("5000")
        expect(page.get_by_label("OCR pages limit", exact=True)).to_have_value("1")
        assert page.evaluate("document.querySelector('#limits-dialog').scrollWidth <= innerWidth")
        if os.environ.get("OMNI_LIMITS_SCREENSHOT"):
            page.screenshot(path=os.environ["OMNI_LIMITS_SCREENSHOT"], full_page=True)
        page.get_by_role("button", name="Cancel", exact=True).click()
        page.locator("#sessions").select_option(second)
        expect(page.locator("#messages .message")).to_have_count(0)
        assert not errors, errors
        if os.environ.get("OMNI_E2E_SCREENSHOT"):
            page.screenshot(path=os.environ["OMNI_E2E_SCREENSHOT"], full_page=True)
        browser.close()
    print("PASS: field search/status/no-match/clear/mobile filters/full CSV while filtered, field table/source links/retained rows, workspace limits/validation/persistence, resource denial/approval, connect, naming, upload, extraction, evidence/page links, page navigation, literal HTML, search/jump/highlight/no matches, switching, reload, mode separation, export payload, owner isolation, mobile viewer/fields/limits/theme; no browser exceptions")
finally:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
