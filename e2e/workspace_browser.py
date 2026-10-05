"""User flows through a real browser and API; no paid model calls.

Run: python e2e/workspace_browser.py
Requires: pip install playwright uvicorn; playwright install chromium
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
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
        expect(page.locator("#messages .assistant")).to_contain_text("Total: 1200")
        expect(page.locator("#evidence")).to_contain_text("Owner: Maya")

        page.get_by_role("button", name="New workspace").click()
        expect(page.locator("#sessions")).not_to_have_value(first)
        second = page.locator("#sessions").input_value()
        rename("Empty workspace")
        expect(page.locator("#messages .message")).to_have_count(0)
        page.locator("#sessions").select_option(first)
        expect(page.locator("#messages .assistant")).to_contain_text("Total: 1200")
        expect(page.locator("#evidence")).to_contain_text("Owner: Maya")
        page.reload()
        expect(page.locator("#sessions")).to_have_value(first)
        expect(page.locator("#messages .assistant")).to_contain_text("Total: 1200")

        page.get_by_role("button", name="Vision chat", exact=True).click()
        expect(page.locator("#messages .message")).to_have_count(0)
        page.get_by_label("Your question", exact=True).fill("Hello from vision mode")
        page.get_by_role("button", name="Send", exact=False).click()
        expect(page.locator("#messages .assistant")).to_have_text("OmniSynthetic chat reply")
        page.get_by_role("button", name="Document Q&A", exact=True).click()
        expect(page.locator("#messages .assistant")).to_contain_text("Total: 1200")
        expect(page.locator("#messages")).not_to_contain_text("Synthetic chat reply")
        page.get_by_role("button", name="Vision chat", exact=True).click()
        expect(page.locator("#messages .assistant")).to_contain_text("Synthetic chat reply")
        page.reload()
        # Document mode is the default after reload; both histories are retained.
        expect(page.locator("#messages .assistant")).to_contain_text("Total: 1200")
        page.get_by_role("button", name="Vision chat", exact=True).click()
        expect(page.locator("#messages .assistant")).to_contain_text("Synthetic chat reply")
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

        page.set_viewport_size({"width": 390, "height": 844})
        page.get_by_label("Toggle color theme").click()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.locator("#sessions").select_option(second)
        expect(page.locator("#messages .message")).to_have_count(0)
        assert not errors, errors
        if os.environ.get("OMNI_E2E_SCREENSHOT"):
            page.screenshot(path=os.environ["OMNI_E2E_SCREENSHOT"], full_page=True)
        browser.close()
    print("PASS: connect, naming, upload, extraction, evidence, switching, reload, mode separation, export payload, owner isolation, mobile theme; no browser exceptions")
finally:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
