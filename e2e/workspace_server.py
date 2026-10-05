"""Isolated browser-test server. Uses synthetic files and a deterministic chat stub."""
import os
from pathlib import Path
import sys
import tempfile

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["OMNI_API_KEYS"] = "browser-owner,browser-other"

import server
from omni import improve
from vision_lab.app import gateway

port = int(sys.argv[1])
with tempfile.TemporaryDirectory(prefix="omni-browser-") as directory:
    server.UPLOADS = str(Path(directory) / "uploads")
    Path(server.UPLOADS).mkdir()
    improve.POL = str(Path(directory) / "policy")
    gateway.BACKEND = gateway.VISION_BACKEND = f"http://127.0.0.1:{port}/test-model"

    @server.app.post("/test-model/v1/chat/completions")
    def test_model(body: dict):
        return {"choices": [{"message": {"role": "assistant", "content": "Synthetic chat reply"}}]}

    uvicorn.run(server.app, host="127.0.0.1", port=port, log_level="error")
