# Vision Chat integration

The former `E:\slm-vision-chat` project now lives here. Its chat and PDF/text
retrieval endpoints run inside Omni Agent's existing `server:app` on port 8090.
There is no separate gateway process.

- `POST /v1/chat/completions`: text/image messages, optional `session_id`,
  optional `doc_ids`, optional `model`; non-streaming responses.
- `POST /v1/documents`: embedding-based PDF/text retrieval uploads.
- `GET /v1/chat/health`: backend health.

Use the same `OMNI_API_KEYS` authentication as the document-agent API.
Chat history and retrieval documents remain in memory and reset on restart.
Ownership is scoped to an API key. Scanned documents belong in Omni Agent's
`/v1/sessions/{id}/documents` OCR pipeline; these two upload APIs return different
document IDs and do not share IDs.

Embedding uploads require `sentence-transformers`; normal chat does not.
Install `pip install -r vision_lab/requirements.txt` for embedding retrieval and
the optional training tools. The embedding model must be downloaded once before
offline retrieval can work.

Run research scripts from this directory, for example:

```powershell
Set-Location vision_lab
python scripts/eval_api.py 8082
python scripts/train.py
```

Local models, datasets, logs, and the llama.cpp runtime were preserved here but
are excluded from Git. Training does not automatically replace the serving
model. Compare tuned and stock models before choosing a deployment.
