"""API gateway: OpenAI-style chat + PDF/text RAG in front of llama-server (the SLM)."""
import asyncio, base64, io, os, re, uuid
import fitz, httpx, numpy as np
from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

BACKEND = os.environ.get("SLM_BACKEND", os.environ.get("OMNI_LLM_URL", "http://127.0.0.1:8081"))
VISION_BACKEND = os.environ.get("SLM_BACKEND", os.environ.get("OMNI_VISION_URL", BACKEND))
API_KEYS = set(filter(None, os.environ.get("OMNI_API_KEYS", os.environ.get("SLM_API_KEYS", "dev-key")).split(",")))
IMG_LIMIT = asyncio.Semaphore(int(os.environ.get("SLM_IMG_CONCURRENCY", "2")))  # image encoding is the CPU bottleneck
TIMEOUT, RETRIES, TOP_K = 90, 2, 3

router = APIRouter(tags=["vision-chat"])
_embed = None
DOCS: dict[str, dict] = {}       # doc_id -> {name, owner, chunks, vecs}
SESSIONS: dict[tuple[str, str], list] = {}   # (owner, session_id) -> history

def embedder():
    global _embed
    if _embed is None:
        from sentence_transformers import SentenceTransformer
        _embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _embed

def auth(authorization: str = Header(default="")):
    key = authorization.removeprefix("Bearer ").strip()
    if key not in API_KEYS: raise HTTPException(401, "invalid API key")
    return key

def chunk(text, size=700, overlap=120):
    text = re.sub(r"\s+", " ", text).strip()
    return [text[i:i + size] for i in range(0, max(len(text), 1), size - overlap)] if text else []

def extract(name, data: bytes):
    if name.lower().endswith(".pdf"):
        with fitz.open(stream=data, filetype="pdf") as pdf:
            return [(f"{name} p.{i+1}", p.get_text()) for i, p in enumerate(pdf)]
    return [(name, data.decode("utf-8", "ignore"))]

@router.post("/v1/documents")
async def upload(file: UploadFile = File(...), key: str = Depends(auth)):
    parts = extract(file.filename, await file.read())
    chunks = [(src, c) for src, t in parts for c in chunk(t)]
    if not chunks: raise HTTPException(422, "no extractable text (scanned PDF? send pages as images instead)")
    vecs = await asyncio.to_thread(embedder().encode, [c for _, c in chunks], normalize_embeddings=True)
    doc_id = uuid.uuid4().hex[:12]; DOCS[doc_id] = {"name": file.filename, "owner": key, "chunks": chunks, "vecs": vecs}
    return {"doc_id": doc_id, "chunks": len(chunks)}

def retrieve(doc_ids, query, owner):
    scored = []
    q = embedder().encode([query], normalize_embeddings=True)[0]
    for d in doc_ids:
        doc = DOCS.get(d)
        if doc and doc["owner"] == owner: scored += [(float(v @ q), src, c) for v, (src, c) in zip(doc["vecs"], doc["chunks"])]
    return sorted(scored, reverse=True)[:TOP_K]

class ChatReq(BaseModel):
    messages: list[dict] = Field(min_length=1)
    model: str | None = None
    doc_ids: list[str] = []
    session_id: str | None = None
    max_tokens: int = Field(default=256, ge=1, le=32768)
    temperature: float = 0.2

def has_image(msgs):
    return any(isinstance(m.get("content"), list) and any(p.get("type") == "image_url" for p in m["content"]) for m in msgs)

@router.post("/v1/chat/completions")
async def chat(req: ChatReq, key: str = Depends(auth)):
    msgs = list(req.messages)
    session_key = (key, req.session_id)
    if req.session_id: msgs = SESSIONS.get(session_key, [])[-8:] + msgs
    if req.doc_ids:
        if any(d not in DOCS or DOCS[d]["owner"] != key for d in req.doc_ids):
            raise HTTPException(404, "unknown document id")
        last = next((m for m in reversed(msgs) if m["role"] == "user"), None)
        if last is None:
            raise HTTPException(422, "document retrieval requires a user message")
        q = last["content"] if isinstance(last["content"], str) else " ".join(p.get("text", "") for p in last["content"])
        hits = await asyncio.to_thread(retrieve, req.doc_ids, q, key)
        ctx = "\n\n".join(f"[{src}] {c}" for _, src, c in hits)
        msgs = [{"role": "system", "content": "Answer ONLY from the context below. If the answer is not there, say you cannot find it in the documents. Cite the [source].\n\nContext:\n" + ctx}] + msgs
    body = {"messages": msgs, "max_tokens": req.max_tokens, "temperature": req.temperature}
    image_request = has_image(msgs)
    model = req.model or os.environ.get("OMNI_UPSTREAM_VISION_MODEL" if image_request else "OMNI_UPSTREAM_MODEL")
    if model:
        body["model"] = model
    upstream_key = os.environ.get("OMNI_UPSTREAM_API_KEY", "")
    headers = {"Authorization": f"Bearer {upstream_key}"} if upstream_key else {}
    gate = IMG_LIMIT if image_request else None
    backend = VISION_BACKEND if gate else BACKEND
    async with httpx.AsyncClient(timeout=TIMEOUT) as cli:
        for attempt in range(RETRIES + 1):
            try:
                if gate: await gate.acquire()
                try: r = await cli.post(f"{backend}/v1/chat/completions", json=body, headers=headers)
                finally:
                    if gate: gate.release()
                r.raise_for_status(); out = r.json(); break
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
                if attempt == RETRIES: raise HTTPException(503, f"model backend unavailable: {type(e).__name__}")
    if req.session_id:
        SESSIONS[session_key] = (SESSIONS.get(session_key, []) + req.messages + [out["choices"][0]["message"]])[-20:]
    return out

@router.get("/v1/chat/health")
async def health():
    async with httpx.AsyncClient(timeout=5) as cli:
        try: ok = (await cli.get(f"{BACKEND}/health")).status_code == 200
        except Exception: ok = False
    return {"gateway": "ok", "backend": ok, "documents": len(DOCS)}
