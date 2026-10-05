"""HTTP API. Sessions have their own workspace and budget; learning is scoped to an API-key owner.

Run:  .venv\\Scripts\\python -m uvicorn server:app --port 8090
Needs the model server:  runtime\\llama\\llama-server.exe -m models\\SmolVLM-256M-Instruct-Q8_0.gguf --mmproj models\\mmproj-SmolVLM-256M-Instruct-Q8_0.gguf -np 4 -t 6 --port 8081
"""
import os, shutil, tempfile, threading, uuid, hashlib
import csv
import io
from pathlib import Path
from datetime import datetime, timezone
from typing import Literal, Annotated
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, Query
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from omni import improve
from omni.agent import Agent, Workspace
from omni.budget import PRESETS, Budget
from omni.evalrun import evaluate
from omni.llm import LLM
from vision_lab.app.gateway import router as chat_router
from vision_lab.app import gateway as vision_gateway

API_KEYS = set(filter(None, os.environ.get("OMNI_API_KEYS", "dev-key").split(",")))
LLM_URL = os.environ.get("OMNI_LLM_URL", os.environ.get("OMNI_LLM", "http://127.0.0.1:8081"))
UPLOADS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uploads")
os.makedirs(UPLOADS, exist_ok=True)

app = FastAPI(title="Omni Agent", version="0.2")
app.include_router(chat_router)
WEB = Path(__file__).resolve().parent / "web"
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/", include_in_schema=False)
def workspace_page():
    return FileResponse(WEB / "index.html")


@app.exception_handler(Exception)
async def unexpected(request: Request, exc: Exception):
    """Never leak a bare 500: say what failed so the caller (and the logs) can act."""
    return JSONResponse(status_code=500, content={"error": type(exc).__name__, "detail": str(exc)[:300]})


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    # Returning raw invalid values (e.g. an overflowed JSON number) can itself fail
    # JSON serialization. Field locations and messages are enough to fix a request.
    return JSONResponse(status_code=422, content={"detail": [
        {"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()
    ]})
llm = LLM(LLM_URL)
OCR_SLOTS = threading.Semaphore(2)     # OCR is the CPU-heavy step; cap how many run at once
SESSIONS: dict[str, dict] = {}


@app.middleware("http")
async def isolate_policy(request: Request, call_next):
    key = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    token = improve.POLICY_OWNER.set(hashlib.sha256(key.encode()).hexdigest())
    try:
        return await call_next(request)
    finally:
        improve.POLICY_OWNER.reset(token)


def auth(authorization: str = Header(default="")):
    key = authorization.removeprefix("Bearer ").strip()
    if key not in API_KEYS:
        raise HTTPException(401, "invalid API key")
    return key


class BudgetSpec(BaseModel):
    preset: str = "balanced"                 # frugal | balanced | thorough
    llm_tokens: int | None = Field(default=None, ge=0)
    seconds: int | None = Field(default=None, ge=0)
    tool_calls: int | None = Field(default=None, ge=0)
    vlm_looks: int | None = Field(default=None, ge=0)
    ocr_pages: int | None = Field(default=None, ge=0)
    auto_grant: dict[Literal["llm_tokens", "seconds", "tool_calls", "vlm_looks", "ocr_pages"], Annotated[int, Field(ge=0)]] = {}  # user-set extra allowance


def make_budget(spec: BudgetSpec) -> Budget:
    if spec.preset not in PRESETS:
        raise HTTPException(422, f"preset must be one of {list(PRESETS)}")
    b = Budget.make(spec.preset, llm_tokens=spec.llm_tokens, seconds=spec.seconds, tool_calls=spec.tool_calls,
                    vlm_looks=spec.vlm_looks, ocr_pages=spec.ocr_pages)
    b.auto_grant = dict(spec.auto_grant)
    return b


def session(sid: str, owner: str) -> dict:
    if sid not in SESSIONS or SESSIONS[sid]["owner"] != owner:
        raise HTTPException(404, "unknown session")
    return SESSIONS[sid]


@app.post("/v1/sessions")
def new_session(spec: BudgetSpec = BudgetSpec(), _=Depends(auth)):
    ws = Workspace()
    sid = uuid.uuid4().hex[:12]
    # Feedback and remembered answers belong to the API-key owner.
    shared = improve.owner_state(_)
    SESSIONS[sid] = {"owner": _, "name": "Untitled workspace", "history": [], "created_at": datetime.now(timezone.utc).isoformat(), "ws": ws, "agent": Agent(ws, llm, shared=shared), "budget": make_budget(spec), "lock": threading.Lock()}
    return {"session": sid, "budget": SESSIONS[sid]["budget"].snapshot()}


@app.get("/v1/sessions")
def list_sessions(_=Depends(auth)):
    return [{"session": sid, "name": s.get("name", "Untitled workspace"), "created_at": s["created_at"], "documents": len(s["ws"].docs)}
            for sid, s in list(SESSIONS.items()) if s["owner"] == _]


class WorkspaceName(BaseModel):
    name: str = Field(min_length=1, max_length=80)


@app.patch("/v1/sessions/{sid}")
def rename_session(sid: str, body: WorkspaceName, _=Depends(auth)):
    s = session(sid, _)
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "workspace name cannot be blank")
    with s["lock"]:
        s["name"] = name
    return {"session": sid, "name": name}


@app.get("/v1/sessions/{sid}/history")
def conversation_history(sid: str, mode: Literal["documents", "chat"] = "documents", _=Depends(auth)):
    s = session(sid, _)
    if mode == "chat":
        # Image bytes are never sent back in the history response or export.
        messages = []
        for m in list(vision_gateway.SESSIONS.get((_, sid), [])):
            content = m.get("content", "")
            attachment = isinstance(content, list) and any(p.get("type") == "image_url" for p in content)
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
            messages.append({"role": m["role"], "content": content, "attachment": attachment})
        return {"messages": messages, "limit": 20}
    with s["lock"]:
        return {"messages": list(s.get("history", [])), "limit": 100}


def csv_cell(value) -> str:
    text = "" if value is None else str(value)
    # Keep uploaded labels/values from becoming executable spreadsheet formulas.
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n")) else text


@app.get("/v1/sessions/{sid}/extractions/{request_id}.csv")
def export_extraction(sid: str, request_id: str, _=Depends(auth)):
    s = session(sid, _)
    with s["lock"]:
        result = next((m.get("result") for m in reversed(s.get("history", []))
                       if m.get("result", {}).get("id") == request_id and m.get("result", {}).get("mode") == "extract"), None)
        if result is None:
            raise HTTPException(404, "Extraction is no longer in this workspace's retained history.")
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(["field", "value", "document", "page", "verified"])
        for f in result.get("fields", []):
            writer.writerow([csv_cell(f.get("field")), csv_cell(f.get("value")), csv_cell(f.get("doc")),
                             f.get("page") or "", "true" if f.get("verified") else "false"])
    # UTF-8 BOM helps spreadsheet tools preserve Bengali and other Unicode text.
    return Response("\ufeff" + output.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="omni-extracted-fields.csv"'})


@app.get("/v1/sessions/{sid}/documents")
def list_documents(sid: str, _=Depends(auth)):
    s = session(sid, _)
    with s["lock"]:
        return [{"doc_id": d.id, "name": d.name, "kind": d.kind, "pages": len(d.pages),
                 "unread_pages": len(s["ws"].pending_ocr([d.id]))} for d in s["ws"].docs.values()]


@app.post("/v1/sessions/{sid}/documents")
def upload(sid: str, file: UploadFile = File(...), read: str = "background", max_pages: int = 60, _=Depends(auth)):
    """read=background (default): return at once and start reading scanned pages in the background, so the first question rarely waits for OCR
    (at most the session's ocr_pages limit; none when it is 0).
    read=lazy: scanned pages are read (OCR) only when a question needs them, within the session's ocr_pages budget.
    read=all: read every scanned page now (costs CPU time, no model tokens; up to max_pages) and cache it, so later questions are instant."""
    if read not in ("background", "lazy", "all"):
        raise HTTPException(422, "read must be background, lazy or all")
    s = session(sid, _)
    path = os.path.join(UPLOADS, f"{uuid.uuid4().hex[:8]}_{os.path.basename(file.filename)}")
    with open(path, "wb") as fh:
        shutil.copyfileobj(file.file, fh)
    with s["lock"]:
        try:
            d = s["ws"].add(path, name=os.path.basename(file.filename))
        except Exception as e:  # unreadable / corrupt / unsupported file
            os.remove(path)
            raise HTTPException(422, f"cannot read this file: {type(e).__name__}")
        read_now = 0
        if read == "all":
            OCR_SLOTS.acquire()
            try:
                for _d, n in s["ws"].pending_ocr([d.id])[:max_pages]:
                    s["ws"].ocr(_d, n)
                    read_now += 1
            finally:
                OCR_SLOTS.release()
        pending = len(s["ws"].pending_ocr([d.id]))
    reading = 0
    if read == "background" and pending:
        reading = int(min(max_pages, pending, s["budget"].user_limits.get("ocr_pages", 0)))
        if reading:
            threading.Thread(target=_read_in_background, args=(s, d.id, reading), daemon=True).start()
    return {"doc_id": d.id, "name": d.name, "kind": d.kind, "pages": len(d.pages), "pages_read_now": read_now,
            "pages_reading_in_background": reading, "pages_still_unread": pending}


def document(s: dict, doc_id: str):
    d = s["ws"].docs.get(doc_id)
    if d is None:
        raise HTTPException(404, "unknown document")
    return d


@app.get("/v1/sessions/{sid}/documents/{doc_id}/pages/{number}")
def document_page(sid: str, doc_id: str, number: int, _=Depends(auth)):
    s = session(sid, _)
    with s["lock"]:
        d = document(s, doc_id)
        if number < 1 or number > len(d.pages):
            raise HTTPException(404, "unknown page")
        p = d.pages[number - 1]
        return {"doc_id": d.id, "name": d.name, "page": number, "pages": len(d.pages),
                "source": p.source, "text": p.text, "needs_ocr": p.source in ("none", "scan-text")}


@app.get("/v1/sessions/{sid}/documents/{doc_id}/pages/{number}/text")
def export_page_text(sid: str, doc_id: str, number: int, _=Depends(auth)):
    page = document_page(sid, doc_id, number, _)
    if not page["text"].strip():
        raise HTTPException(409, "No cached text is available for this page. Read it within your resource budget first.")
    note = "Partial cached text; this page still needs OCR." if page["needs_ocr"] else (
        "OCR text; check uncertain readings against the original file." if page["source"] == "ocr" else
        "Extracted document text; original page formatting is not preserved.")
    content = f"Document: {page['name']}\nPage: {number} of {page['pages']}\nSource: {page['source']}\nNote: {note}\n\n{page['text']}"
    return Response("\ufeff" + content, media_type="text/plain", headers={
        "Content-Disposition": f'attachment; filename="omni-page-{number}.txt"', "Cache-Control": "no-store"})


@app.get("/v1/sessions/{sid}/documents/{doc_id}/search")
def search_document(sid: str, doc_id: str, q: str = Query(min_length=2, max_length=200), _=Depends(auth)):
    query = q.strip()
    if len(query) < 2:
        raise HTTPException(422, "search needs at least two characters")
    s = session(sid, _)
    with s["lock"]:
        d = document(s, doc_id)
        hits = []
        for p in d.pages:
            position = p.text.lower().find(query.lower())
            if position >= 0:
                hits.append({"page": p.n, "source": p.source,
                             "excerpt": p.text[max(0, position - 80):position + len(query) + 160]})
        return {"matches": hits[:50], "total_matching_pages": len(hits),
                "unread_pages": len(s["ws"].pending_ocr([d.id]))}


def _read_in_background(s, doc_id, limit):
    """OCR a fresh upload page by page. The session lock is taken per page, so a question asked meanwhile waits for at most one page,
    and a page the question already read is skipped."""
    for _ in range(limit):
        with s["lock"]:
            todo = s["ws"].pending_ocr([doc_id])
            if not todo:
                return
            with OCR_SLOTS:
                try:
                    s["ws"].ocr(*todo[0])
                except Exception:
                    return  # a page that cannot be read now is left to the on-demand path, which reports the problem


class AskReq(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    doc_ids: list[str] = []
    mode: str = "auto"                       # auto | extract
    targets: list[str] = []                  # extract: named fields (empty = every label:value pair found)
    budget: BudgetSpec | None = None         # optional per-request override of the session budget


@app.post("/v1/sessions/{sid}/ask")
def ask(sid: str, req: AskReq, _=Depends(auth)):
    s = session(sid, _)
    unknown = [d for d in req.doc_ids if d not in s["ws"].docs]
    if unknown:
        raise HTTPException(404, f"unknown document id(s): {unknown}")
    b = make_budget(req.budget) if req.budget else s["budget"]
    with s["lock"]:                          # one request at a time per user; different users run in parallel
        b.new_turn()
        need_ocr = bool(s["ws"].pending_ocr(req.doc_ids or None))
        if need_ocr:
            OCR_SLOTS.acquire()
        try:
            if req.mode == "extract":
                result = s["agent"].extract(req.question, b, req.doc_ids or None, uuid.uuid4().hex[:10], targets=req.targets or None)
            else:
                result = s["agent"].ask(req.question, b, req.doc_ids or None, req.mode)
            text = result.get("answer") or "\n".join(
                f"{f['field']}: {f.get('value') if f.get('value') is not None else 'Not found'}" + ("" if f.get("verified") else " (unverified)")
                for f in result.get("fields", [])) or "No fields found."
            s["history"] = (s.get("history", []) + [
                {"role": "user", "content": req.question},
                {"role": "assistant", "content": text, "result": result},
            ])[-100:]
            return result
        finally:
            if need_ocr:
                OCR_SLOTS.release()


class Feedback(BaseModel):
    id: str
    verdict: Literal["good", "bad"]
    correction: str | None = None


@app.post("/v1/sessions/{sid}/feedback")
def feedback(sid: str, fb: Feedback, _=Depends(auth)):
    return session(sid, _)["agent"].feedback(fb.id, fb.verdict, fb.correction)


@app.get("/v1/sessions/{sid}/budget")
def get_budget(sid: str, _=Depends(auth)):
    s = session(sid, _)
    with s["lock"]:
        b = s["budget"]
        return {"budget": b.snapshot(), "pending_requests": [p for p in b.pending if p["status"] == "waiting"], "audit": b.log[-30:]}


class Resolve(BaseModel):
    request_id: int = Field(ge=0)
    approve: bool


@app.post("/v1/sessions/{sid}/budget/resolve")
def resolve(sid: str, r: Resolve, _=Depends(auth)):
    s = session(sid, _)
    with s["lock"]:
        b = s["budget"]
        if r.request_id >= len(b.pending):
            raise HTTPException(404, "no such request")
        if b.pending[r.request_id]["status"] != "waiting":
            raise HTTPException(409, "request already resolved")
        b.resolve(r.request_id, r.approve)
        return {"budget": b.snapshot()}


class Throttle(BaseModel):
    resource: str
    cap: float = Field(ge=0, allow_inf_nan=False)


class LimitChanges(BaseModel):
    limits: dict[Literal["llm_tokens", "seconds", "tool_calls", "vlm_looks", "ocr_pages"], Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(min_length=1)


def change_limits(s: dict, limits: dict) -> dict:
    for resource, cap in limits.items():
        if resource not in s["budget"].user_limits:
            raise HTTPException(422, "unknown resource")
        if resource != "seconds" and not float(cap).is_integer():
            raise HTTPException(422, "token, tool, vision and OCR limits must be whole numbers")
    with s["lock"]:
        b = s["budget"]
        before = dict(b.user_limits)
        b.user_limits.update(limits)
        b.log.append({"op": "user_limits", "before": before, "limits": dict(limits)})
        return {"budget": b.snapshot(), "pending_requests": [p for p in b.pending if p["status"] == "waiting"]}


@app.post("/v1/sessions/{sid}/budget/limits")
def set_limits(sid: str, changes: LimitChanges, _=Depends(auth)):
    """Validate all supplied ceilings first, then change them together without resetting usage."""
    return change_limits(session(sid, _), changes.limits)


@app.post("/v1/sessions/{sid}/budget/limit")
def set_limit(sid: str, t: Throttle, _=Depends(auth)):
    """The user can lower or raise their own hard ceiling at any time."""
    return change_limits(session(sid, _), {t.resource: t.cap})


@app.get("/v1/policy")
def policy(_=Depends(auth)):
    SHARED = improve.owner_state(_)
    live, cand = SHARED[0], SHARED[1]
    return {"live_version": live.version, "candidate_updates": cand.n_updates, "memory_items": len(SHARED[2].items),
            "trust": SHARED[3].c, "token_stats": SHARED[4].s}


@app.post("/v1/policy/gate")
def gate(pdf: str = "synth_text.pdf", _=Depends(auth)):
    """Evaluate live vs candidate on the frozen eval; promote the candidate only if it does not regress."""
    SHARED = improve.owner_state(_)
    live_r, cand_r = evaluate(SHARED[0], pdf, llm=llm), evaluate(SHARED[1], pdf, llm=llm)
    ok, why = improve.promote(cand_r, live_r, min_gain=0.01)  # a tie proves nothing: require a measured gain
    if ok:
        SHARED[0].theta, SHARED[0].version = improve.load_live().theta, improve.load_live().version
    return {"promoted": ok, "reason": why, "live": {k: live_r[k] for k in ("overall", "by_cat")}, "candidate": {k: cand_r[k] for k in ("overall", "by_cat")}}


@app.post("/v1/policy/rollback")
def rollback(_=Depends(auth)):
    SHARED = improve.owner_state(_)
    ok = improve.rollback()
    if ok:
        j = improve.load_live()
        SHARED[0].theta, SHARED[0].version = j.theta, j.version
    return {"rolled_back": ok}


@app.get("/health")
def health():
    return {"api": "ok", "model_server": llm.alive(), "vision_specialist": llm.vision, "sessions": len(SESSIONS)}
