"""HTTP API. Sessions have their own workspace and budget; learning is scoped to an API-key owner.

Run:  .venv\\Scripts\\python -m uvicorn server:app --port 8090
Needs the model server:  runtime\\llama\\llama-server.exe -m models\\SmolVLM-256M-Instruct-Q8_0.gguf --mmproj models\\mmproj-SmolVLM-256M-Instruct-Q8_0.gguf -np 4 -t 6 --port 8081
"""
import os, shutil, tempfile, threading, uuid, hashlib
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from omni import improve
from omni.agent import Agent, Workspace
from omni.budget import PRESETS, Budget
from omni.evalrun import evaluate
from omni.llm import LLM
from vision_lab.app.gateway import router as chat_router

API_KEYS = set(filter(None, os.environ.get("OMNI_API_KEYS", "dev-key").split(",")))
LLM_URL = os.environ.get("OMNI_LLM_URL", os.environ.get("OMNI_LLM", "http://127.0.0.1:8081"))
UPLOADS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uploads")
os.makedirs(UPLOADS, exist_ok=True)

app = FastAPI(title="Omni Agent", version="0.2")
app.include_router(chat_router)


@app.exception_handler(Exception)
async def unexpected(request: Request, exc: Exception):
    """Never leak a bare 500: say what failed so the caller (and the logs) can act."""
    return JSONResponse(status_code=500, content={"error": type(exc).__name__, "detail": str(exc)[:300]})
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
    llm_tokens: int | None = None
    seconds: int | None = None
    tool_calls: int | None = None
    vlm_looks: int | None = None
    ocr_pages: int | None = None
    auto_grant: dict[str, int] = {}          # extra the agent may take WITHOUT asking, per resource (user-set)


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
    SESSIONS[sid] = {"owner": _, "ws": ws, "agent": Agent(ws, llm, shared=shared), "budget": make_budget(spec), "lock": threading.Lock()}
    return {"session": sid, "budget": SESSIONS[sid]["budget"].snapshot()}


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
            d = s["ws"].add(path)
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
    b.new_turn()
    with s["lock"]:                          # one request at a time per user; different users run in parallel
        need_ocr = bool(s["ws"].pending_ocr(req.doc_ids or None))
        if need_ocr:
            OCR_SLOTS.acquire()
        try:
            if req.mode == "extract":
                return s["agent"].extract(req.question, b, req.doc_ids or None, uuid.uuid4().hex[:10], targets=req.targets or None)
            return s["agent"].ask(req.question, b, req.doc_ids or None, req.mode)
        finally:
            if need_ocr:
                OCR_SLOTS.release()


class Feedback(BaseModel):
    id: str
    verdict: str                             # good | bad
    correction: str | None = None


@app.post("/v1/sessions/{sid}/feedback")
def feedback(sid: str, fb: Feedback, _=Depends(auth)):
    return session(sid, _)["agent"].feedback(fb.id, fb.verdict, fb.correction)


@app.get("/v1/sessions/{sid}/budget")
def get_budget(sid: str, _=Depends(auth)):
    b = session(sid, _)["budget"]
    return {"budget": b.snapshot(), "pending_requests": [p for p in b.pending if p["status"] == "waiting"], "audit": b.log[-30:]}


class Resolve(BaseModel):
    request_id: int
    approve: bool


@app.post("/v1/sessions/{sid}/budget/resolve")
def resolve(sid: str, r: Resolve, _=Depends(auth)):
    b = session(sid, _)["budget"]
    if r.request_id >= len(b.pending):
        raise HTTPException(404, "no such request")
    b.resolve(r.request_id, r.approve)
    return {"budget": b.snapshot()}


class Throttle(BaseModel):
    resource: str
    cap: float


@app.post("/v1/sessions/{sid}/budget/limit")
def set_limit(sid: str, t: Throttle, _=Depends(auth)):
    """The user can lower or raise their own hard ceiling at any time."""
    b = session(sid, _)["budget"]
    if t.resource not in b.user_limits:
        raise HTTPException(422, f"resource must be one of {list(b.user_limits)}")
    b.user_limits[t.resource] = t.cap
    return {"budget": b.snapshot()}


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
