import json, threading, time
import httpx

BASE, H = "http://127.0.0.1:8090", {"Authorization": "Bearer dev-key"}
cli = httpx.Client(base_url=BASE, headers=H, timeout=180)


def new_session(**budget):
    return cli.post("/v1/sessions", json=budget).json()["session"]


def upload(sid, path):
    with open(path, "rb") as fh:
        return cli.post(f"/v1/sessions/{sid}/documents", files={"file": (path.split("/")[-1], fh)}).json()


def ask(sid, q, **kw):
    return cli.post(f"/v1/sessions/{sid}/ask", json={"question": q, **kw}).json()


def show(title, r):
    keys = ("answer", "strategy", "support", "verified", "source", "seconds", "budget_note", "pending_requests", "ambiguous")
    print(f"\n== {title}")
    for k in keys:
        if k in r:
            v = r[k]
            print(f"   {k}: {v if k != 'budget' else v}")
    if "budget" in r:
        print("   spent:", {k: v["used"] for k, v in r["budget"].items() if v["used"]})
    if r.get("steps"):
        print("   steps:", r["steps"])


# 1) plain lookup: zero-token path
s = new_session(preset="frugal")
d = upload(s, "data/synth_text_final.pdf"); print("uploaded:", d)
r = ask(s, "What is the TIN of the applicant " + json.load(open("data/evalset_final.json"))[2]["q"].split("applicant ")[-1].rstrip("?"), doc_ids=[d["doc_id"]])
show("field lookup on a frugal budget", r)

# 2) a question that needs the language model, with a token budget that is too small
s2 = new_session(preset="frugal", llm_tokens=60)
d2 = upload(s2, "data/synth_text_final.pdf")
r = ask(s2, "Briefly describe what kind of document this is.", doc_ids=[d2["doc_id"]])
show("LLM question with only 60 tokens (agent must ask the user)", r)
pend = cli.get(f"/v1/sessions/{s2}/budget").json()["pending_requests"]
print("   pending user decisions:", pend)
if pend:
    cli.post(f"/v1/sessions/{s2}/budget/resolve", json={"request_id": pend[0]["id"], "approve": True})
    r = ask(s2, "Briefly describe what kind of document this is.", doc_ids=[d2["doc_id"]])
    show("same question after the user approved the extension", r)

# 3) auto-grant: the user pre-approves up to 800 extra tokens; agent extends itself, audit shows it
s3 = new_session(preset="frugal", llm_tokens=60, auto_grant={"llm_tokens": 800})
d3 = upload(s3, "data/synth_text_final.pdf")
r = ask(s3, "Briefly describe what kind of document this is.", doc_ids=[d3["doc_id"]])
show("auto-grant (no prompt to the user)", r)
print("   audit:", [a for a in cli.get(f"/v1/sessions/{s3}/budget").json()["audit"] if a["op"] in ("auto_grant", "throttle", "request")][:4])

# 4) four users in parallel, each with an own session and different questions
items = json.load(open("data/evalset_final.json"))
res = {}


def user(i):
    sid = new_session(preset="balanced")
    d = upload(sid, "data/synth_text_final.pdf")
    t = time.time()
    ok = 0
    for it in items[i * 6:(i * 6) + 6]:
        a = ask(sid, it["q"], doc_ids=[d["doc_id"]])["answer"]
        ok += (it["gold"] or "").replace(",", "") in a.replace(",", "") if it["gold"] else ("Not found" in a)
    res[i] = (ok, round(time.time() - t, 1))


t0 = time.time(); ts = [threading.Thread(target=user, args=(i,)) for i in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
print("\n== 4 parallel users x 6 questions:", res, "wall %.1fs" % (time.time() - t0))
print("health:", cli.get("/health").json())
