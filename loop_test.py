import json
import httpx

cli = httpx.Client(base_url="http://127.0.0.1:8090", headers={"Authorization": "Bearer dev-key"}, timeout=600)
sid = cli.post("/v1/sessions", json={"preset": "balanced"}).json()["session"]
with open("data/synth_text_final.pdf", "rb") as fh:
    doc = cli.post(f"/v1/sessions/{sid}/documents", files={"file": ("f.pdf", fh)}).json()["doc_id"]

q = "Briefly describe what kind of document this is."
r1 = cli.post(f"/v1/sessions/{sid}/ask", json={"question": q, "doc_ids": [doc]}).json()
print("1st answer :", r1["answer"][:90], "| source:", r1["source"], "| tokens:", r1["budget"]["llm_tokens"]["used"])

fix = "A synthetic (fictional) home loan application form used for testing document AI."
print("feedback   :", cli.post(f"/v1/sessions/{sid}/feedback", json={"id": r1["id"], "verdict": "bad", "correction": fix}).json())

r2 = cli.post(f"/v1/sessions/{sid}/ask", json={"question": q, "doc_ids": [doc]}).json()
print("2nd answer :", r2["answer"][:90], "| source:", r2["source"], "| tokens:", r2["budget"]["llm_tokens"]["used"])

# another user (new session, same documents) benefits from the confirmed answer too
sid2 = cli.post("/v1/sessions", json={"preset": "balanced"}).json()["session"]
with open("data/synth_text_final.pdf", "rb") as fh:
    doc2 = cli.post(f"/v1/sessions/{sid2}/documents", files={"file": ("f.pdf", fh)}).json()["doc_id"]
r3 = cli.post(f"/v1/sessions/{sid2}/ask", json={"question": q, "doc_ids": [doc2]}).json()
print("other user :", r3["answer"][:90], "| source:", r3["source"])

print("policy     :", cli.get("/v1/policy").json())
g = cli.post("/v1/policy/gate", params={"pdf": "synth_text.pdf"}).json()
print("gate       :", json.dumps({k: g[k] for k in ("promoted", "reason")}), "| live", round(g["live"]["overall"], 3), "| candidate", round(g["candidate"]["overall"], 3))
print("policy     :", cli.get("/v1/policy").json())
