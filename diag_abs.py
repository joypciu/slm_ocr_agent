import json,sys
from omni.agent import Workspace
ws=Workspace(); ids={}
for f in ("data/evalset_real.json","data/evalset_real_holdout.json"):
    for it in json.load(open(f)):
        if any(k in it["q"] for k in ("Where does the customer work","customer's designation","required for lawyers","salary of a salaried")):
            d=ids.get(it["doc"]) or ws.add(it["doc"]); ids[it["doc"]]=d
            for p in d.pages: pass
            ws.read_all([d.id]) if hasattr(ws,"read_all") else None
            print(it["q"][:50],"|",ws.unseen_in(it["q"],[d.id]),round(ws.absent_share(it["q"],[d.id]),2),round(ws.px.best_coverage(it["q"],[d.id]),2))
