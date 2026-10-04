import re, sys
from collections import Counter
import ocr_eval as E
cat=Counter(); ex={}
for f in sorted(x for x in E.truth if x.startswith(E.SETS[sys.argv[1]])):
    boxes=E.read(f)
    for w in E.truth[f]:
        g=E.norm(w["text"])
        if not re.search(r"\d",g): continue
        cx,cy=(w["box"][0]+w["box"][2])/2,(w["box"][1]+w["box"][3])/2
        cover=[b for b in boxes if b[1]-0.005<=cx<=b[3]+0.005 and b[2]-0.004<=cy<=b[4]+0.004]
        if not cover: continue
        text=" ".join(b[0].lower() for b in cover); core=re.sub(r"[^a-z0-9]","",g)
        if core in re.split(r"[^a-z0-9]+",text) or (len(core)>=3 and core in re.sub(r"[^a-z0-9]","",text)): continue
        toks=re.split(r"[^a-z0-9]+",text)
        wh=(w["box"][3]-w["box"][1])/max(w["box"][2]-w["box"][0],1e-6)
        if wh>2: k="vertical text"
        elif len(core)<=2: k="1-2 char token (qty/page no)"
        elif any(len(t)==len(core) and sum(a!=b for a,b in zip(t,core))==1 for t in toks): k="one character wrong"
        elif any(t.startswith(core[:2]) or t.endswith(core[-2:]) for t in toks if t): k="partly read / split / glued"
        else: k="other"
        cat[k]+=1; ex.setdefault(k,[]).append((w["text"],[b[0][:28] for b in cover]))
print(cat.most_common())
for k,v in ex.items(): print(k, v[:6])
