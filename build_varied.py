"""Build a varied-forms benchmark from public datasets with ground truth (run with an environment that has `datasets`):
   receipts (CORD v2), invoices (Donut invoices), noisy scanned business forms (FUNSD).  Output: data/varied/*, data/evalset_varied.json"""
import json, os, random, re, sys
from datasets import load_dataset

OUT = "data/varied"
N_CORD, N_INV, N_FUNSD = int(os.environ.get("N_CORD", 30)), int(os.environ.get("N_INV", 12)), int(os.environ.get("N_FUNSD", 24))
random.seed(11)
items = []


def add(doc, cat, q, gold):
    items.append({"doc": doc, "cat": cat, "q": q, "gold": gold})


def digits(s):
    return re.sub(r"\D", "", str(s))


# ------------------------------------------------------------------ receipts
os.makedirs(f"{OUT}/cord", exist_ok=True)
n = 0
for ex in load_dataset("naver-clova-ix/cord-v2", split="test", streaming=True):
    gt = json.loads(ex["ground_truth"])["gt_parse"]
    tot = gt.get("total", {}) or {}
    sub = gt.get("sub_total", {}) or {}
    if not tot.get("total_price"):
        continue
    p = f"{OUT}/cord/r{n:03d}.png"
    ex["image"].convert("RGB").save(p)
    add(p, "receipt", "What is the total amount?", {"digits": digits(tot["total_price"])})
    if sub.get("tax_price") and digits(sub["tax_price"]):
        add(p, "receipt", "What is the tax amount?", {"digits": digits(sub["tax_price"])})
    if sub.get("subtotal_price") and digits(sub["subtotal_price"]):
        add(p, "receipt", "What is the subtotal?", {"digits": digits(sub["subtotal_price"])})
    menu = gt.get("menu")
    first = menu[0] if isinstance(menu, list) and menu else (menu if isinstance(menu, dict) else None)
    if first and first.get("nm") and first.get("price") and len(first["nm"]) >= 4 and digits(first["price"]):
        add(p, "receipt", f"What is the price of {first['nm']}?", {"digits": digits(first["price"])})
    n += 1
    if n >= N_CORD:
        break
print("receipts:", n, "docs")

# ------------------------------------------------------------------ invoices
os.makedirs(f"{OUT}/invoice", exist_ok=True)
n = 0
for ex in load_dataset("katanaml-org/invoices-donut-data-v1", split="test", streaming=True):
    gt = json.loads(ex["ground_truth"])["gt_parse"]
    h, s = gt.get("header", {}), gt.get("summary", {})
    if not (h.get("invoice_no") and h.get("invoice_date") and s.get("total_gross_worth")):
        continue
    img = ex["image"].convert("RGB")
    img.thumbnail((1754, 2480))  # about A4 at 150 dpi
    p = f"{OUT}/invoice/i{n:03d}.png"
    img.save(p)
    add(p, "invoice", "What is the invoice number?", {"digits": digits(h["invoice_no"])})
    add(p, "invoice", "What is the invoice date?", {"digits": digits(h["invoice_date"])})
    if h.get("seller_tax_id"):
        add(p, "invoice", "What is the seller's tax ID?", {"digits": digits(h["seller_tax_id"])})
    add(p, "invoice", "What is the total gross worth?", {"digits": digits(s["total_gross_worth"])})
    words = re.findall(r"[A-Za-z]+", h.get("seller", ""))[:2]
    if len(words) == 2:
        add(p, "invoice", "What is the name of the seller?", {"words": words})
    words = re.findall(r"[A-Za-z]+", h.get("client", ""))[:2]
    if len(words) == 2:
        add(p, "invoice", "What is the name of the client?", {"words": words})
    n += 1
    if n >= N_INV:
        break
print("invoices:", n, "docs")

# ------------------------------------------------------------------ FUNSD (noisy scanned forms): label -> value pairs from the dataset's own annotation
os.makedirs(f"{OUT}/funsd", exist_ok=True)


def entities(words, boxes, tags):
    ents, cur = [], None
    for w, b, t in zip(words, boxes, tags):
        kind = {1: "H", 2: "H", 3: "Q", 4: "Q", 5: "A", 6: "A"}.get(t)
        begin = t in (1, 3, 5)
        if kind is None:
            cur = None
            continue
        if cur is None or begin or cur["kind"] != kind:
            cur = {"kind": kind, "words": [], "x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3]}
            ents.append(cur)
        cur["words"].append(w)
        cur["x0"], cur["y0"], cur["x1"], cur["y1"] = min(cur["x0"], b[0]), min(cur["y0"], b[1]), max(cur["x1"], b[2]), max(cur["y1"], b[3])
    return ents


n = 0
for ex in load_dataset("nielsr/funsd", split="test", streaming=True):
    ents = entities(ex["words"], ex["bboxes"], ex["ner_tags"])
    qs = [e for e in ents if e["kind"] == "Q"]
    ans = [e for e in ents if e["kind"] == "A"]
    pairs = []
    for q in qs:
        qt = " ".join(q["words"]).strip()
        if not (2 <= len(qt) <= 40) or not re.search(r"[A-Za-z]{2,}", qt) or not qt.endswith(":"):
            continue
        qh = q["y1"] - q["y0"]
        cands = [a for a in ans if a["x0"] >= q["x1"] - 8 and abs((a["y0"] + a["y1"]) / 2 - (q["y0"] + q["y1"]) / 2) <= 0.8 * max(qh, a["y1"] - a["y0"])]
        if not cands:
            continue
        a = min(cands, key=lambda c: c["x0"] - q["x1"])
        at = " ".join(a["words"]).strip()
        if 1 <= len(at) <= 40 and re.search(r"[A-Za-z0-9]{2,}", at):
            pairs.append((qt, at))
    if len({p[0].lower() for p in pairs}) < 2:
        continue
    seen = set()
    chosen = []
    for qt, at in pairs:
        if qt.lower() not in seen and sum(1 for p in pairs if p[0].lower() == qt.lower()) == 1:  # the label must be unique on the form
            seen.add(qt.lower())
            chosen.append((qt, at))
    if len(chosen) < 2:
        continue
    p = f"{OUT}/funsd/f{n:03d}.png"
    ex["image"].convert("RGB").save(p)
    for qt, at in random.sample(chosen, min(4, len(chosen))):
        add(p, "funsd", f"What is the value of '{qt.rstrip(':').strip()}'?", {"text": at})
    n += 1
    if n >= N_FUNSD:
        break
print("funsd forms:", n, "docs")

json.dump(items, open("data/evalset_varied.json", "w"), indent=1, ensure_ascii=False)
from collections import Counter
print(len(items), "questions", dict(Counter(i["cat"] for i in items)))
for it in [i for i in items if i["cat"] == "funsd"][:8]:
    print("   funsd sample:", it["q"], "->", it["gold"]["text"])
