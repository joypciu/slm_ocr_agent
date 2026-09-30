"""Build a FROZEN eval set from the synthetic home-loan forms (ground truth = the generated text layer, not model output).
Creates a text PDF and a scan-only PDF (rasterized, no text layer) of the same forms, plus data/evalset.json."""
import json, re, random
import fitz

import os
SRC = os.environ.get("SYNTH_FORMS_PDF", "synthetic_home_loan_forms_ALL50.pdf")  # a PDF of synthetic (fictional) loan forms; not included in this repository
import sys
START = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_FORMS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
SUF = sys.argv[3] if len(sys.argv) > 3 else ""
random.seed(7)
src = fitz.open(SRC)
starts = [i for i in range(len(src)) if "Purpose of Loan:" in src[i].get_text()]
print("form starts found:", len(starts), "pages:", len(src))
bounds = [(s, (starts[i + 1] if i + 1 < len(starts) else len(src))) for i, s in enumerate(starts)]
FIELDS = {
    "Full Name": r"Full Name:\s*(.+?)\s+Profession:",
    "Loan Amount (BDT)": r"Loan Amount \(BDT\):\s*([\d,]+)",
    "Date of Birth": r"Date of Birth:\s*(\d\d/\d\d/\d{4})",
    "TIN": r"TIN:\s*(\d+)",
    "Purpose of Loan": r"Purpose of Loan:\s*(.+?)\s+Loan Amount",
}
text_pdf, scan_pdf = fitz.open(), fitz.open()
items, page_no = [], 0
for fi, (s, e) in enumerate(bounds[START:START + N_FORMS]):
    first = re.sub(r"[ \t]+", " ", src[s].get_text())
    kv = {}
    for label, pat in FIELDS.items():
        m = re.search(pat, first.replace("\n", " "))
        if m:
            kv[label] = m.group(1).strip()
    name = kv.get("Full Name")
    if not name:
        continue
    for label, phr in [("Loan Amount (BDT)", "loan amount in BDT"), ("Date of Birth", "date of birth"), ("TIN", "TIN"), ("Purpose of Loan", "purpose of the loan")]:
        if label in kv:
            q = f"What is the {phr} of the applicant {name}?" if label not in ("Loan Amount (BDT)", "Purpose of Loan") else f"What is the {phr} for the application of {name}?"
            items.append({"q": q, "gold": kv[label], "cat": "text-field", "form": fi})
    for p in range(s, e):
        text_pdf.insert_pdf(src, from_page=p, to_page=p)
        pix = src[p].get_pixmap(dpi=150)
        pg = scan_pdf.new_page(width=src[p].rect.width, height=src[p].rect.height)
        pg.insert_image(pg.rect, stream=pix.tobytes("jpeg", jpg_quality=80))
    page_no += e - s
# unanswerable questions: the right behaviour is to say it is not in the documents
first_name = next((i["q"].split("applicant ")[-1].rstrip("?") for i in items if "applicant " in i["q"]), "the applicant")
for q in ["What is the applicant's blood group?", "What is the name of the applicant's pet?", f"What is the credit score of {first_name}?"]:
    items.append({"q": q, "gold": None, "cat": "unanswerable", "form": -1})
text_pdf.save(f"data/synth_text{SUF}.pdf"); scan_pdf.save(f"data/synth_scan{SUF}.pdf")
json.dump(items, open(f"data/evalset{SUF}.json", "w"), indent=1)
from collections import Counter
print("pages written:", page_no, "| questions:", len(items), dict(Counter(i["cat"] for i in items)))
print("sample:", items[0], items[-1])
