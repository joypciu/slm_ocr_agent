# Omni Agent

Open **http://127.0.0.1:8090** for the new document workspace. Connect using an
`OMNI_API_KEYS` key (the development default is `dev-key`), create a workspace,
and drop in files. Ask document questions or extract fields, inspect source
excerpts and OCR progress, send feedback, approve resource requests, and export
the conversation as JSON. Light/dark themes and mobile layouts work offline.
The key is kept in tab-scoped session storage and is excluded from exports.
Extracted fields appear in a table with values, source-page links, and individual
verification status. **Export fields CSV** downloads rows with `field`, `value`,
`document`, `page`, and `verified` columns, preserving Unicode text. Potential
spreadsheet formula cells are prefixed with an apostrophe; this also treats
negative values conservatively as text. `GET /v1/sessions/{id}/extractions/{request_id}.csv`
uses the same API-key ownership as the workspace. Exports remain available while
the extraction is in the retained in-memory history and do not rerun extraction.
**Edit workspace limits** changes document-question ceilings for tokens, OCR pages,
vision looks, tools, and time. Changes save together, keep recorded usage, and
are logged in the resource audit. Lowering a ceiling below usage leaves zero
allowance; agent soft caps can remain lower. Time resets per question, while the
other question-resource counts accumulate in the workspace. Negative, fractional
resource counts and non-finite limits are rejected. Settings do not cancel uploads
or background OCR already running and do not control Vision chat's backend limits.
`POST /v1/sessions/{id}/budget/limits` accepts a nonempty `limits` object and changes
the supplied ceilings atomically. The existing single-resource `/budget/limit`
endpoint uses the same validation and locking.
Use **Read** beside a file or **Read page text** on an evidence card to inspect
the full cached page text, navigate pages, and search that document. Search shows
matching pages and highlights your query after jumping to a result. It uses cached
text only and explicitly reports pages still awaiting OCR; preview/search never
start model inference or OCR. OCR text is marked for review. **Read new scans**
lets you choose background reading or reading only when you ask a question.
This is a text viewer; compare OCR readings with your original file as needed.
Name workspaces with **Rename workspace**. Switching workspaces or reloading
restores document answers and their evidence. Document Q&A and Vision chat have
separate visible histories; the last selected workspace is remembered in this tab.
History retains the latest 50 document question/answer pairs and 20 vision-chat
messages. Image bytes are excluded from history responses and exports.

**Vision chat** supports text conversations and image attachments. It uses the
model backend's token limits; the document resource-budget panel is hidden in
this mode. No model is automatically downloaded or started by the web UI.
Server-side sessions and conversation history remain in memory and disappear
on server restart; JSON exports provide a manual record.

Owner-scoped `GET /v1/sessions` and `GET /v1/sessions/{id}/documents` let clients
resume an existing workspace and inspect unread-page counts. Field extraction
now handles single-document `Label: value` lines without repeated-label training.
Unread scanned pages are OCRed within the document session's resource limits.
Text-layer fields are grounded in extracted text; OCR fields remain unverified.
`PATCH /v1/sessions/{id}` accepts a `name` (1–80 characters), and
`GET /v1/sessions/{id}/history?mode=documents|chat` returns owner-scoped history.
Owner-scoped `GET /v1/sessions/{id}/documents/{doc_id}/pages/{number}` provides
cached page text without filesystem paths. `/search?q=...` searches cached text,
returning the first 50 matching pages and the total match/unread-page counts.

SLM Vision Chat is now integrated into this repository as `vision_lab/`.
The same `server:app` exposes its `/v1/chat/completions` text/image chat and
`/v1/documents` embedding retrieval endpoints alongside Omni's OCR/session API.
See [vision_lab/README.md](vision_lab/README.md) for optional dependencies and
training tools. Models and datasets remain local and are excluded from Git.

Sessions, chat history, confirmed answers, routing feedback, and policy state
are scoped to the caller's API key. Use distinct keys for separate users.
Previously shared policy files remain on disk but are not automatically imported
into any user's API policy. CLI benchmarks still use their legacy local policy.

AI Gateway can provide the models: set `OMNI_LLM_URL` and `OMNI_VISION_URL` to
its base URL **without `/v1`**, `OMNI_UPSTREAM_API_KEY` to a gateway user key,
and `OMNI_UPSTREAM_MODEL` / `OMNI_UPSTREAM_VISION_MODEL` to registered model IDs.
Environment settings are listed in `.env.example`; load them into your shell.
Run uvicorn directly when using existing gateway backends; `start.ps1` starts
local llama.cpp servers. The agent API itself is a separate service; Gateway
proxies model inference rather than Omni's session/upload endpoints.

A CPU-only document and picture agent. It reads PDFs, scans, photos, Word, CSV and text files; answers questions; extracts fields;
uses specialist sub-agents automatically; lets the user and the agent share control of tokens and other resources; and improves from
feedback without retraining any model. Runs fully offline.

## Run
```
.\start.ps1                    # full: Qwen3-0.6B reasoner + SmolVLM-256M vision specialist + OCR   (~691 MB of weights)
.\start.ps1 -Profile docs      # reasoner + OCR only, no picture understanding                     (~411 MB, under 500)
.\start.ps1 -Profile vision    # one SmolVLM model for everything, weakest on documents            (~313 MB, under 500)
```
API on :8090 (key from env `OMNI_API_KEYS`, default `dev-key`). Flow: `POST /v1/sessions` (budget) -> `/documents?read=background|lazy|all` (background, the default, starts OCR at upload so the first question rarely waits) -> `/ask` ->
`/feedback`. Also `/budget`, `/budget/resolve`, `/budget/limit`, `/v1/policy`, `/v1/policy/gate`, `/v1/policy/rollback`, `/health`.
Tests: `python e2e_test.py` (59 checks through the API; start it with an empty `policy/memory.json`, remembered answers from an earlier run fail the learning check), `run_full_eval.py`, `run_holdout.py`, `api_test.py`.

Browser user-flow verification: install `playwright` and `uvicorn`, run
`python -m playwright install chromium`, then `python e2e/workspace_browser.py`.
It starts an isolated server with synthetic documents, verifies naming, uploads,
extraction/evidence, switching/reload, separate modes, export payloads, owner
isolation, and mobile layouts. Vision replies use a deterministic test backend;
this checks the integration rather than model answer quality. GitHub CI runs it
on every push and pull request. The test checks the generated export Blob;
operating-system download saving is outside this test's verification scope.

## Architecture
```
question -> memory of confirmed answers
         -> photo route (image with little text -> vision sub-agent looks at the pixels)
         -> strategies, escalating while the answer is unverified:
              fields  learned "Label: value" lookup, 0 tokens
              text    reads candidate pages one at a time in rank order, stops at the first supported answer
              ocr     reads scanned pages on demand (PP-OCR ONNX; embedded text layers of scans are not trusted), cached
         -> refinement by sub-agents (only where they help, only on scanned/handwritten material):
              vision reader     re-reads the row as an image when a date/amount/number must be checked; OCR and vision compared.
                                OCR stays the primary reader (it beat vision on real handwriting); on disputes the learned trust
                                starts with a slight OCR lean and moves with user feedback
              checkbox reader   model-free image processing (ink in the square next to each option); experimental, shown as an
                                alternative reading, never replaces the answer (see limits)
              verifier          format checks; when the two readings differ, learned trust decides and the other reading is shown
```
* **Tokens and resources, shared control:** the user sets hard limits (tokens, seconds, tool calls, vision looks, OCR pages). The agent sets its
  own per-question allowance from what each question type has really cost (never above the user's ceiling), raises it when a sub-agent
  needs room, refuses a prompt that will not fit before calling the model, and can ask for more (auto-granted only inside a user-set
  allowance, otherwise it waits for the user). Everything is audited (`/budget`).
* **Self-improvement (no model retraining):** a routing bandit learns as a candidate and is promoted only after beating the live one on a frozen
  eval with no regression (`/v1/policy/gate`, rollback available); corrected answers are remembered for sessions belonging to the same API-key owner; trust in
  OCR vs vision readings is learned from feedback; token allowances are learned per question type.
* **Verification:** an answer counts only if its content words are in the evidence (question words ignored), the page covers the question's
  terms and named entities, and numeric questions get numbers. Several matching records with different values are listed, not guessed.

## Vision, measured on a synthetic benchmark (`vision_bench.py`, `vision_eval.py`; labelled cases, dev = tuned on, test = untouched)
| Task | Result |
|---|---|
| Handwriting-style crops (dates, amounts, names, places), *synthetic handwriting fonts*: PP-OCR vs the vision model | OCR 25% / 25% (dev / test) vs vision 87.5% / 93.8% (dates 10/10, amounts 10/10, names 7/10) |
| **Real** handwriting crops from two scanned forms (20, labelled by eye): PP-OCR vs the vision model | **OCR 13/20 exact, mean similarity 0.96; vision 10/20, 0.79**: the synthetic result did not carry over |
| Real 'Label: value' handwriting rows (8): vision reading the whole row vs only the value part | whole row: mean similarity 0.48 (it describes the image or repeats the label); **value only (label cropped off, 12% margin): 0.91**; OCR: 0.98. Used for the vision cross-check |
| Which checkbox is ticked: vision model naming the option (2 prompts, must agree) | 8% / 0%: near-useless, half the answers "uncertain" and the rest confidently wrong |
| ... vision model transcribing the row with tick marks | 0% / 0% |
| ... vision model, one yes/no question per option | 17% / 8% (answers only 28% of rows) |
| ... **model-free reader** (ink component next to each option label) | **83% / 83%**, right on 97% / 95% of the rows it commits to |
| Picture questions (colour / count / left-right of shapes) | 78% (colour 15/15, count 11/15, side 9/15); voting over scales and a mirrored image did not help |
The synthetic handwriting fonts are *harder for OCR* than the real ink on these forms (neat block letters), so the vision advantage measured on them is not real. The vision model reads short isolated tokens well
(dates, single words) but drifts on longer lines that include the printed label, so the pipeline keeps OCR as the primary reader and uses vision only as a cross-check. On the real scanned forms the model-free checkbox reader got only **1 of 6** rows (two confidently
wrong), because real option boundaries, skew and large hand-drawn ticks break its assumptions, so it stays experimental. TrOCR was also tried and rejected.

## Measured results (read the caveats)
| Test | Result |
|---|---|
| End-to-end through the API (59 checks: files, questions, vision, extraction, budgets, learning, 4 parallel users, abuse) | **59/59** |
| Photo questions (colour, count people, describe) | 4/4 |
| Synthetic loan forms, untouched final set (text / scan-only) | 100% / 96% |
| Real UCB pages, 30 questions (tuned on them) | 33% at first -> 70.0% -> **76.7%** after option-row reading (+3 near-misses) |
| Real UCB held-out pages, 22 questions | **81.8%** (72.7% before the title and fuzzy-word fixes; those were found on this set, so it is no longer untouched) (earlier 63.6% before fixing a cross-document vocabulary bug; all "not in the document" cases correct) |
| Bangla loan form (12 questions, English and Bangla) | born-digital **8/12** (was 2/12 before the Bijoy conversion); degraded **scan 5/12** (was unreadable) |
| Text-only profile on the same real sets, for comparison | 70% / 68.2% |
| **Varied public forms** (`build_varied.py`, `eval_varied.py`: 30 CORD receipts, 12 invoices, 24 noisy scanned FUNSD forms; 238 questions, text profile) | first run **18.5%**: 183 answers were a wrong "not found", because the abstain check treated generic words ("amount", "number", "name") and form wording ("tax" vs VAT, "subtotal") as missing from the page. After a generic-word list and a form-synonym table: **80.3%** (receipts 70.3%, invoices 91.7%, FUNSD 81.3%). Caveat: the fix was found on this set, so it is not an untouched score; the real UCB sets did not change (70.0% / 72.7%). Remaining receipt misses are mostly misread totals, flagged unverified |
| **Fresh varied forms, never tuned on** (`FRESH=1 python build_varied.py`: other splits, 30 receipts, 12 invoices, 24 FUNSD forms, 250 questions) | baseline before the row reader **76.4%** (receipts 72.2%, invoices 95.8%, FUNSD 64.8%) -> **79.2%** (receipts 77.8%, invoices 95.8%, FUNSD 67.0%). On the first varied set the same changes give 84.5% (receipts 80.2%). Gains come from reading labelled rows without the model (`omni/amounts.py`: total / subtotal / tax / price of an item, typo-tolerant labels, tendered amounts and misread rows rejected; on the fresh set it answered 51 of 90 receipt questions, 45 correct) and from fewer wrong "not found" answers (glued or dropped OCR letters, quoted labels). FUNSD answers are scored against a heuristic label/value pairing, so part of its remaining error is the benchmark's, not the pipeline's |
| **... with the form-label reader** (`omni/labels.py`, same fresh set, still never tuned on before this measurement) | **86.4%** (receipts 77.8%, invoices 95.8%, scanned FUNSD forms 67.0% -> **87.5%**); first varied set 85.7%; real UCB sets unchanged (76.7% / 81.8%). For a question that names a label ('value of FAX', 'the booking branch') the value after that label's colon is read straight from the OCR, using the OCR's own box boundaries to know where it ends; 0 tokens. On the fresh set it answered 73 questions, 69 right before two guards were added (a following label read as the value; one-letter scraps) |
| **Third set, untouched until this one measurement** (`FRESH=2 python build_varied.py`: train splits, no image shared with the other sets; 260 questions) | before this round **80.4%** (receipts 76.8%, invoices 88.9%, FUNSD 77.5%) -> **83.8%** (receipts 77.8%, invoices **100%**, FUNSD 77.5%). This is the number to quote. On the second set, which this round was tuned on, the same changes read receipts 77.8% -> 84.4% and FUNSD 87.5%, so most of the receipt gain and the FUNSD level there did not carry over. Added: invoice summary tables read by column position ('total gross worth' = the Total row's box under the 'Gross worth' header), 'name of the client/seller' read from the field under that label, receipt rows with quantities, item names on their own row and 'PB 1'-style tax labels, and amounts written with a space as the thousands separator |
| **... after the OCR round** (same third set, measured once) | **86.5%** (receipts 79.8%, invoices 100%, FUNSD 83.1%), from 83.8%; real UCB sets unchanged (76.7% / 81.8%) |
| **OCR misreads, against word-level ground truth** (`build_ocr_truth.py`, `ocr_eval.py`: every word of the FUNSD forms and CORD receipts in the benchmark images; third set) | words read exactly 87.3% -> **90.2%** (misread 10.5% -> 8.2%, missed 2.2% -> 1.6%); numbers 86.0% -> **87.4%** (misread 8.9% -> 7.5%). From detection at 960 px instead of 736 and a second recogniser on unsure boxes. Numbers missed on receipts (~13%) are mostly faint or cut-off print |
Differences of one or two questions are noise on sets this small.

## Bangla
Many Bangladeshi PDFs are typeset with a legacy Bijoy font: the text layer is Latin-looking gibberish (`cÖwZôv‡bi bvg` is really `প্রতিষ্ঠানের নাম`) with no Bengali script in it, so
the pipeline used to trust and index nonsense. Now `omni/legacy_bn.py` detects such pages (a set of Bijoy-only glyphs, and never on ordinary English/accented text) and converts
them to Unicode Bengali (character map, pre-base vowel signs moved after their cluster, ে+া -> ো, reph moved in front, URLs/numbers kept). Search normalises Unicode (NFC) and
maps Bengali digits to ASCII (`১.৭` = `1.7`). Word-overlap vetoes are not applied across scripts (English question, Bengali form), and an explicit item number that exists on the page is treated
as grounding; such a question is answered by returning that line as written (0 tokens). Measured on one real bank form: **2/12 -> 8/12**.
**Scanned Bangla** goes through Tesseract 5 with the Bengali model (`python download_models.py bn`; on Windows the installer is unpacked with 7-Zip because it needs administrator rights,
everything stays in `runtime/tesseract`). It is chosen automatically: the Latin OCR is tried first and, when it is unsure of most lines (median confidence 0.67 on Bengali against 0.99 on English pages), the page
is re-read with Tesseract `ben+eng` (accurate model, page-segmentation mode 4, 150 dpi: mean line similarity 0.86 on a degraded scan of the form; upscaling made it much worse). Numbered items are repaired by
sequence (Tesseract reads ১ as ৯, ৫ as 0/6...: `৯.২` -> `১.২`). On the scanned form the same 12 questions score **5/12** (born-digital: 8/12); one more is right in substance but the OCR misspelt one word.
`ocr_lang` can be forced per environment with `OMNI_OCR_LANG=bn|en|auto`.
Limits, honestly: the map covers standard Bijoy letters and the conjuncts seen on that form (unknown glyphs are reported, not hidden; two conjunct readings are ambiguous, e.g. `যন্তপাতি` for যন্ত্রপাতি);
scanned Bangla depends on Tesseract being installed (without it, Bangla scans come back as unreadable text); handwritten Bangla is not supported; the item-number repair does not recover section 5 (read as `0.x`); a 0.6B model is unreliable at reading Bengali and at answering across languages
(it picked item 1.4 instead of 5.4 for the guarantor's TIN, and invented "SAMSIM card" for what must be signed); cross-language answers are never marked verified; only one Bangla form was tested.

## Known limits
* **Checkbox state is not solved on real scans.** The vision model is near-useless at it; the model-free reader is good on synthetic rows (83%) but got 1/6
  on real forms, so it is only offered as an unverified alternative. Next step: make row/option isolation robust on skewed scans.
* Handwriting is read with 1-2 letter errors (the vision model helps on dates and some words, not others). TrOCR handwriting model tested and rejected.
* Option lists: when a form row names the options ("Application Type  Secured [] Unsecured Grid") the row is returned as written, with no model (real set 70.0% -> 76.7%); lists spread over several rows still go through the small model and can be incomplete.
* "Title of this form" is a heuristic (tallest heading-like lines, not Label: value rows); it fails when the page has no real title row (one held-out page) and keeps a bank header when it is as tall as the title.
* Multi-part scanned bundles are slow to search on demand: upload with `read=all` to read them once and cache.
* **ONNX and a deeper model were measured and rejected** (6 threads, Ryzen 5 7530U): Qwen3-0.6B on ONNX Runtime (`onnx_bench.py`) reads the prompt at 113 t/s (int4, 919 MB) or 181 t/s (int8, 618 MB) and writes at 8.4 / 6.4 t/s, against 262 / 61.6 t/s for the 373 MB GGUF on llama.cpp. Qwen3-1.7B (twice the layers, Q8, 1.7 GB) runs at 46 / 16 t/s and scored 86.7% on the real set and 68.2% on the held-out set (0.6B: 76.7% / 81.8%): the same total, 3x slower per question, so more depth does not help consistently. The remaining misses are mostly OCR misreads and missing evidence, not reasoning. Adding untrained layers to the 0.6B would not help, and training was scrapped earlier.
* **Time to answer:** with `read=background` the wait after asking about a just-uploaded scan dropped from 1.8-3.3 s (11.5 s on the first call, which loads the OCR engine) to 0.0-0.8 s in a test that asks 3 s after upload; answers from the label and amount readers need no model call, so a warm question averages 0.3 s on the varied sets (was 0.5 s).
* A second OCR opinion on unsure boxes (English PP-OCRv5 recogniser, keep the more confident reading; `OMNI_REREAD=0` turns it off) fixed 3 of 20 real handwriting crops and lowers word misreads on the ground-truth sets; its disagreements also feed the OCR warnings.
* **OCR warnings.** An answer whose value comes from an OCR box below 0.90 confidence, or where the two recognisers disagree, carries `confidence: "low OCR confidence"`, a `warning` and the competing reading in `alternatives`; the answer text itself is unchanged. Measured: such boxes hold a misread number ~40% of the time (average ~9%); on the third set 11 of 260 answers were flagged and 64% of those were right, against 88% of unflagged ones. But only 4 of the 35 wrong answers were flagged: about half of all OCR misreads are confident ones that no signal catches, so a missing warning is no proof.
* **Follow-up questions.** In a chat, 'and the subtotal?', "what about 'DATE'?", 'invoice date?' or 'What is his phone number?' (after a question naming a person) are rewritten into a standalone question from the previous turn on the same documents, with no model; the response carries `understood_as` so a wrong guess is visible, and small talk, full questions and Bengali questions are left as asked. Measured with `followup_eval.py` (each document's other questions asked as follow-ups after its first): third set **69.6% -> 87.1%** (second set 77.7% -> 91.3%), about 4x faster because the rewritten questions reach the zero-token readers. `OMNI_FOLLOWUP=0` turns it off.
* **Session answer cache.** A question asked again in the same session, on the same documents in the same reading state, is answered from the session's earlier verified answer (0 tokens, `source: "session cache"`). Only word-for-word repeats (case, punctuation and filler words aside; every name and number identical): no similarity matching. Only verified answers; never budget-limited or ambiguous ones; the evidence rows are re-checked on the page before reuse; a 'bad' rating drops the entry; nothing is shared between sessions (that is what user-confirmed memory is for).
* A detection size of 1280 px read slightly more of the benchmark images but garbled two clean typed lines of a real letter, so the default is 960 px (`OMNI_DET_SIDE`).
* **Speed is OCR-bound.** A page takes 0.6 s (receipt) to 2.5 s (dense A4 form) to read cold, of which recognition is ~18 ms per text line. Batch size and thread counts changed nothing (`ocr_speed.py`); dropping the direction classifier saves 17% but changes 2% of the text and would break upside-down scans; two OCR engines in parallel save about 20% on multi-page files. Cached pages and rows answered without the model cost ~0.3-0.5 s per question.
* The full profile is ~691 MB, above the original 500 MB target; `-Profile docs` (411 MB) and `vision` (313 MB) fit.
* Eval sets are small (12-30 questions): one question is 3-8 points.

## Privacy
`data/real/`, `data/photos/two_portraits.png`, `data/cache/` (OCR text), `data/uploads/`, `data/e2e/` contain real or derived customer data. Keep local.
