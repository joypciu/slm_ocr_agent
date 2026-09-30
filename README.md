# Omni Agent

A CPU-only document and picture agent. It reads PDFs, scans, photos, Word, CSV and text files; answers questions; extracts fields;
uses specialist sub-agents automatically; lets the user and the agent share control of tokens and other resources; and improves from
feedback without retraining any model. Runs fully offline.

## Run
```
.\start.ps1                    # full: Qwen3-0.6B reasoner + SmolVLM-256M vision specialist + OCR   (~691 MB of weights)
.\start.ps1 -Profile docs      # reasoner + OCR only, no picture understanding                     (~411 MB, under 500)
.\start.ps1 -Profile vision    # one SmolVLM model for everything, weakest on documents            (~313 MB, under 500)
```
API on :8090 (key from env `OMNI_API_KEYS`, default `dev-key`). Flow: `POST /v1/sessions` (budget) -> `/documents?read=lazy|all` -> `/ask` ->
`/feedback`. Also `/budget`, `/budget/resolve`, `/budget/limit`, `/v1/policy`, `/v1/policy/gate`, `/v1/policy/rollback`, `/health`.
Tests: `python e2e_test.py` (58 checks through the API), `run_full_eval.py`, `run_holdout.py`, `api_test.py`.

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
  eval with no regression (`/v1/policy/gate`, rollback available); corrected answers are remembered and shared across users; trust in
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
| End-to-end through the API (58 checks: files, questions, vision, extraction, budgets, learning, 4 parallel users, abuse) | **57/58** |
| Photo questions (colour, count people, describe) | 4/4 |
| Synthetic loan forms, untouched final set (text / scan-only) | 100% / 96% |
| Real UCB pages, 30 questions (tuned on them) | 33% at first -> **70.0%** (+3 near-misses) |
| Real UCB held-out pages, 22 questions | **72.7%** (was 63.6% before fixing a cross-document vocabulary bug; all "not in the document" cases correct) |
| Bangla loan form (12 questions, English and Bangla) | **8/12**, was 2/12 before the Bijoy conversion (details below) |
| Text-only profile on the same real sets, for comparison | 70% / 68.2% |
Differences of one or two questions are noise on sets this small.

## Bangla
Many Bangladeshi PDFs are typeset with a legacy Bijoy font: the text layer is Latin-looking gibberish (`cÖwZôv‡bi bvg` is really `প্রতিষ্ঠানের নাম`) with no Bengali script in it, so
the pipeline used to trust and index nonsense. Now `omni/legacy_bn.py` detects such pages (a set of Bijoy-only glyphs, and never on ordinary English/accented text) and converts
them to Unicode Bengali (character map, pre-base vowel signs moved after their cluster, ে+া -> ো, reph moved in front, URLs/numbers kept). Search normalises Unicode (NFC) and
maps Bengali digits to ASCII (`১.৭` = `1.7`). Word-overlap vetoes are not applied across scripts (English question, Bengali form), and an explicit item number that exists on the page is treated
as grounding; such a question is answered by returning that line as written (0 tokens). Measured on one real bank form: **2/12 -> 8/12**.
Limits, honestly: the map covers standard Bijoy letters and the conjuncts seen on that form (unknown glyphs are reported, not hidden; two conjunct readings are ambiguous, e.g. `যন্তপাতি` for যন্ত্রপাতি);
**scanned Bangla is not supported** (RapidOCR has no Bengali recogniser; Tesseract `ben` would be the route); a 0.6B model is unreliable at reading Bengali and at answering across languages
(it picked item 1.4 instead of 5.4 for the guarantor's TIN, and invented "SAMSIM card" for what must be signed); cross-language answers are never marked verified; only one Bangla form was tested.

## Known limits
* **Checkbox state is not solved on real scans.** The vision model is near-useless at it; the model-free reader is good on synthetic rows (83%) but got 1/6
  on real forms, so it is only offered as an unverified alternative. Next step: make row/option isolation robust on skewed scans.
* Handwriting is read with 1-2 letter errors (the vision model helps on dates and some words, not others). TrOCR handwriting model tested and rejected.
* Long option lists can come back incomplete (small model). "Title of this form" is a heuristic that fails with logos or handwriting on the top row.
* Multi-part scanned bundles are slow to search on demand: upload with `read=all` to read them once and cache.
* The full profile is ~691 MB, above the original 500 MB target; `-Profile docs` (411 MB) and `vision` (313 MB) fit.
* Eval sets are small (12-30 questions): one question is 3-8 points.

## Privacy
`data/real/`, `data/photos/two_portraits.png`, `data/cache/` (OCR text), `data/uploads/`, `data/e2e/` contain real or derived customer data. Keep local.
