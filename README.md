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
                                On rows the OCR itself is unsure about (handwriting) the learned trust leans towards vision
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
| Handwriting-style crops (dates, amounts, names, places): PP-OCR vs the vision model | OCR 25% / 25% (dev / test) vs **vision 87.5% / 93.8%** (dates 10/10, amounts 10/10, names 7/10) |
| Which checkbox is ticked: vision model naming the option (2 prompts, must agree) | 8% / 0%: near-useless, half the answers "uncertain" and the rest confidently wrong |
| ... vision model transcribing the row with tick marks | 0% / 0% |
| ... vision model, one yes/no question per option | 17% / 8% (answers only 28% of rows) |
| ... **model-free reader** (ink component next to each option label) | **83% / 83%**, right on 97% / 95% of the rows it commits to |
| Picture questions (colour / count / left-right of shapes) | 78% (colour 15/15, count 11/15, side 9/15); voting over scales and a mirrored image did not help |
The synthetic handwriting is cleaner than real ink. On your real scanned forms the model-free checkbox reader got only **1 of 6** rows (two confidently
wrong), because real option boundaries, skew and large hand-drawn ticks break its assumptions, so it stays experimental. TrOCR was also tried and rejected.

## Measured results (read the caveats)
| Test | Result |
|---|---|
| End-to-end through the API (58 checks: files, questions, vision, extraction, budgets, learning, 4 parallel users, abuse) | **57/58** |
| Photo questions (colour, count people, describe) | 4/4 |
| Synthetic loan forms, untouched final set (text / scan-only) | 100% / 96% |
| Real UCB pages, 30 questions (tuned on them) | 33% at first -> **66.7%** (+4 near-misses) |
| Real UCB held-out pages, 22 questions | **63.6%** (5/5 "not in the document" correct) |
| Text-only profile on the same real sets, for comparison | 70% / 68.2% |
Differences of one or two questions are noise on sets this small.

## Known limits
* **Checkbox state is not solved on real scans.** The vision model is near-useless at it; the model-free reader is good on synthetic rows (83%) but got 1/6
  on real forms, so it is only offered as an unverified alternative. Next step: make row/option isolation robust on skewed scans.
* Handwriting is read with 1-2 letter errors (the vision model helps on dates and some words, not others). TrOCR handwriting model tested and rejected.
* Long option lists can come back incomplete (small model). "Title of this form" is a heuristic that fails with logos or handwriting on the top row.
* Multi-part scanned bundles are slow to search on demand: upload with `read=all` to read them once and cache.
* The full profile is ~691 MB, above the original 500 MB target; `-Profile docs` (411 MB) and `vision` (313 MB) fit.
* Bangla untested. Eval sets are small.

## Privacy
`data/real/`, `data/photos/two_portraits.png`, `data/cache/` (OCR text), `data/uploads/`, `data/e2e/` contain real or derived customer data. Keep local.
