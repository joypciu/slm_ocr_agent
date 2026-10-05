"""The agent: act -> verify -> escalate, under a two-tier budget, with a learned router and confirmed-answer memory."""
from __future__ import annotations
import difflib, json, os, re, time, unicodedata, uuid
from collections import Counter
from .budget import Budget, BudgetExceeded
from .index import BM25, chunk_page, page_unit, toks
from .fields import Lexicon, line_fields
from . import amounts, followup, ingest, improve, labels, subagents as sa
from .llm import LLM

VISUAL_KW = re.compile(r"\b(look|image|photo|picture|logo|signature|stamp|chart|graph|diagram|color|colour|handwrit\w*|checkbox|ticked|checked|shown|see)\b", re.I)
EXTRACT_KW = re.compile(r"\b(extract|list all|all the|every|fields?|table of|summari[sz]e)\b", re.I)
META = re.compile(r"\b(describe|summari[sz]e|summary|overview|what kind|what type|what is this|explain|tell me about|about this)\b", re.I)  # open questions: absence of their words proves nothing
TITLE_Q = re.compile(r"\b(title|heading|name) of (this|the) (form|document|letter|page|file)\b", re.I)
LIST_Q = re.compile(r"\b(which|what) [a-z ]*(types|options|levels|brands|documents|products|facilities|sections|categories)\b|\blist\b", re.I)
DESCRIBE_OR_VISUAL = re.compile(r"\b(describe|what is in|what color|what colour|photo|picture|image|logo|stamp|signature)\b", re.I)
ANSWER_TOKENS = {"lookup": 96, "list": 220, "open": 300, "visual": 120}   # how long an answer each kind of question needs
NUMERIC_Q = re.compile(r"how (many|much|long|old)|amount|salary|fee|rate|period|date|years?|number|percent|total|term", re.I)
NOT_FOUND = re.compile(r"not found in the documents|not found in the pages read so far|cannot find|no information|not (mentioned|provided|stated)", re.I)
OCR_STOP_SCORE = 3.0       # stop OCRing more pages once a page matches this well
SUPPORT_OK = 0.6           # an answer whose words are >=60% present in its evidence counts as verified
CHECKBOX_TRUSTED = False    # flip to True only when a checkbox reader beats chance on the synthetic benchmark (vision_eval.py)
HW_CONF = 0.90              # OCR rows below this confidence are treated as handwriting-like
MAX_PAGES_READ = 4          # the agent reads at most this many candidate pages per question, one at a time
SHORT_PAGE = 3200          # pages up to this many characters are given to the model whole, if the token budget allows
BIG_CORPUS_PAGES = 20      # abstain up front from word statistics only on corpora at least this large
WARMUP_PAGES = 6          # read at least this many scanned pages so repeated "Label:" patterns can be learned
COVER_OK = 0.80            # a page only grounds an answer if it contains >=90% (idf-weighted) of the question terms
NL = chr(10)
OPTION_Q = re.compile(r"\b(?:which|what|list)\b[^?]*?\b([a-z]+) (types?|options?|levels?|brands?|categories|category|kinds?)\b", re.I)
OPTION_SKIP = {"the", "of", "what", "which", "are", "all", "different", "those", "these", "available", "many", "can", "is", "does", "do"}
BOX_GLYPHS = re.compile("[" + chr(0x25a0) + "-" + chr(0x25ff) + chr(0x2610) + "-" + chr(0x2612) + chr(0x2022) + chr(0xb7) + "]")
OCR_FLAG_BELOW = 0.90      # answers read from OCR boxes below this confidence carry a 'please check' warning
STOP_ANS = {"the", "and", "for", "was", "are", "this", "that", "with", "from", "not", "value", "documents", "found"}
CACHE_FILLER = {"please", "can", "could", "would", "you", "tell", "me", "kindly", "the", "a", "an", "show", "give"}
ABSENT_SHARE = 0.30        # abstain when this much of the question's weight is terms that appear nowhere in the documents

SYSTEM = ("You answer questions using ONLY the numbered evidence, which is text read from one document. "
          "Words like 'customer', 'applicant', 'he', 'she', 'the company' refer to the person or organisation the document names. "
          "When asked for a specific value, reply with just that value. When asked which options, types or levels exist, list ALL of them. "
          "Otherwise reply in one or two short sentences and quote values exactly as written. "
          "Answer in the language of the question, quoting terms from the evidence exactly as written (Bengali stays Bengali). "
          "Only if the evidence truly has no answer, reply exactly, in English: Not found in the documents.")


QUESTION_WORDS = {"What", "Which", "Who", "When", "Where", "How", "Why", "Tell", "Give", "List", "Find", "Show"}


def entities(q: str):
    """Multi-word capitalised phrases in the question (people, places, organisations): they must appear as a phrase."""
    return [m for m in re.findall(r"(?<![A-Za-z])[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+)+", q) if m.split()[0] not in QUESTION_WORDS]


def has_entities(ents, text: str) -> bool:
    low = text.lower()
    return all(e.lower() in low for e in ents)


ROLE_WORDS = {"customer", "customers", "applicant", "applicants", "client", "borrower", "person", "company", "he", "she"}  # paraphrases of whoever the document names
# how forms commonly label what a question calls something else (a question word counts as present when any of these is)
FORM_SYNONYMS = {"work": ("employer", "organization", "organisation", "company", "occupation", "profession", "business", "employed"), "designation": ("position", "title", "post", "occupation", "rank", "profession"), "number": ("no", "nr", "num", "nbr", "id"), "tax": ("vat", "gst", "ppn", "pb1", "gross", "net"), "subtotal": ("sub", "total", "net"),
                 "seller": ("from", "vendor", "supplier", "seller", "sold"), "client": ("bill", "buyer", "customer", "to", "billed", "ship"),
                 "total": ("sum", "due", "balance", "grand"), "phone": ("tel", "telephone", "mobile", "cell"), "address": ("addr", "street"),
                 "date": ("dated", "issued"), "invoice": ("inv", "bill"), "id": ("tax", "no", "number"), "worth": ("gross", "net", "total"), "name": ("seller", "client", "from", "to", "bill")}
ROLE_WORDS |= {"show", "shows", "shown", "need", "needs", "needed", "required", "require", "requires", "name", "number", "amount", "price", "value", "figure", "cost", "sum", "worth"}  # names for "a number": a receipt says TOTAL, never "total amount"


def _bn_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return sum(1 for c in letters if "ঀ" <= c <= "৿") / max(len(letters), 1)


def script_mismatch(q: str, text: str) -> bool:
    """The question is in one script (Latin or Bengali) and the document in the other: word overlap says nothing about relevance."""
    return (_bn_ratio(q) >= 0.5) != (_bn_ratio(text) >= 0.5)


ITEM_WORD = re.compile(r"\b(item|items|section|clause|point|serial|row|line|no\.|number)\b|নং|ক্রমিক|ধারা|অনুচ্ছেদ|নম্বর", re.I)


def item_refs(q: str):
    """Explicit item/section numbers in the question ('item 1.7', '5.4 নং', '১.৭ নং ক্রমিকে'): strong, language-independent grounding.
    Only counted when the question also uses an item word, so 'worked there 4.6 years' is not mistaken for an item number."""
    if not ITEM_WORD.search(q):
        return []
    return re.findall(r"(?<![\d.])\d{1,2}\.\d{1,2}(?![\d.])", q.translate(str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")))


_BN_DIGIT_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def _norm_words(t: str) -> str:
    return " ".join(sorted(toks(t)))


def support(answer: str, evidence: str, question: str = "") -> float:
    """Share of the answer's content words found in the evidence. Words merely echoed from the question don't count either way."""
    skip = set(toks(question))
    a = [w for w in toks(answer) if w not in skip]
    if not a:
        return 0.0
    ev = set(toks(evidence))
    return sum(w in ev for w in a) / len(a)


class Workspace:
    def __init__(self):
        self.docs: dict[str, ingest.Doc] = {}
        self.ix = BM25()      # small chunks (evidence for the language model)
        self.px = BM25()      # whole pages (finding the right page)
        self.lex = Lexicon()  # 'Label:' phrases learned from the documents

    def _index_page(self, d, p):
        if p.lines:
            self.ix.add(chunk_page(d, p))
            self.px.add(page_unit(d, p))
            if p.source != "scan-text":  # someone else's OCR is too noisy to learn labels from
                self.lex.learn(p.text)

    def add(self, path: str, name: str | None = None) -> ingest.Doc:
        d = ingest.load(path)
        if d.id in self.docs:
            return self.docs[d.id]
        if name:
            d.name = name
        self.docs[d.id] = d
        for p in d.pages:
            self._index_page(d, p)
        return d

    def ocr(self, doc, n):
        ingest.run_ocr(doc, n)
        self.ix.drop_page(doc.id, n)
        self.px.drop_page(doc.id, n)
        self._index_page(doc, doc.pages[n - 1])

    def scope(self, doc_ids):
        return [self.docs[i] for i in (doc_ids or self.docs) if i in self.docs]

    def fuzzy_scores(self, q, doc_ids):
        """Page relevance that survives OCR misspellings: idf-weighted share of each query word's letter-trigrams found on the page."""
        def tri(t):
            t = " " + re.sub("[^a-z0-9]+", " ", t.lower()) + " "
            return {t[i:i + 3] for i in range(len(t) - 2)}
        words = [w for w in dict.fromkeys(toks(q)) if len(w) >= 4]
        wt = {w: (tri(w), self.px.idf(w)) for w in words}
        cache = self.__dict__.setdefault("_tri_cache", {})
        out = {}
        for d in self.scope(doc_ids):
            for p in d.pages:
                if not p.lines:
                    continue
                key = (d.id, p.n, p.source)
                if key not in cache:
                    cache[key] = tri(p.text)
                pt = cache[key]
                out[(d.id, p.n)] = sum(idf * len(tw & pt) / max(len(tw), 1) for tw, idf in wt.values())
        return out

    def pending_ocr(self, doc_ids, q=None):
        """Pages we have not read ourselves: blank scans, and scans carrying someone else's untrusted OCR layer.
        With a question, the most relevant pages (by their embedded text) come first."""
        pend = [(d, p.n) for d in self.scope(doc_ids) if d.kind in ("pdf", "image") for p in d.pages if p.source in ("none", "scan-text")]
        if q:
            bm = {(c["doc"], c["page"]): sc for sc, c in self.px.search(q, 500, doc_ids or None)}
            fz = self.fuzzy_scores(q, doc_ids)
            top_b, top_f = max(bm.values(), default=1.0) or 1.0, max(fz.values(), default=1.0) or 1.0
            pend.sort(key=lambda dn: -(bm.get((dn[0].id, dn[1]), 0.0) / top_b + fz.get((dn[0].id, dn[1]), 0.0) / top_f))
        return pend

    def should_abstain(self, q, doc_ids):
        """Terms that appear nowhere AND nothing in the documents covers the question (not judged across scripts, or when the question cites an item number that exists)."""
        sample = " ".join(c["text"] for c in self.px.chunks if not doc_ids or c["doc"] in doc_ids)[:3000]
        if script_mismatch(q, sample):
            return False
        if re.search(r"['\"][^'\"]{2,40}['\"]", q) and sum(len(d.pages) for d in self.scope(doc_ids)) < BIG_CORPUS_PAGES:
            return False  # the user quotes a label: answer from the closest row, the grounding check still rejects an unsupported value
        refs = item_refs(q)
        if refs and all(r in set(toks(sample)) for r in refs):
            return False
        return self.absent_share(q, doc_ids) >= ABSENT_SHARE and self.px.best_coverage(q, doc_ids or None) < 0.75

    def retrieve(self, q, doc_ids, k_pages=2, k_chunks=3):
        """Two stages: pick pages that cover the question, then the best chunks inside those pages."""
        ents = entities(q)
        ph = self.px.search(q, k_pages * 4, doc_ids or None)
        ph.sort(key=lambda h: (not has_entities(ents, h[1]["text"]), -h[0]))
        ph = ph[:k_pages]
        if not ph:
            return [], []
        pages = {(c["doc"], c["page"]) for _, c in ph}
        return ph, self.ix.search(q, k_chunks, doc_ids or None, pages)

    def evidence(self, q, doc_ids, k=2, span=420, max_chars=None):
        """Sentence-level evidence: the few rows/sentences that best cover the question, with their neighbours.
        A 256M model answers far more reliably from ~300 focused characters than from pages of text."""
        qt = list(dict.fromkeys(toks(q)))
        out = []
        top = self.px.search(q, 2, doc_ids or None)
        if top and len(top[0][1]["text"]) <= min(SHORT_PAGE, max_chars or SHORT_PAGE):  # a short page fits in the model's context: give it all of it
            sc, page = top[0]
            return [(1.0, {"doc": page["doc"], "name": page["name"], "page": page["page"], "text": page["text"].replace("\n\n", "\n"), "box": page["box"]})]
        for sc, page in top:
            units = []
            for blk in page["text"].split("\n\n"):
                units += [u.strip() for u in re.split(r"(?<=[.!?])\s+", blk) if u.strip()]
            scored = sorted(((self.px.coverage(qt, Counter(toks(u))) + 0.001 * min(len(u), 300), i) for i, u in enumerate(units)), reverse=True)
            used = set()
            for s_, i in scored[:k]:
                if s_ < 0.05 or i in used:
                    continue
                lo, hi = max(0, i - 1), min(len(units), i + 2)
                used.update(range(lo, hi))
                out.append((s_ + sc / 100, {"doc": page["doc"], "name": page["name"], "page": page["page"], "text": " | ".join(units[lo:hi])[:span], "box": page["box"]}))
        return sorted(out, key=lambda t: -t[0])[:3]

    def rank_pages(self, q, doc_ids, k=4):
        """Pages by keyword score plus OCR-noise-tolerant letter-trigram score; pages holding the named entity first."""
        pages = {(c["doc"], c["page"]): c for c in self.px.chunks if not doc_ids or c["doc"] in doc_ids}
        bm = {(c["doc"], c["page"]): sc for sc, c in self.px.search(q, 500, doc_ids or None)}
        fz = self.fuzzy_scores(q, doc_ids)
        tb, tf = max(bm.values(), default=1.0) or 1.0, max(fz.values(), default=1.0) or 1.0
        ents = entities(q)
        keyed = sorted(pages, key=lambda key: (not has_entities(ents, pages[key]["text"]), -(bm.get(key, 0.0) / tb + fz.get(key, 0.0) / tf)))
        return [pages[key] for key in keyed[:k]]

    def page_evidence(self, q, page, max_chars, k=2, span=420):
        """The whole page if the token budget allows it, otherwise the rows/sentences of this page that best cover the question."""
        if len(page["text"]) <= min(SHORT_PAGE, max_chars):
            return [(1.0, {**page, "text": page["text"].replace("\n\n", "\n")})]
        qt = list(dict.fromkeys(toks(q)))
        units = []
        for blk in page["text"].split("\n\n"):
            units += [u.strip() for u in re.split(r"(?<=[.!?])\s+", blk) if u.strip()]
        scored = sorted(((self.px.coverage(qt, Counter(toks(u))) + 0.001 * min(len(u), 300), i) for i, u in enumerate(units)), reverse=True)
        out, used = [], set()
        for s_, i in scored[:k]:
            if s_ < 0.05 or i in used:
                continue
            lo, hi = max(0, i - 1), min(len(units), i + 2)
            used.update(range(lo, hi))
            out.append((s_, {**page, "text": " | ".join(units[lo:hi])[:span]}))
        return out

    def entity_page_exists(self, q, doc_ids):
        ents = entities(q)
        if not ents:
            return True
        return any(has_entities(ents, c["text"]) for c in self.px.chunks if not doc_ids or c["doc"] in doc_ids)

    def unseen_in(self, q, doc_ids):
        """Question terms with no counterpart in the documents being asked about (word forms allowed). Other uploaded documents do not count."""
        pref, real, blob = set(), set(), []
        for i, c in enumerate(self.px.chunks):
            if not doc_ids or c["doc"] in doc_ids:
                blob.append(c["text"].lower())
                for t in self.px.tf[i]:
                    real.add(t)
                    pref.add(t)
                    pref.update((t[:3], t[:4], t[:5]))
        blob = " ".join(blob)
        out = []
        for w in dict.fromkeys(toks(q)):
            if len(w) >= 4 and w in blob:
                continue  # glued to its neighbour by the OCR ('1993DEPARTMENT')
            if w in pref or any(s in pref for s in FORM_SYNONYMS.get(w, ())):
                continue
            if len(w) > 4 and (w.endswith("ed") or w.endswith("ing")):
                continue  # verbs get paraphrased; their absence proves nothing
            st = self.px.stem(w)
            if st in pref or st[:5] in pref or (st[:4] in pref and len(st) <= 5):
                continue
            if len(w) >= 5 and any(abs(len(t) - len(w)) <= 2 and difflib.SequenceMatcher(None, w, t).ratio() >= 0.82 for t in real):
                continue  # OCR misspelt it (or the form spells it a little differently)
            out.append(w)
        return out

    def absent_share(self, q, doc_ids=None):
        """Share of the question's weight carried by terms that appear nowhere in the documents being asked about."""
        qt = [w for w in dict.fromkeys(toks(q)) if w not in ROLE_WORDS]
        if not qt or not self.px.chunks:
            return 0.0
        tot = sum(self.px.idf(w) for w in qt)
        return sum(self.px.idf(w) for w in self.unseen_in(q, doc_ids) if w not in ROLE_WORDS) / tot if tot else 0.0


class Agent:
    def __init__(self, ws: Workspace, llm: LLM, explore=0.0, shared=None):
        """`shared` = (live_router, candidate_router, memory) so sessions of one API-key owner share learning."""
        self.ws, self.llm, self.explore = ws, llm, explore
        if shared:
            self.live, self.cand, self.memory = shared[:3]
            self.trust, self.stats = (shared[3], shared[4]) if len(shared) >= 5 else (improve.Trust(), improve.TokenStats())
        else:
            self.live, self.cand, self.memory = improve.load_live(), improve.load_candidate(), improve.Memory()
            self.trust, self.stats = improve.Trust(), improve.TokenStats()
        self.requests: dict[str, dict] = {}
        self.answers: dict[tuple, dict] = {}   # this session's verified answers (see _cache_key)
        self.last_turn: dict[tuple, str] = {}  # previous question per document scope, for follow-ups ('and the subtotal?')

    # ------------------------------------------------------------------ token planning
    @staticmethod
    def q_class(q):
        if DESCRIBE_OR_VISUAL.search(q):
            return "visual"
        if META.search(q) or EXTRACT_KW.search(q):
            return "open"
        if LIST_Q.search(q):
            return "list"
        return "lookup"

    # ------------------------------------------------------------------ features / routing
    def features(self, q, doc_ids, b: Budget):
        docs = self.ws.scope(doc_ids)
        pages = [p for d in docs for p in d.pages] or [None]
        need = sum(1 for p in pages if p is not None and p.source in ("none", "scan-text")) / len(pages)
        hits = self.ws.px.search(q, 1, doc_ids or None)
        return [1.0, need, 1.0 if VISUAL_KW.search(q) else 0.0, 1.0 if EXTRACT_KW.search(q) else 0.0,
                min(len(q) / 120, 1.0), min((hits[0][0] if hits else 0) / 10, 1.0),
                1.0 if b.can("vlm_looks") else 0.0, b.left("llm_tokens") / max(b.user_limits["llm_tokens"], 1),
                1.0 if self.ws.lex.question_has_label(q) and hits else 0.0]

    def _feasible(self, action, doc_ids, b: Budget, q=""):
        if action == "ocr":
            return bool(self.ws.pending_ocr(doc_ids)) and b.can("ocr_pages")
        if action == "look":
            return bool(DESCRIBE_OR_VISUAL.search(q)) and self.llm.vision and b.can("vlm_looks") and any(d.kind in ("pdf", "image") for d in self.ws.scope(doc_ids))
        if action == "fields":
            return bool(self.ws.lex.count)
        return True

    # ------------------------------------------------------------------ strategies
    def _touch(self, q, doc_ids, b):
        """Re-read with our own OCR the scanned pages this question is about (never more than the top two)."""
        done = []
        for d, n in [dn for dn in self.ws.pending_ocr(doc_ids, q) if dn[0].pages[dn[1] - 1].source == "scan-text"][:2]:
            pg = d.pages[n - 1]
            if pg.source == "scan-text" and b.can("ocr_pages"):
                b.spend("ocr_pages", 1, f"re-read {d.name} p.{pg.n}")
                self.ws.ocr(d, pg.n)
                done.append(f"re-read {d.name} p.{pg.n} with our OCR (embedded text layer is untrusted)")
        return done

    def s_fields(self, q, doc_ids, b, _max):
        """Deterministic 'Label: value' lookup. Costs zero LLM tokens; returns None when it cannot answer."""
        b.spend("tool_calls", 1, "field lookup")
        self._pre = self._touch(q, doc_ids, b)
        ph, _ = self.ws.retrieve(q, doc_ids, k_pages=3)
        ents = entities(q)
        ent_words = set(toks(" ".join(ents)))
        qt = set(toks(q)) - ent_words            # a name word must never help a label match
        rare = [w for w in dict.fromkeys(toks(q)) if 0 < self.ws.px.df.get(w, 0) <= 4]
        found = []                                # (label_words_matched, page_rank, field, page)
        for rank, (_, page) in enumerate(ph):
            text = page["text"]
            if self.ws.px.coverage(list(dict.fromkeys(toks(q))), Counter(toks(text))) < COVER_OK or not has_entities(ents, text):
                continue                          # this page is not about the entity asked for
            low = text.lower()
            anchors = [a for a in ([low.find(e.lower()) for e in ents] or [low.find(w) for w in rare]) if a >= 0]
            anchor = min(anchors) if anchors else -1
            cands, best_k = [], 0
            for f in self.ws.lex.parse(text):
                lt = toks(f["label"])
                for k in range(len(lt), 0, -1):
                    if (k >= 2 or len(lt) == 1) and all(w in qt for w in lt[-k:]):
                        if k > best_k:
                            best_k, cands = k, []
                        if k == best_k:
                            cands.append(f)
                        break
            if not cands:
                continue
            after = [f for f in cands if f["pos"] >= anchor] or cands
            found.append((best_k, rank, min(after, key=lambda f: f["pos"]), page))
        if not found:
            return None
        top_k = max(k for k, *_ in found)
        group = sorted((t for t in found if t[0] == top_k), key=lambda t: t[1])
        distinct = {}
        for _, _, f, page in group:
            distinct.setdefault(re.sub(r"\s+", "", f["value"].lower()), (f, page))
        if len(distinct) > 1:                     # several different people/records match: never pick silently
            parts = [f"{f['value'][:60]} ({pg['name']} p.{pg['page']})" for f, pg in list(distinct.values())[:3]]
            evs = [{"doc": pg["doc"], "name": pg["name"], "page": pg["page"], "box": pg["box"], "text": f"{f['label']}: {f['value'][:60]}"} for f, pg in list(distinct.values())[:3]]
            return {"answer": "Multiple matches: " + "; ".join(parts), "evidence": evs, "support": 1.0, "ambiguous": True,
                    "steps": getattr(self, "_pre", []) + [f"{len(distinct)} different values match '{' '.join(ents) or 'the question'}': reported all, did not guess"]}
        f, page = next(iter(distinct.values()))
        val = f["value"][:80].strip()
        ev = {"doc": page["doc"], "name": page["name"], "page": page["page"], "box": page["box"], "text": f"{f['label']}: {val}"}
        return {"answer": val, "evidence": [ev], "support": 1.0, "steps": getattr(self, "_pre", []) + [f"field '{f['label']}' on {page['name']} p.{page['page']} (0 tokens)"]}

    def _answer(self, q, hits, b, max_tokens):
        ev = "\n".join(f"[{i + 1}] {c['text']}" for i, (_, c) in enumerate(hits))  # no file names: the model would echo them
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Evidence:\n{ev}\n\nQuestion: {q}"}]
        return self.llm.chat(msgs, b, max_tokens=max_tokens, why="answer"), ev

    def s_title(self, q, doc_ids):
        """'What is the title of this form?': the tallest lines in the top half of the first page."""
        for d in self.ws.scope(doc_ids):
            lines = [(t, bx) for t, bx in d.pages[0].lines if bx[1] < 0.6 and len(t) >= 3]
            heads = [(t, bx) for t, bx in lines if ":" not in t and re.search(r"[A-Za-z]{3}", t) and not re.match(r"\s*\d", t)]
            lines = heads or lines  # a title is not a 'Label: value' line, a numbered row or something low on the page
            if lines:
                tall = max(bx[3] - bx[1] for _, bx in lines)
                top = [(t, bx) for t, bx in lines if bx[3] - bx[1] >= 0.75 * tall]
                title = " ".join(t for t, _ in sorted(top, key=lambda tb: tb[1][1]))[:80]
                ev = {"doc": d.id, "name": d.name, "page": 1, "box": (0, 0, 1, 0.6), "text": title}
                return {"answer": title, "evidence": [ev], "support": 1.0, "steps": ["title = the tallest heading lines on page 1 (0 tokens)"]}
        return None

    def _label_value(self, q, doc_ids):
        """The question names a form label ('value of FAX', 'the booking branch'): read what follows that label's colon, 0 tokens."""
        for page in self.ws.rank_pages(q, doc_ids, k=1):
            pg = self.ws.docs[page["doc"]].pages[page["page"] - 1]
            hit = labels.read(q, pg.lines, getattr(pg, "segs", None))
            if hit:
                val, row = hit
                ev = {"doc": page["doc"], "name": page["name"], "page": page["page"], "box": page["box"], "text": row}
                return {"answer": val, "evidence": [ev], "support": 1.0, "steps": [f"value after the label in '{row[:80]}' (0 tokens)"]}
        return None

    def _amount_row(self, q, doc_ids):
        """'What is the total / subtotal / tax?' and 'What is the price of X?' on receipts and invoices: read the labelled row, 0 tokens."""
        if not (amounts.SIMPLE_Q.match(q) or amounts.PRICE_Q.match(q) or amounts.COLUMN_Q.match(q)):
            return None
        for page in self.ws.rank_pages(q, doc_ids, k=1):
            pg = self.ws.docs[page["doc"]].pages[page["page"] - 1]
            hit = amounts.column_total(q, pg.lines, getattr(pg, "segs", None)) or amounts.answer(q, page["text"])
            if hit:
                val, row = hit
                ev = {"doc": page["doc"], "name": page["name"], "page": page["page"], "box": page["box"], "text": row}
                return {"answer": val, "evidence": [ev], "support": 1.0, "steps": [f"labelled row '{row}' (0 tokens)"]}
        return None

    def _option_row(self, q, doc_ids):
        """'What are the application types?': a form row such as 'Application Type  Secured [] Unsecured Grid' already lists every option, and a small model drops some.
        Returns the rest of that row (box glyphs removed) when the row clearly starts with the named kind of thing."""
        m = OPTION_Q.search(q)
        if not m or m.group(1).lower() in OPTION_SKIP:
            return None
        noun = m.group(2).lower()
        noun = "category" if noun.startswith("categor") else noun.rstrip("s")
        key = m.group(1).lower() + " " + noun
        found = {}
        for page in self.ws.rank_pages(q, doc_ids, k=2):
            for ln in page["text"].split(NL):
                i = ln.lower().find(key)
                if i < 0 or i > 6:
                    continue
                rest = BOX_GLYPHS.sub(" ", ln[i + len(key):]).strip(" :-|")
                rest = " ".join(rest.split())
                if len(rest.split()) >= 2 and not rest.startswith("("):
                    found.setdefault(rest, (rest, {"doc": page["doc"], "name": page["name"], "page": page["page"], "box": page["box"], "text": ln.strip()}))
        return next(iter(found.values())) if len(found) == 1 else None

    def s_text(self, q, doc_ids, b, max_tokens):
        b.spend("tool_calls", 1, "search")
        if TITLE_Q.search(q):
            t = self.s_title(q, doc_ids)
            if t:
                return t
        pre = self._touch(q, doc_ids, b)
        amt = self._amount_row(q, doc_ids) or self._label_value(q, doc_ids)
        if amt:
            amt["steps"] = list(pre) + amt["steps"]
            return amt
        row = self._option_row(q, doc_ids)
        if row:
            return {"answer": row[0], "evidence": [row[1]], "support": 1.0, "steps": list(pre) + ["options = the rest of the form row that names them, as written (0 tokens)"]}
        ents = entities(q)
        qtok = [w for w in dict.fromkeys(toks(q)) if w not in ROLE_WORDS]
        best, steps, tried_pages = None, list(pre), 0
        for page in self.ws.rank_pages(q, doc_ids, k=MAX_PAGES_READ):
            room = int(max(600, (b.left("llm_tokens") - 300) * 3.0))  # characters of evidence the remaining token budget can afford
            hits = self.ws.page_evidence(q, page, room)
            if not hits:
                continue
            try:
                ans, ev = self._answer(q, hits, b, max_tokens)
            except BudgetExceeded:
                if best is None:
                    raise
                steps.append("token budget exhausted while reading further pages")
                break
            tried_pages += 1
            ptoks = set(toks(page["text"]))
            refs = item_refs(q)
            refs_ok = bool(refs) and all(r in ptoks for r in refs)     # 'item 1.7' is on this page: grounded whatever the language
            cross = script_mismatch(q, page["text"])                    # question and page in different scripts: no word-overlap veto
            grounded = bool(META.search(q)) or refs_ok or cross or (self.ws.px.coverage(qtok, Counter(toks(page["text"]))) >= COVER_OK and has_entities(ents, page["text"]))
            sup = 0.0 if NOT_FOUND.search(ans) or not grounded else support(ans, ev, q)
            if cross and not refs_ok:
                sup = min(sup, 0.55)  # an answer across languages cannot be checked against the text: shown, but not marked verified
                # ...but a NUMBER in the answer can be checked in any language: if the model states a figure that is not on the page, it invented it
                # (a lone digit proves nothing: OCR noise leaves stray digits everywhere, so only percentages and figures of 2+ digits are checked)
                fig = r"\d[\d,.]*%?"
                stated = [n for n in re.findall(fig, unicodedata.normalize("NFC", ans).translate(_BN_DIGIT_MAP)) if n.endswith("%") or len(re.sub(r"\D", "", n)) >= 2]
                page_nums = set(re.findall(fig, unicodedata.normalize("NFC", page["text"]).translate(_BN_DIGIT_MAP)))
                qnums = set(re.findall(fig, q.translate(_BN_DIGIT_MAP)))
                if any(n.rstrip(",.") not in {x.rstrip(",.") for x in page_nums | qnums} for n in stated):
                    ans, sup = "Not found in the documents.", 0.0
            if NUMERIC_Q.search(q) and not any(ch.isdigit() for ch in ans):
                sup = 0.0  # a question about an amount/date/count needs an answer with a number in it
            cand = {"answer": ans, "evidence": [c for _, c in hits], "support": sup}
            if best is None or sup > best["support"] or (best["support"] == 0.0 and NOT_FOUND.search(best["answer"]) and not NOT_FOUND.search(ans)):
                best = cand
            if sup >= SUPPORT_OK:
                break
        if best is None and META.search(q):  # an open question: start from the beginning of the documents
            first = [c for d in self.ws.scope(doc_ids) for c in self.ws.ix.chunks if c["doc"] == d.id][:3]
            if first:
                ans, ev = self._answer(q, [(0.0, c) for c in first], b, max_tokens)
                best = {"answer": ans, "evidence": first, "support": support(ans, ev)}
        if best is None:
            return {"answer": "Not found in the documents.", "evidence": [], "support": 0.0, "steps": steps + ["search: no pages"]}
        best["steps"] = steps + [f"read {tried_pages} page(s) in rank order"]
        return best

    def s_ocr(self, q, doc_ids, b, max_tokens):
        steps = []
        for d, n in self.ws.pending_ocr(doc_ids, q):
            if not b.can("ocr_pages"):
                steps.append("ocr budget exhausted")
                break
            b.spend("ocr_pages", 1, f"ocr {d.name} p.{n}")
            self.ws.ocr(d, n)
            steps.append(f"ocr {d.name} p.{n}")
            done = sum(1 for dd in self.ws.scope(doc_ids) for pp in dd.pages if pp.source == "ocr")
            if self.ws.px.best_coverage(q, doc_ids or None) >= COVER_OK and self.ws.entity_page_exists(q, doc_ids) and (done >= WARMUP_PAGES or self.ws.lex.question_has_label(q)):
                break
        refs = item_refs(q)
        hit = self._item_lookup(refs, doc_ids) if refs else None  # now that the page has been read, "item 1.3" may be a line we can return as written
        if hit:
            ev = [{"doc": d.id, "name": d.name, "page": pn, "box": box, "text": ln} for ln, d, pn, box in hit]
            return {"answer": " | ".join(ln for ln, *_ in hit), "evidence": ev, "support": 1.0,
                    "steps": steps + [f"read the pages, then found the line for item {', '.join(refs)}: returned as written (0 tokens)"]}
        r = (self.s_fields(q, doc_ids, b, max_tokens) if self.ws.lex.question_has_label(q) else None) or self.s_text(q, doc_ids, b, max_tokens)
        r["steps"] = steps + r["steps"]
        return r

    def s_look(self, q, doc_ids, b, max_tokens):
        b.spend("vlm_looks", 1, "look")
        hits = self.ws.px.search(q, 1, doc_ids or None)
        if hits:
            c = hits[0][1]
            doc, n = self.ws.docs[c["doc"]], c["page"]
        else:
            doc = next(d for d in self.ws.scope(doc_ids) if d.kind in ("pdf", "image"))
            n = 1
        page = doc.pages[n - 1]
        hint = page.text[:500] if page.lines else ""
        img = ingest.render(doc, n)
        content = [self.llm.image_part(img), {"type": "text", "text": (f"Text read from this page: {hint}\n" if hint else "") + f"Question: {q}\nAnswer briefly using only what is visible."}]
        ans = self.llm.chat([{"role": "user", "content": content}], b, max_tokens=max_tokens, why="look", vision=True)
        ev = [{"doc": doc.id, "name": doc.name, "page": n, "text": hint, "box": (0, 0, 1, 1)}]
        return {"answer": ans, "evidence": ev, "support": 0.3 if not hint else min(0.55, support(ans, hint)), "steps": [f"look {doc.name} p.{n}"]}

    @staticmethod
    def _afford(b, need):
        """A sub-agent needs `need` tokens: the agent raises its OWN allowance for this question, never above the user's ceiling."""
        if b.left("llm_tokens") < need:
            b.throttle("llm_tokens", min(b.user_limits["llm_tokens"], b.spent["llm_tokens"] + need), "sub-agent needs room: agent raises its own allowance")

    # ------------------------------------------------------------------ sub-agent dispatch
    def _photo_route(self, q, doc_ids, b, rid, t0):
        """Pictures with little or no text are answered by the vision sub-agent straight from the pixels."""
        docs = self.ws.scope(doc_ids)
        if not docs or not all(d.kind == "image" for d in docs) or not self.llm.vision or not b.can("vlm_looks"):
            return None
        d = docs[0]
        steps = []
        if d.pages[0].source in ("none", "scan-text") and b.can("ocr_pages"):  # see first whether the picture holds any text (cheap, cached)
            b.spend("ocr_pages", 1, f"ocr {d.name}")
            self.ws.ocr(d, 1)
            steps.append(f"ocr {d.name}: is there text in this picture?")
        if len(d.pages[0].text) >= 60 and not DESCRIBE_OR_VISUAL.search(q):
            return None  # a photographed document: the text pipeline handles it
        try:
            self._afford(b, 700)
            ans = sa.describe_image(self.llm, b, ingest.render(d, 1), q)
        except BudgetExceeded as e:
            b.log.append({"op": "skip", "why": f"photo route: {e}"})
            return None
        steps.append("vision sub-agent looked at the picture (answers about pixels can't be checked against text)")
        self.requests[rid] = {"x": None, "action": "photo", "q": q, "shas": [d.sha], "answer": ans, "reading": None}
        return {"id": rid, "answer": ans, "source": "computed", "strategy": "photo", "tried": ["photo"], "support": 0.5, "verified": False,
                "evidence": [{"name": d.name, "page": 1, "box": (0, 0, 1, 1), "text": d.pages[0].text[:200]}], "steps": steps,
                "budget": b.snapshot(), "seconds": round(time.time() - t0, 1)}

    def _ocr_check(self, q, best):
        """Flag an answer whose value was read from an unsure part of a scan, and show the competing reading.
        Measured on FUNSD/CORD word ground truth: numbers in boxes below 0.90 confidence are misread ~32% of the time (average ~9%), and the
        two recognisers disagreeing marks ~30%; about half of all misreads are confident ones that no signal catches, so a missing flag is no proof."""
        if not best.get("evidence") or NOT_FOUND.search(best.get("answer", "")):
            return best
        ev0 = best["evidence"][0]
        doc = self.ws.docs.get(ev0.get("doc"))
        if doc is None or not ev0.get("page") or doc.kind not in ("pdf", "image"):
            return best
        page = doc.pages[ev0["page"] - 1]
        if page.source != "ocr" or not getattr(page, "segs", None):
            return best
        qset = set(re.sub(r"[^a-z0-9]", " ", q.lower()).split())
        vals = [re.sub(r"[^a-z0-9]", "", t) for t in re.split(r"\s+", best["answer"].lower())]
        vals = [v for v in vals if (len(v) >= 2 and any(c.isdigit() for c in v)) or (len(v) >= 3 and v not in qset and v not in STOP_ANS)]
        if not vals:
            return best
        low, alts = 1.0, []
        for parts in page.segs:
            for p in parts or []:
                key = re.sub(r"[^a-z0-9]", "", p[0].lower())
                if any(v in key for v in vals):
                    if len(p) > 3:
                        low = min(low, p[3])
                    if len(p) > 4 and p[4]:
                        alts.append(p[4])
        if low >= OCR_FLAG_BELOW and not alts:
            return best
        note = f"read from a part of the scan the OCR was unsure of (confidence {low:.2f})" if low < OCR_FLAG_BELOW else "the two OCR readers disagree on this part of the scan"
        out = {**best, "confidence": best.get("confidence") or "low OCR confidence", "warning": note + (f"; it may also read: {alts[0]!r}" if alts else "") + ". Please check the document.",
               "steps": list(best.get("steps", [])) + [f"OCR check: {note}"]}
        if alts:
            out["alternatives"] = list(best.get("alternatives", [])) + [{"source": "second OCR reader", "value": alts[0]}]
        return out

    def _refine(self, q, best, doc_ids, b):
        """Cross-check a scanned/handwritten answer with the vision sub-agent, only where it can help:
        dates/amounts/numbers that OCR may have misread, and which-box-is-ticked questions. Never on typed or born-digital text."""
        if not best.get("evidence") or best.get("strategy") not in ("fields", "text", "ocr"):
            return best
        ev0 = best["evidence"][0]
        doc = self.ws.docs.get(ev0.get("doc"))
        if doc is None or doc.kind not in ("pdf", "image") or not ev0.get("page"):
            return best
        page = doc.pages[ev0["page"] - 1]
        if page.source not in ("ocr", "scan-text") or not page.lines or NOT_FOUND.search(best["answer"]):
            return best
        kind = sa.kind_of(q)
        ans = best["answer"]
        steps = list(best.get("steps", []))
        try:
            if sa.CHECK_Q.search(q):
                # Which box is ticked: measured, a 256M vision model is close to useless at this (8-17% on synthetic rows). A model-free reader
                # (ink in the square next to each option) is right on 95% of the synthetic rows it commits to, but only 1 of 6 real scanned rows,
                # so its reading is offered as an alternative and never replaces the answer.
                ri = sa.find_row_idx(page, "", q)
                if ri:
                    res = sa.read_checkbox_row(sa.crop(doc, page.n, ri[2], pad=0.008), q)
                    if res["confident"]:
                        steps.append(f"experimental checkbox reader (image processing) sees '{res['option']}' ticked; not verified on real scans")
                        return {**best, "support": min(best["support"], 0.5), "confidence": "unverified", "steps": steps,
                                "alternatives": [{"source": "pixels", "values": [res["option"]], "note": "experimental"}]}
                    steps.append(f"checkbox reader could not decide ({res['detail']}): the OCR text is shown, box state unknown")
                    return {**best, "support": min(best["support"], 0.5), "confidence": "uncertain", "steps": steps}
                return best
            if not self.llm.vision:
                return best
            if kind in ("date", "amount", "number") and b.can("vlm_looks"):
                self._afford(b, 500)
                ocr_val = sa.extract_value(kind, ans)
                ri = sa.find_row_idx(page, ocr_val or "", q)
                row = (ri[1], ri[2]) if ri else None
                if row:
                    # OCR is weak on handwriting (measured: 25% vs the vision model's ~90% on handwriting-style crops), and its own
                    # confidence tells them apart: on low-confidence rows the learned trust starts out leaning towards vision
                    hw = sa.row_confidence(page, ri[0]) < HW_CONF
                    kind_key = kind + ("_hw" if hw else "")
                    seen = sa.read_crop(self.llm, b, sa.value_crop(sa.crop(doc, page.n, row[1]), row[0]))
                    v_val = sa.extract_value(kind, seen)
                    reading = {"kind": kind_key, "ocr": ocr_val, "vision": v_val, "used": "ocr"}
                    if v_val and ocr_val and sa.same_value(kind, v_val, ocr_val):
                        steps.append(f"OCR and the vision sub-agent read the same {kind}: {ocr_val}")
                        return {**best, "support": max(best["support"], 0.9), "agreement": True, "steps": steps, "reading": reading}
                    if v_val and not ocr_val:
                        steps.append(f"OCR gave no valid {kind}; vision sub-agent read {v_val}")
                        reading["used"] = "vision"
                        return {**best, "answer": v_val, "support": max(best["support"], 0.6), "refined_by": "vision", "steps": steps, "reading": reading}
                    if v_val and ocr_val:  # they disagree: learned trust decides, the other reading is shown
                        pick = self.trust.prefer(kind_key)
                        chosen, other = (ocr_val, v_val) if pick == "ocr" else (v_val, ocr_val)
                        reading["used"] = pick
                        steps.append(f"OCR read {ocr_val!r}, vision sub-agent read {v_val!r}: using the {pick} reading (learned trust)")
                        return {**best, "answer": chosen, "confidence": "disputed", "steps": steps, "reading": reading,
                                "alternatives": [{"source": "vision" if pick == "ocr" else "ocr", "value": other}]}
                    best = {**best, "reading": reading}
            elif kind == "word" and b.can("vlm_looks"):
                # names/places/other words: if the row the answer came from was read with low confidence, offer the vision reading too
                ri = sa.find_row_idx(page, ans, q)
                if ri and sa.row_confidence(page, ri[0]) < HW_CONF:
                    self._afford(b, 500)
                    seen = sa.read_crop(self.llm, b, sa.value_crop(sa.crop(doc, page.n, ri[2]), ri[1]))
                    if seen and _norm_words(seen) != _norm_words(sa.value_of(ri[1])):
                        steps.append(f"the answer comes from a low-confidence (handwriting-like) row; the vision sub-agent reads it as {seen[:60]!r}")
                        best = {**best, "steps": steps, "confidence": "low-ocr-confidence", "alternatives": [{"source": "vision", "value": seen[:80]}]}
        except BudgetExceeded as e:
            b.log.append({"op": "skip", "why": f"vision cross-check: {e}"})
            steps.append(f"vision cross-check skipped: {e}")
            best = {**best, "steps": steps}
        return best

    def _item_lookup(self, refs, doc_ids):
        """'What does item 1.7 ask for?': the page has a line that starts with that item number, so return that line. Exact, free, language-independent."""
        found = {}
        for d in self.ws.scope(doc_ids):
            for p in d.pages:
                for t, box in p.lines:
                    for ln in t.split("\n"):
                        norm_ln = unicodedata.normalize("NFC", ln).translate(_BN_DIGIT_MAP).strip()
                        for ref in refs:
                            m = re.match(r"^" + re.escape(ref) + r"(?![\d.])[\s.:)-]*(.+)$", norm_ln)
                            if m and len(m.group(1)) > 1:
                                found.setdefault(ref, []).append((ln.strip(), d, p.n, box))
        if len(found) != len(set(refs)) or any(len({x[0] for x in v}) > 1 for v in found.values()):
            return None  # some number has no such line, or several different lines claim it: do not guess
        return [v[0] for v in found.values()]

    # ------------------------------------------------------------------ main entry
    def ask(self, question: str, b: Budget, doc_ids=None, mode="auto") -> dict:
        rid = uuid.uuid4().hex[:10]
        t0 = time.time()
        asked = question
        prev = self.last_turn.get(tuple(sorted(doc_ids or [])))
        if mode == "auto" and os.environ.get("OMNI_FOLLOWUP", "1") == "1":
            question, how = followup.resolve(question, prev)
        else:
            how = None
        self.last_turn[tuple(sorted(doc_ids or []))] = question
        if how:
            out = self.ask(question, b, doc_ids, mode="auto_resolved")
            out["understood_as"] = question
            out["asked"] = asked
            out["steps"] = [how] + out.get("steps", [])
            return out
        if mode == "auto_resolved":
            mode = "auto"
        shas = sorted(d.sha for d in self.ws.scope(doc_ids))
        mem = self.memory.recall(question, shas)
        if mem and mode == "auto":
            return {"id": rid, "answer": mem["a"], "source": "memory", "strategy": "memory", "support": 1.0, "evidence": [],
                    "steps": ["recalled a user-confirmed answer (0 tokens)"], "budget": b.snapshot(), "seconds": 0.0}
        if mode == "extract":
            return self.extract(question, b, doc_ids, rid)
        ckey = self._cache_key(question, doc_ids)
        hit = self.answers.get(ckey) if mode == "auto" else None
        if hit:
            if self._evidence_still_there(hit["ev"]):
                self.requests[rid] = dict(hit["req"])
                return {**hit["res"], "id": rid, "source": "session cache", "budget": b.snapshot(), "seconds": round(time.time() - t0, 2),
                        "steps": hit["res"]["steps"] + ["same question on the same documents and pages earlier in this session; its evidence re-checked (0 tokens)"]}
            self.answers.pop(ckey, None)
        # word statistics are only trustworthy on a big enough corpus; on a small one, answer first and verify afterwards
        big = sum(len(d.pages) for d in self.ws.scope(doc_ids)) >= BIG_CORPUS_PAGES
        if big and not META.search(question) and not self.ws.pending_ocr(doc_ids) and self.ws.should_abstain(question, doc_ids):
            missing = self.ws.unseen_in(question, doc_ids)
            return {"id": rid, "answer": "Not found in the documents.", "source": "abstain", "strategy": "abstain", "support": 1.0, "verified": True,
                    "evidence": [], "steps": [f"these question terms appear nowhere in the documents: {', '.join(missing)}"], "budget": b.snapshot(), "seconds": round(time.time() - t0, 1)}
        refs = item_refs(question)
        if refs:
            hit = self._item_lookup(refs, doc_ids)
            if hit:
                ev = [{"doc": d.id, "name": d.name, "page": pn, "box": box, "text": ln} for ln, d, pn, box in hit]
                ans = " | ".join(ln for ln, *_ in hit)
                self.requests[rid] = {"x": None, "action": "item", "q": question, "shas": shas, "answer": ans, "reading": None}
                return {"id": rid, "answer": ans, "source": "computed", "strategy": "item", "tried": ["item"], "support": 1.0, "verified": True, "evidence": ev,
                        "steps": [f"the document has a line for item {', '.join(refs)}: returned as written (0 tokens)"], "budget": b.snapshot(), "seconds": round(time.time() - t0, 1)}
        cls = self.q_class(question)
        tokens_before = b.spent["llm_tokens"]
        # the agent sets its own token allowance from what this kind of question has cost before (never above the user's ceiling);
        # it is for THIS question, so it sits on top of what the session has already spent (budgets accumulate across a session)
        b.throttle("llm_tokens", b.spent["llm_tokens"] + self.stats.allowance(cls, b.user_limits["llm_tokens"]), f"{cls} question: allowance learned from past spend")
        photo = self._photo_route(question, doc_ids, b, rid, t0)
        if photo:
            self.stats.record(cls, b.spent["llm_tokens"] - tokens_before)
            return photo
        x = self.features(question, doc_ids, b)
        order, scores = self.live.rank(x, self.explore)
        order = [a for a in order if self._feasible(a, doc_ids, b, question)] or ["text"]
        mt = ANSWER_TOKENS[cls]
        best, tried, note = None, [], None
        for action in order[:4]:
            r, stop = None, False
            for attempt in (0, 1):
                try:
                    r = getattr(self, "s_" + action)(question, doc_ids, b, mt)
                    break
                except BudgetExceeded as e:
                    extra = max(e.need - e.left, 100) if e.resource == "llm_tokens" else 1
                    if attempt == 0 and b.request_extension(e.resource, extra, f"strategy '{action}' needs more {e.resource}"):
                        continue  # granted (auto or by the user): retry the same strategy
                    note = f"stopped: {e} (the agent asked for more; approve at /v1/sessions/<id>/budget/resolve)"
                    stop = True
                    break
            if stop:
                break
            if r is None:
                continue
            tried.append(action)
            r["strategy"] = action
            if best is None or r["support"] > best["support"]:
                best = r
            if r["support"] >= SUPPORT_OK:
                break
            # not verified: give itself room to try a costlier strategy (still <= the user's ceiling)
            b.throttle("llm_tokens", b.user_limits["llm_tokens"], "answer unverified: escalate")
        if best is None or (best["support"] < SUPPORT_OK and not self.ws.pending_ocr(doc_ids) and NOT_FOUND.search(best["answer"])):
            best = best or {"answer": "Not found in the documents." if not note else "I ran out of budget before I could answer.", "evidence": [], "support": 0.0, "steps": [], "strategy": "none"}
        left = len(self.ws.pending_ocr(doc_ids))
        if best["support"] < SUPPORT_OK and not left and not META.search(question) and self.ws.should_abstain(question, doc_ids):
            missing = self.ws.unseen_in(question, doc_ids)
            best = {"answer": "Not found in the documents.", "evidence": [], "support": 1.0, "strategy": "abstain",
                    "steps": best["steps"] + [f"after reading every page, these question terms appear nowhere: {', '.join(missing)}"]}
        if best["support"] < SUPPORT_OK and left and not best["answer"].startswith("Not found"):
            best["steps"] = best["steps"] + [f"unverified: {left} pages are still unread (OCR budget)"]
        if best["support"] < SUPPORT_OK and left and best["strategy"] in ("text", "ocr", "fields", "none"):
            best["answer"] = f"Not found in the pages read so far ({left} pages still unread; raise the OCR budget to search them)."
        best = self._refine(question, best, doc_ids, b)
        best = self._ocr_check(question, best)
        used = best["strategy"]
        self.stats.record(cls, b.spent["llm_tokens"] - tokens_before)
        self.requests[rid] = {"x": x, "action": used, "q": question, "shas": shas, "answer": best["answer"], "reading": best.get("reading")}
        if used in improve.ACTIONS:  # weak self-supervised signal; real feedback is stronger
            self.cand.update(used, x, best["support"] - 0.05 * len(tried), lr=0.03)
        res = {"id": rid, "answer": best["answer"], "source": "computed", "strategy": used, "tried": tried, "support": round(best["support"], 2),
               "verified": best["support"] >= SUPPORT_OK, "evidence": [{k: e[k] for k in ("name", "page", "box", "text")} for e in best["evidence"][:3]],
               "steps": best["steps"], "budget": b.snapshot(), "seconds": round(time.time() - t0, 1)}
        if best.get("ambiguous"):
            res["ambiguous"] = True
        for k in ("alternatives", "refined_by", "agreement", "confidence", "warning"):
            if k in best:
                res[k] = best[k]
        if note:
            res["budget_note"] = note
        if b.pending:
            res["pending_requests"] = [p for p in b.pending if p["status"] == "waiting"]
        improve.log_trace({"id": rid, "q": question, "strategy": used, "tried": tried, "support": res["support"], "x": x, "answer": best["answer"], "spent": b.snapshot()})
        if res["verified"] and not note and not left and not res.get("ambiguous"):
            self.requests[rid]["ckey"] = ckey
            self.answers[ckey] = {"res": dict(res), "req": dict(self.requests[rid]), "ev": [dict(e) for e in best["evidence"][:3]]}
        return res

    # ------------------------------------------------------------------ session answer cache
    def _cache_key(self, q, doc_ids):
        """A repeat is the same question word for word (case, punctuation and filler words aside; every name and number identical) on the same
        documents in the same reading state (a page OCR'd since then changes the key). No similarity matching: 'loan amount' must never answer 'loan term'."""
        t = unicodedata.normalize("NFC", q.lower()).replace("what's", "what is")
        words = tuple(w for w in re.findall(r"\w+", t) if w not in CACHE_FILLER)
        state = tuple((d.sha, tuple(p.source for p in d.pages)) for d in sorted(self.ws.scope(doc_ids), key=lambda d: d.sha))
        return words, state

    def _evidence_still_there(self, evidence):
        """Before an answer is reused, every row it was based on must still be on its page as stored."""
        for e in evidence:
            d = self.ws.docs.get(e.get("doc"))
            if d is None:
                return False
            if e.get("page") and e.get("text"):
                page_text = " ".join(d.pages[e["page"] - 1].text.split())
                if " ".join(str(e["text"]).split())[:200] not in page_text:
                    return False
        return True

    # ------------------------------------------------------------------ extraction ("extract anything")
    def extract(self, question, b, doc_ids, rid, targets=None):
        docs = self.ws.scope(doc_ids)
        out, steps = [], []
        if not targets:  # schema-free: every learned 'Label: value' pair, zero tokens
            budget_note = None
            for doc, number in self.ws.pending_ocr(doc_ids):
                try:
                    b.spend("seconds", 0, "field extraction")
                    for resource in ("ocr_pages", "tool_calls"):
                        if not b.can(resource, 1):
                            raise BudgetExceeded(resource, 1, b.left(resource))
                    b.spend("ocr_pages", 1, "read page for field extraction")
                    b.spend("tool_calls", 1, "OCR for field extraction")
                except BudgetExceeded as exc:
                    b.request_extension(exc.resource, max(exc.need - exc.left, 1), "Read remaining pages for field extraction")
                    budget_note = "Some pages remain unread. Approve more resources to extract their fields."
                    break
                self.ws.ocr(doc, number)
            for d in docs:
                for p in d.pages:
                    direct = line_fields(p.text)
                    direct_labels = {f["label"].lower() for f in direct}
                    fields = direct + [f for f in self.ws.lex.parse(p.text) if f["label"].lower() not in direct_labels]
                    for f in fields:
                        out.append({"field": f["label"], "value": f["value"][:120], "doc": d.name, "doc_id": d.id, "page": p.n, "verified": p.source == "text"})
            steps.append(f"learned-label pairing: {len(out)} label:value pairs (0 tokens)")
            result = {"id": rid, "mode": "extract", "fields": out, "steps": steps, "budget": b.snapshot()}
            if budget_note:
                result["budget_note"] = budget_note
            return result
        schema = {"type": "object", "properties": {t: {"type": ["string", "null"]} for t in targets}, "required": list(targets)}
        ctx = []
        for t in targets:
            b.spend("tool_calls", 1, f"search {t}")
            ctx += [c for _, c in self.ws.retrieve(t, doc_ids)[1]]
        seen, uniq = set(), []
        for c in ctx:
            key = (c["doc"], c["page"], c["text"][:40])
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        ev = "\n".join(f"[{c['name']} p.{c['page']}] {c['text']}" for c in uniq[:6])[:2400]
        msgs = [{"role": "system", "content": "Extract the requested fields from the evidence. Use null when a field is not present. Copy values exactly."},
                {"role": "user", "content": f"Evidence:\n{ev}\n\nFields: {', '.join(targets)}"}]
        try:
            raw = self.llm.chat(msgs, b, max_tokens=300, schema=schema, why="extract")
        except BudgetExceeded as e:
            b.request_extension(e.resource, max(e.need - e.left, 100), "targeted extraction needs more tokens")
            return {"id": rid, "mode": "extract", "fields": [], "steps": steps + ["stopped: token budget too small for this extraction"],
                    "budget_note": f"stopped: {e} (the agent asked for more; approve at /v1/sessions/<id>/budget/resolve)", "budget": b.snapshot(),
                    "pending_requests": [p for p in b.pending if p["status"] == "waiting"]}
        try:
            vals = json.loads(raw)
        except Exception:
            vals = {}
        for t in targets:
            v = vals.get(t)
            src = next((c for c in uniq if v and support(str(v), c["text"]) >= 0.8), None)
            grounded_text = bool(src and self.ws.docs[src["doc"]].pages[src["page"] - 1].source == "text")
            out.append({"field": t, "value": v, "verified": grounded_text, "doc": src["name"] if src else None, "doc_id": src["doc"] if src else None, "page": src["page"] if src else None, "box": src["box"] if src else None})
        return {"id": rid, "mode": "extract", "fields": out, "steps": steps + ["schema-constrained extraction"], "budget": b.snapshot()}

    # ------------------------------------------------------------------ user feedback -> learning
    def feedback(self, rid, verdict: str, correction: str | None = None):
        rec = self.requests.get(rid)
        if not rec:
            return {"ok": False, "error": "unknown request id"}
        if verdict != "good":
            self.answers.pop(rec.get("ckey"), None)  # an answer the user says is wrong is never served again from the cache
        reading = rec.get("reading")
        if verdict == "good":
            self.memory.remember(rec["q"], rec["answer"], rec["shas"], verified=True)
            if rec["action"] in improve.ACTIONS:
                self.cand.update(rec["action"], rec["x"], 1.0, lr=0.15)
            if reading:  # the reading that was used was right; a source that disagreed with it was wrong
                self.trust.update(reading["kind"], reading["used"], True)
                other = "vision" if reading["used"] == "ocr" else "ocr"
                if reading.get(other) and not sa.same_value(reading["kind"], reading.get(other) or "", reading.get(reading["used"]) or ""):
                    self.trust.update(reading["kind"], other, False)
        else:
            if rec["action"] in improve.ACTIONS:
                self.cand.update(rec["action"], rec["x"], -1.0, lr=0.15)
            if correction:
                self.memory.remember(rec["q"], correction, rec["shas"], verified=True, corrected=True)
                if reading:  # whichever source matches the user's correction was right
                    for src in ("ocr", "vision"):
                        if reading.get(src):
                            self.trust.update(reading["kind"], src, sa.same_value(reading["kind"], reading[src], correction))
        improve.save_candidate(self.cand)
        return {"ok": True, "candidate_updates": self.cand.n_updates, "remembered": verdict == "good" or bool(correction)}
