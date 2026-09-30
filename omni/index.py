"""BM25 keyword search (no embedding model, no GPU, no torch) with two levels and coverage-aware ranking.

Level 1 ranks whole pages (so a name on one line and a value on another still belong together).
Level 2 ranks small chunks inside the chosen pages to build compact evidence for the language model.
Ranking multiplies BM25 by idf-weighted *coverage* of the query terms, so a short chunk that repeats one term
cannot beat a chunk that contains all of them.
"""
import math, re
from collections import Counter

STOP = set("a an the of to in on for and or is are was were be by with at as from that this it its what which who how when where do does did i you your please tell me about there here has have had can could would should been being also than then into their them they he she his her him many much long allowed required needed used given made written stated mentioned chosen sent taken held known form document page file letter".split())


def toks(s):
    """Lower-case word tokens. Edge punctuation is stripped ('female.' == 'female') but inner dots survive ('4.6', 'e&g' stays split)."""
    out = []
    for w in re.findall(r"[a-z0-9ঀ-৿%$.@-]+", s.lower()):
        w = w.strip(".-@")
        if len(w) > 1 and w not in STOP:
            out.append(w)
    return out


def chunk_page(doc, page, target=450):
    out, cur, box, n = [], [], None, 0

    def flush():
        out.append({"doc": doc.id, "name": doc.name, "page": page.n, "text": " ".join(cur), "box": box})

    for t, b in page.lines:
        cur.append(t)
        n += len(t)
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
        if n >= target:
            flush()
            cur, box, n = [], None, 0
    if cur:
        flush()
    return out


def page_unit(doc, page):
    if not page.lines:
        return []
    xs = [b for _, b in page.lines]
    box = (min(b[0] for b in xs), min(b[1] for b in xs), max(b[2] for b in xs), max(b[3] for b in xs))
    return [{"doc": doc.id, "name": doc.name, "page": page.n, "text": "\n\n".join(t for t, _ in page.lines), "box": box}]


class BM25:
    def __init__(self, k1=1.4, b=0.75):
        self.k1, self.b = k1, b
        self.chunks, self.tf, self.len, self.df = [], [], [], Counter()

    def add(self, chunks):
        for c in chunks:
            t = Counter(toks(c["text"]))
            self.chunks.append(c)
            self.tf.append(t)
            self.len.append(sum(t.values()))
            self.df.update(t.keys())

    def drop_page(self, doc_id, page):
        keep = [i for i, c in enumerate(self.chunks) if not (c["doc"] == doc_id and c["page"] == page)]
        old = (self.chunks, self.tf, self.len)
        self.chunks, self.tf, self.len, self.df = [], [], [], Counter()
        for i in keep:
            self.chunks.append(old[0][i])
            self.tf.append(old[1][i])
            self.len.append(old[2][i])
            self.df.update(old[1][i].keys())

    def idf(self, w):
        N = max(len(self.chunks), 1)
        return math.log(1 + (N - self.df.get(w, 0) + .5) / (self.df.get(w, 0) + .5))

    @staticmethod
    def stem(w):
        for suf in ("ing", "ed", "es", "ly", "er", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= (3 if suf == "s" else 4):
                return w[:-len(suf)]
        return w

    def _prefixes(self):
        if getattr(self, "_pref_n", -1) != len(self.chunks):
            self._pref = {t[:k] for tf in self.tf for t in tf for k in (3, 4, 5)} | {t for tf in self.tf for t in tf}
            self._pref_n = len(self.chunks)
        return self._pref

    def unseen(self, q):
        """Query terms with no counterpart in the documents. Word forms match ('signed' ~ 'signature', 'brands' ~ 'brand')."""
        pref = self._prefixes()
        out = []
        for w in dict.fromkeys(toks(q)):
            if self.df.get(w, 0):
                continue
            if len(w) > 4 and (w.endswith("ed") or w.endswith("ing")):
                continue  # verbs get paraphrased ('signed' vs 'signature'); their absence proves nothing
            s = self.stem(w)
            if s in pref or s[:5] in pref or (s[:4] in pref and len(s) <= 5):
                continue
            out.append(w)
        return out

    @staticmethod
    def _covered(w, tf):
        """Exact match, or a word-form match (same first 4 letters for longer words: applicant ~ applying, brands ~ brand)."""
        if w in tf:
            return True
        return len(w) >= 5 and any(len(t) >= 4 and t[:4] == w[:4] for t in tf)

    def coverage(self, qt, tf):
        tot = sum(self.idf(w) for w in qt) or 1.0
        return sum(self.idf(w) for w in qt if self._covered(w, tf)) / tot

    def search(self, q, k=4, doc_ids=None, pages=None):
        qt = list(dict.fromkeys(toks(q)))
        N = max(len(self.chunks), 1)
        avg = (sum(self.len) / N) or 1
        res = []
        for i, c in enumerate(self.chunks):
            if doc_ids and c["doc"] not in doc_ids:
                continue
            if pages is not None and (c["doc"], c["page"]) not in pages:
                continue
            s = 0.0
            for w in qt:
                f = self.tf[i].get(w, 0)
                if f:
                    s += self.idf(w) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / avg))
            if s > 0:
                res.append((s * (0.25 + 0.75 * self.coverage(qt, self.tf[i])), c))
        return sorted(res, key=lambda x: -x[0])[:k]

    def best_coverage(self, q, doc_ids=None):
        """Highest idf-weighted share of the question's terms found together in a single indexed unit."""
        qt = list(dict.fromkeys(toks(q)))
        best = 0.0
        for i, c in enumerate(self.chunks):
            if doc_ids and c["doc"] not in doc_ids:
                continue
            best = max(best, self.coverage(qt, self.tf[i]))
        return best
