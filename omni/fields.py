"""Schema-free 'Label: value' understanding learned from the documents themselves (0 tokens, no model).

A label is a phrase that keeps appearing right before a colon across the corpus ("Date of Birth", "TIN").
Frequency, not hand-written rules, separates labels from the values that precede them.
Blocks (separated by a blank line) are the layout unit: a value never runs past the end of its block, except when the
label's block holds no value at all (the value was read as its own box), in which case the next block is the value.
"""
import re
from collections import Counter
from .index import toks

MIN_COUNT = 3
MAX_WORDS = 4


def _blocks(text):
    pos = 0
    for blk in text.split("\n\n"):
        yield pos, blk
        pos += len(blk) + 2


def _trail(seg):
    """Trailing words of a segment, cut at the first punctuation-only word (a dash, bullet...)."""
    out = []
    for w in reversed(seg):
        if not re.search(r"[A-Za-z0-9]", w):
            break
        out.append(w)
    return list(reversed(out))


def _segments(blk):
    """Yield (words_before_colon, colon_index, seg_start) using only words after the previous colon."""
    prev = -1
    for m in re.finditer(r":", blk):
        yield blk[prev + 1:m.start()].split(), m.start(), prev + 1
        prev = m.start()


class Lexicon:
    def __init__(self):
        self.count = Counter()

    def learn(self, text):
        for _, blk in _blocks(text):
            for seg, _, _ in _segments(blk):
                seg = _trail(seg)
                for n in range(1, min(MAX_WORDS, len(seg)) + 1):
                    self.count[" ".join(seg[-n:]).lower()] += 1

    def parse(self, text):
        """-> list of dict(label, value, pos): the longest frequent suffix before each colon is the label."""
        out = []
        blocks = list(_blocks(text))
        for bi, (off, blk) in enumerate(blocks):
            found = []
            for seg, cpos, seg_start in _segments(blk):
                seg = _trail(seg)
                if not seg:
                    continue
                n_words = None
                for n in range(min(MAX_WORDS, len(seg)), 0, -1):
                    if self.count[" ".join(seg[-n:]).lower()] >= MIN_COUNT:
                        n_words = n
                        break
                if n_words is None:
                    continue
                start = blk.rfind(seg[-n_words], seg_start, cpos)
                found.append({"label": " ".join(seg[-n_words:]), "colon": cpos, "start": start})
            for i, f in enumerate(found):
                end = found[i + 1]["start"] if i + 1 < len(found) else len(blk)
                val = re.sub(r"\s+", " ", blk[f["colon"] + 1:end]).strip(" \t-|")
                if not val and i + 1 == len(found) and bi + 1 < len(blocks):
                    nxt = blocks[bi + 1][1]
                    if ":" not in nxt:  # the value was read as its own box
                        val = re.sub(r"\s+", " ", nxt).strip()
                if val:
                    out.append({"label": f["label"], "value": val, "pos": off + f["start"]})
        return out

    def question_has_label(self, question):
        qt = set(toks(question))
        for lab, c in self.count.items():
            if c >= MIN_COUNT and len(lab) > 2:
                lt = toks(lab)
                if lt and all(w in qt for w in lt):
                    return True
        return False
