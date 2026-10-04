"""Turns a chat follow-up into a standalone question using the previous turn, without a model:
   'and the subtotal?' after 'What is the total amount?'       -> 'What is the subtotal?'
   "what about 'DATE'?" after "What is the value of 'FAX'?"    -> "What is the value of 'DATE'?"
   'invoice date?'                                             -> 'What is the invoice date?'
   'What is his phone number?' after "What is Karim Rahman's address?" -> "What is Karim Rahman's phone number?"
Anything else is left exactly as asked. The rewrite is shown to the user, so a wrong guess is visible."""
from __future__ import annotations
import re

LEAD = re.compile(r"^\s*(?:ok(?:ay)?,?\s*|then\s+|so\s+)?(?:and|also|what about|how about|and what about|and how about)\s+(.+?)\s*\??\s*$", re.I)
QWORD = re.compile(r"^\s*(what|which|who|whom|whose|when|where|why|how|is|are|was|were|do|does|did|can|could|would|will|should|list|show|give|tell|"
                   r"describe|summari[sz]e|extract|find|explain|compare|please)\b", re.I)
PRON = re.compile(r"\b(his|her|their|its)\b", re.I)
FRAME = re.compile(r"^\s*((?:what|which|who|when|where|how much|how many)\s+(?:is|are|was|were)\s+)", re.I)
NAME = re.compile(r"(?<![A-Za-z])[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+)+")
SMALL_TALK = {"thanks", "thank you", "thank you so much", "thx", "ok", "okay", "hi", "hello", "bye", "yes", "no", "great", "cool", "nice", "good", "fine", "sure", "got it"}
QUESTION_WORDS = {"What", "Which", "Who", "When", "Where", "Why", "How", "Is", "Are", "Was", "The"}


def _names(q):
    return [m.rstrip("'’s").rstrip("'’") for m in NAME.findall(q) if m.split()[0] not in QUESTION_WORDS]


def resolve(q, prev):
    """-> (standalone question, note) ; note is None when the question was left as asked."""
    if not prev or re.search(r"[ঀ-৿]", q + prev):  # no previous turn, or Bengali: leave it alone
        return q, None
    m = LEAD.match(q)
    if m:
        body = m.group(1).strip()
    elif not QWORD.match(q) and len(q.split()) <= 5 and not q.strip().endswith("."):
        body = q.strip().rstrip("?").strip()          # 'invoice date?', 'the tax?'
    else:
        names = _names(prev)
        if names and PRON.search(q) and not _names(q):  # 'What is his phone number?' after a question about a named person
            new = PRON.sub(f"{names[0]}'s", q, count=1)
            return new, f"'{PRON.search(q).group(0)}' read as {names[0]}: {new}"
        return q, None
    if not body or not re.search(r"[A-Za-z0-9]", body) or body.lower().strip(" !.,") in SMALL_TALK:
        return q, None
    pm = FRAME.match(prev)
    frame = pm.group(1) if pm else "What is "
    frame = frame[0].upper() + frame[1:]
    if re.match(r"^['\"‘“].+['\"’”]$", body) and re.search(r"\bvalue (?:of|for)\b", prev, re.I):
        new = f"What is the value of {body}?"
    else:
        names = _names(prev)
        if PRON.match(body):
            body = PRON.sub(f"{names[0]}'s" if names else "the", body, count=1)
        elif not re.match(r"^(the|a|an|this|that|these|those)\b", body, re.I) and not NAME.match(body) and not re.match(r"^['\"\d]", body):
            body = "the " + body
        new = f"{frame}{body}?"
    return new, f"follow-up read as: {new}"
