"""Legacy Bengali (Bijoy / SutonnyMJ) text -> Unicode Bengali.

Many Bangladeshi PDFs are typeset with a Bijoy font: the text layer holds Latin-looking gibberish such as `cÖwZôv‡bi bvg` that is really
`প্রতিষ্ঠানের নাম`. Nothing in the pipeline can use it until it is converted. The map below was checked against a real bank form (see tests).
Unknown legacy glyphs are left as they are and reported, so the caller can see how complete a conversion was.
"""
from __future__ import annotations
import re

LETTERS = {
    "K": "ক", "L": "খ", "M": "গ", "N": "ঘ", "O": "ঙ", "P": "চ", "Q": "ছ", "R": "জ", "S": "ঝ", "T": "ঞ", "U": "ট", "V": "ঠ", "W": "ড", "X": "ঢ", "Y": "ণ",
    "Z": "ত", "_": "থ", "`": "দ", "a": "ধ", "b": "ন", "c": "প", "d": "ফ", "e": "ব", "f": "ভ", "g": "ম", "h": "য", "i": "র", "j": "ল", "k": "শ", "l": "ষ",
    "m": "স", "n": "হ", "o": "ড়", "p": "ঢ়", "q": "য়", "r": "ৎ", "s": "ং", "t": "ঃ", "u": "ঁ", "^": "ব", "’": "থ", "ú": "প", "œ": "ন",
}
VOWELS = {"A": "অ", "B": "ই", "C": "ঈ", "D": "উ", "E": "ঊ", "F": "ঋ", "G": "এ", "H": "ঐ", "I": "ও", "J": "ঔ"}
# consonant-cluster keys: each is one complete unit (may end in a halant so the next consonant joins it)
CLUSTERS = {
    "¤": "ম্", "›": "ন্", "š": "ন্", "¯": "স্", "³": "ক্ত", "µ": "ক্র", "ÿ": "ক্ষ", "Î": "ত্র", "Ë": "ত্ত",
    "Ä": "ঞ্জ", "Ý": "ন্স", "·": "ক্স", "ô": "ষ্ঠ", "Í": "ত", "ó": "ষ্ট",
}
PHALA = {"¨": "্য", "ª": "্র", "Ö": "্র", "«": "্র"}
PRE = {"w": "ি", "†": "ে", "‡": "ে", "ˆ": "ৈ", "‰": "ৈ"}
POST = {"v": "া", "x": "ী", "y": "ু", "z": "ূ", "~": "ূ", "æ": "ূ", "‚": "ূ", "…": "ৃ", "„": "ৃ", "Š": "ৗ"}
REPH = "©"
OTHER = {"|": "।", "Õ": "’"}

# glyphs that only appear in Bijoy text (never in ordinary English/French/German): the detector looks for these
SIGNATURE = set("†‡¨ªÖ¯³µÿÎËÄÝô¤‚…„›šˆ‰©·")


def is_legacy(text: str) -> bool:
    """True when the text is Bijoy-encoded Bengali: no Bengali script, but many Bijoy-only glyphs among the letters."""
    if not text or any("ঀ" <= c <= "৿" for c in text):
        return False
    letters = sum(1 for c in text if c.isalpha())
    sig = sum(1 for c in text if c in SIGNATURE)
    return sig >= 6 and sig / max(letters, 1) >= 0.03


def _tokens(s: str):
    out = []
    for ch in s:
        if ch == "A" and out and False:
            pass
        out.append(ch)
    toks = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "A" and i + 1 < len(s) and s[i + 1] == "v":  # 'Av' = আ
            toks.append(("V", "আ"))
            i += 2
            continue
        if ch in VOWELS:
            toks.append(("V", VOWELS[ch]))
        elif ch in CLUSTERS:
            t = CLUSTERS[ch]
            toks.append(("CH" if t.endswith("্") else "C", t))
        elif ch in LETTERS:
            toks.append(("C" if ch not in "stu" else "POST", LETTERS[ch]))
        elif ch in PHALA:
            toks.append(("PHALA", PHALA[ch]))
        elif ch in PRE:
            toks.append(("PRE", PRE[ch]))
        elif ch in POST:
            toks.append(("POST", POST[ch]))
        elif ch == REPH:
            toks.append(("REPH", "র্"))
        elif ch in OTHER:
            toks.append(("X", OTHER[ch]))
        else:
            toks.append(("X", ch))
        i += 1
    return toks


def _cluster(toks, i):
    """Read one consonant cluster starting at i (consonants joined by halants, with ya/ra-phala): (text, next index)."""
    n = len(toks)
    text = ""
    joined = False
    while i < n:
        k, t = toks[i]
        if k in ("C", "CH", "V") and (text == "" or joined):
            text += t
            joined = k == "CH"
            i += 1
        elif k == "PHALA" and text:
            text += t
            joined = False
            i += 1
        else:
            break
    return text, i


_KEEP = re.compile(r"^[\s\d.,:;/\-()%\[\]|]*$|(www\.|https?:|@|\.com|\.org|\.bd)")


def convert(s: str) -> str:
    """Convert a line. Whole tokens that are plainly Latin (URLs, e-mail addresses, numbers) are kept as they are."""
    parts = re.split(r"(\s+)", s)
    return "".join(p if (not p.strip() or _KEEP.search(p)) else _convert_token(p) for p in parts)


def _convert_token(s: str) -> str:
    """Bijoy -> Unicode. Pre-base vowel signs (ি ে ৈ) are moved after their cluster, ে+া becomes ো, and the reph (র্) goes in front."""
    toks = _tokens(s)
    out = []
    i, n = 0, len(toks)
    while i < n:
        k, t = toks[i]
        if k == "PRE":
            pre = t
            cl, j = _cluster(toks, i + 1)
            if not cl:
                out.append(pre)
                i += 1
                continue
            if j < n and toks[j][0] == "REPH":
                cl = "র্" + cl
                j += 1
            if pre == "ে" and j < n and toks[j][1] == "া":
                pre, j = "ো", j + 1
            elif pre == "ে" and j < n and toks[j][1] == "ৗ":
                pre, j = "ৌ", j + 1
            out.append(cl + pre)
            i = j
        elif k in ("C", "CH", "V"):
            cl, j = _cluster(toks, i)
            if j < n and toks[j][0] == "REPH":
                cl = "র্" + cl
                j += 1
            out.append(cl)
            i = j
        else:
            out.append(t)
            i += 1
    return "".join(out)


def unknown_glyphs(s: str):
    """High glyphs the converter did not recognise (a completeness check)."""
    known = set(LETTERS) | set(VOWELS) | set(CLUSTERS) | set(PHALA) | set(PRE) | set(POST) | {REPH} | set(OTHER)
    return sorted({c for c in s if ord(c) > 127 and c not in known})
