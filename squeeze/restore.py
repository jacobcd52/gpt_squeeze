"""Put missing spaces back into squeezed text, separately for glued numbers and fused words.

Uses the same tokenization and number rule as score.py. Words are restored more liberally
than the score counts (two-word fusions, case changes, punctuation run into a word), as in
the ARC report. Code spans, paths and identifiers are left untouched.
"""
import re
from functools import lru_cache

from wordfreq import zipf_frequency

from squeeze.score import UNITS, _CODE_BLOCK, _INLINE, _PIECE, _TOKEN_SKIP, _VOCAB, _is_word, segment


def _num_glued(prev, p, nxt):
    """Mirror of score.py's glued-number rule. Returns (space_before, space_after)."""
    before = after = False
    if prev[:1].isalpha():
        pl = prev.lower()
        long_num = len(p) >= 3 or "." in p or ":" in p
        if long_num and len(pl) >= 2 and not (len(pl) <= 2 and pl in UNITS):
            before = True
        elif len(p) >= 2 and len(pl) >= 3 and _is_word(pl):
            before = True
        elif len(pl) >= 3 and nxt[:1].isalpha():
            before = True
    if nxt[:1].isalpha():
        nl = nxt.lower()
        if not (nl in UNITS or nl.rstrip("s") in UNITS):
            unit = next((u for u in sorted(UNITS, key=len, reverse=True)
                         if len(u) >= 2 and nl.startswith(u) and _is_word(nl[len(u):]) and len(nl) - len(u) >= 2), None)
            if unit or len(nl) >= 3:
                after = True
    return before, after


def _split_case(run):
    """Split at lower->ACRONYM and ACRONYM->lower boundaries (needRECOMPILE, CSVhas); keep CamelCase (NumPy)."""
    parts = re.findall(r"[A-Z]{2,}(?=[A-Z][a-z]|[a-z]|$)|[A-Z]?[a-z]+|[A-Z]+", run)
    if "".join(parts) != run:
        return [run]
    merged = []
    for x in parts:
        if merged and x[:1].isupper() and x[1:].islower() and merged[-1][-1:].islower():
            merged[-1] += x  # CamelCase: lower part followed by Titlecase part
        else:
            merged.append(x)
    return merged


SHORT_OK = {"a", "i", "an", "as", "at", "be", "by", "do", "go", "if", "in", "is", "it", "me", "my", "no", "of",
            "on", "or", "so", "to", "up", "us", "we", "vs", "ok", "re"}


def _good_part(w):
    if len(w) <= 2:
        return w in SHORT_OK
    if len(w) == 3:
        return zipf_frequency(w, "en") >= 3.8
    if len(w) == 4:
        return zipf_frequency(w, "en") >= 3.3
    return _is_word(w)


@lru_cache(maxsize=200000)
def _segment_good(s):
    """Split lowercase s into good parts only: fewest parts, then most frequent. None if impossible."""
    n = len(s)
    best = [None] * (n + 1)
    best[0] = (0, 0.0, ())
    for i in range(1, n + 1):
        for j in range(max(0, i - 24), i):
            if best[j] is None or not _good_part(s[j:i]):
                continue
            w = s[j:i]
            cand = (best[j][0] + 1, best[j][1] - zipf_frequency(w, "en"), best[j][2] + (w,))
            if best[i] is None or cand[:2] < best[i][:2]:
                best[i] = cand
    return best[n][2] if best[n] else None


def _split_words(run):
    m = re.match(r"(?i)^(.+?['\u2019](?:t|s|re|ll|ve|d|m))([a-z].*)$", run)
    if m:  # contraction followed by fused words: keep the contraction whole
        return [m.group(1)] + _split_words(m.group(2))
    if "'" in run or "\u2019" in run or re.fullmatch(r"[A-Z]{2,}s", run):
        return [run]
    out = []
    for part in _split_case(run):
        low = part.lower()
        if re.search(r"[a-z][A-Z]", part) and not _is_word(low):
            # CamelCase that is not itself a word (thenpossiblyBellman): split at the boundary, then recurse
            out.extend(w for sub in re.split(r"(?<=[a-z])(?=[A-Z])", part) for w in _split_words(sub))
            continue
        if len(part) >= 6 and not _is_word(low) and not part.isupper():
            seg = _segment_good(low)
            if seg and len(seg) >= 2:
                # re-apply original casing
                i, pieces = 0, []
                for w in seg:
                    pieces.append(part[i:i + len(w)])
                    i += len(w)
                out.extend(pieces)
                continue
        out.append(part)
    return out


_PIECE_R = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*|\d+(?:[.:]\d+)*|\.\d+|[^A-Za-z\d]+")


def restore_token(tok, numbers=True, words=True):
    pieces = _PIECE_R.findall(tok)
    out = []
    for k, p in enumerate(pieces):
        prev = pieces[k - 1] if k else ""
        nxt = pieces[k + 1] if k + 1 < len(pieces) else ""
        if p[0].isalpha():
            if words:
                out.append(" ".join(_split_words(p)))
            else:
                out.append(p)
        elif p[0].isdigit() or (p[0] == "." and len(p) > 1 and p[1].isdigit()):
            b, a = _num_glued(prev, p, nxt) if numbers else (False, False)
            out.append((" " if b else "") + p + (" " if a else ""))
        else:
            if words and p in (",", ";", ":") and nxt[:1].isalpha() and prev and not prev[:1].isdigit():
                out.append(p + " ")
            else:
                out.append(p)
    return "".join(out)


_PATHLIKE = re.compile(r"^[~.]?/|^\w+:/|\.(py|md|txt|csv|json|log|sh|cpp|hpp|c|h|png|pdf|zip|tex|bin|so|gz)\b|/[\w.-]+/")


def restore(text, numbers=True, words=True):
    """Return text with missing spaces restored outside code blocks, inline code and identifiers."""
    protected = []

    def stash(m):
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    t = _CODE_BLOCK.sub(stash, text)
    t = _INLINE.sub(stash, t)
    out = []
    for chunk in re.split(r"(\s+)", t):
        if not chunk or chunk.isspace() or "\x00" in chunk or _PATHLIKE.search(chunk):
            out.append(chunk)
            continue
        # treat "/" as a separator between independently restorable parts
        parts = chunk.split("/")
        out.append("/".join(p if (not p or _TOKEN_SKIP.search(p)) else restore_token(p, numbers, words) for p in parts))
    t = "".join(out)
    return re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], t)
