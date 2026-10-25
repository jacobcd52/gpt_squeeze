"""Missing-spaces score, approximating the ARC 'Squeezed notes' metric.

score = 100 * missing_spaces / words, where words counts recovered words.
A missing space is either (a) a boundary inside a fused letter run (>=10 letters,
not a word, splits into >=3 common words) or (b) a word glued to a number.
"""
import math
import re
from functools import lru_cache

from wordfreq import top_n_list, zipf_frequency

_VOCAB = {w for w in top_n_list("en", 60000) if w.isalpha() and (len(w) > 1 or w in ("a", "i"))}
_VOCAB |= {"ok", "csv", "json", "mse", "api", "cpu", "gpu", "url", "todo", "repo", "eval", "init",
           "config", "params", "grads", "baseline", "subagent", "subagents", "dev", "regen",
           "timeout", "runtime", "dataset", "datasets", "benchmark", "benchmarks", "handoff"}
UNITS = {"ms", "us", "ns", "s", "sec", "secs", "min", "mins", "h", "hr", "hrs", "x", "k", "m", "b",
         "gb", "mb", "kb", "tb", "px", "d", "th", "st", "nd", "rd", "e", "utc", "pct", "bit", "bits",
         "byte", "bytes", "hz", "khz", "mhz", "ghz", "g", "kg", "cm", "mm", "km", "w", "kw", "v", "am", "pm", "t", "n", "f", "c", "p"}


def _is_word(w):
    if w in _VOCAB or zipf_frequency(w, "en") >= 1.0:
        return True
    for suf in ("s", "es", "ed", "ing", "ly", "ness", "ation", "ations"):
        if w.endswith(suf) and len(w) - len(suf) >= 4 and zipf_frequency(w[: -len(suf)], "en") >= 2.0:
            return True
    return False


@lru_cache(maxsize=200000)
def segment(s):
    """Best split of lowercase s into vocab words (fewest words, then most frequent). None if impossible."""
    n = len(s)
    best = [None] * (n + 1)
    best[0] = (0, 0.0, ())
    for i in range(1, n + 1):
        for j in range(max(0, i - 24), i):
            if best[j] is None:
                continue
            w = s[j:i]
            if w not in _VOCAB:
                continue
            cand = (best[j][0] + 1, best[j][1] - zipf_frequency(w, "en"), best[j][2] + (w,))
            if best[i] is None or cand[:2] < best[i][:2]:
                best[i] = cand
    return best[n][2] if best[n] else None


def fused_words(run):
    """Number of missing spaces inside an alphabetic run (0 if not fused)."""
    low = run.lower()
    if len(low) < 10 or _is_word(low):
        return 0
    seg = segment(low)
    if seg and len(seg) >= 3:
        return len(seg) - 1
    return 0


_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_INLINE = re.compile(r"`[^`\n]*`")
_TOKEN_SKIP = re.compile(r"[/_=<>\\{}\[\]|$]|\w\.(py|md|txt|csv|json|log|sh|cpp|hpp|c|h|png|pdf|zip|tex|html|js|ts|npz|npy|pt|pkl|yaml|toml)\b|^https?:")
_PIECE = re.compile(r"[A-Za-z]+|\d+(?:[.:]\d+)*|\.\d+|[^A-Za-z\d]+")


def score_text(text):
    text = _CODE_BLOCK.sub(" ", text)
    text = _INLINE.sub(" ", text)
    words = 0
    missing = 0
    numbers = 0
    glued_numbers = 0
    for tok in text.split():
        if _TOKEN_SKIP.search(tok):
            continue
        pieces = _PIECE.findall(tok)
        if not any(p[0].isalpha() for p in pieces):
            if any(p[0].isdigit() for p in pieces):
                numbers += sum(1 for p in pieces if p[0].isdigit())
            continue
        words += 1
        for k, p in enumerate(pieces):
            if p[0].isalpha():
                missing += fused_words(p)
            elif p[0].isdigit() or (p[0] == "." and len(p) > 1 and p[1].isdigit()):
                numbers += 1
                prev = pieces[k - 1] if k > 0 else ""
                nxt = pieces[k + 1] if k + 1 < len(pieces) else ""
                g = False
                if prev[:1].isalpha():
                    pl = prev.lower()
                    long_num = len(p) >= 3 or "." in p or ":" in p
                    if long_num and len(pl) >= 2 and not (len(pl) <= 2 and pl in UNITS):
                        g = True
                    elif len(p) >= 2 and len(pl) >= 3 and _is_word(pl):
                        g = True
                    elif len(pl) >= 3 and nxt[:1].isalpha():
                        g = True
                if nxt[:1].isalpha():
                    nl = nxt.lower()
                    if nl in UNITS or nl.rstrip("s") in UNITS:
                        pass
                    else:
                        unit = next((u for u in sorted(UNITS, key=len, reverse=True)
                                     if len(u) >= 2 and nl.startswith(u) and _is_word(nl[len(u):]) and len(nl) - len(u) >= 2), None)
                        if unit or len(nl) >= 3:
                            g = True
                if g:
                    glued_numbers += 1
                    missing += 1
    total_words = words + missing
    return {
        "words": total_words,
        "missing": missing,
        "score": 100.0 * missing / total_words if total_words else 0.0,
        "numbers": numbers,
        "glued_numbers": glued_numbers,
        "glued_frac": glued_numbers / numbers if numbers else None,
    }


if __name__ == "__main__":
    import sys
    print(score_text(sys.stdin.read()))
