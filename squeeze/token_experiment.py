"""Does restoring the missing spaces change the token count of the agents' notes?

For each handoff/status note (and, as a control, each message to the user) in
data/token_experiment_input.jsonl, count o200k_base tokens as written, with only
glued numbers separated, with only fused words split, and with both.
Writes data/token_experiment_results.jsonl and prints summary tables.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from squeeze.restore import restore  # noqa: E402
from squeeze.score import score_text  # noqa: E402

enc = tiktoken.get_encoding("o200k_base")
ntok = lambda t: len(enc.encode(t))  # noqa: E731
VARIANTS = {"num": (True, False), "word": (False, True), "both": (True, True)}

recs = [json.loads(l) for l in open("data/token_experiment_input.jsonl")]
out = []
for r in recs:
    t0 = r["text"]
    row = {k: r[k] for k in ("run", "t", "kind", "path", "context")}
    row.update(score=score_text(t0)["score"], tok=ntok(t0), chars=len(t0), spaces=t0.count(" "))
    for v, (n, w) in VARIANTS.items():
        tv = restore(t0, numbers=n, words=w)
        row[f"tok_{v}"] = ntok(tv)
        row[f"chars_{v}"] = len(tv)
        row[f"added_{v}"] = tv.count(" ") - row["spaces"]
    out.append(row)
with open("data/token_experiment_results.jsonl", "w") as fh:
    for row in out:
        fh.write(json.dumps(row) + "\n")


def pct(a, b):
    return 100.0 * (a - b) / b


def pooled(rows, v):
    T = sum(r["tok"] for r in rows)
    return pct(sum(r[f"tok_{v}"] for r in rows), T), pct(sum(r[f"chars_{v}"] for r in rows), sum(r["chars"] for r in rows))


bins = [("messages to user (control)", lambda r: r["kind"] == "message"),
        ("notes, score <5", lambda r: r["kind"] == "note" and r["score"] < 5),
        ("notes, score 5-20", lambda r: r["kind"] == "note" and 5 <= r["score"] < 20),
        ("notes, score 20-50", lambda r: r["kind"] == "note" and 20 <= r["score"] < 50),
        ("notes, score >=50", lambda r: r["kind"] == "note" and r["score"] >= 50),
        ("all notes scoring >=20", lambda r: r["kind"] == "note" and r["score"] >= 20)]
print("Pooled change when spaces are restored (negative = squeezed text costs MORE tokens)")
print(f"{'':28s} {'n':>4s} {'runs':>4s} {'tokens':>8s} | {'num only':>9s} {'words only':>10s} {'both':>8s} | {'chars both':>10s}")
for name, f in bins:
    rows = [r for r in out if f(r)]
    if not rows:
        continue
    vals = {v: pooled(rows, v) for v in VARIANTS}
    print(f"{name:28s} {len(rows):4d} {len({r['run'] for r in rows}):4d} {sum(r['tok'] for r in rows):8d} | "
          f"{vals['num'][0]:+8.2f}% {vals['word'][0]:+9.2f}% {vals['both'][0]:+7.2f}% | {vals['both'][1]:+9.2f}%")

print("\nPer restored space (notes scoring >=20): mean token change per space added")
rows = [r for r in out if r["kind"] == "note" and r["score"] >= 20]
for v in ("num", "word"):
    added = sum(r[f"added_{v}"] for r in rows)
    dt = sum(r[f"tok_{v}"] - r["tok"] for r in rows)
    print(f"  {v:5s}: {added} spaces added, {dt:+d} tokens, {dt / added:+.3f} tokens per space")

print("\nPer run (notes scoring >=20): pooled % token change when restored")
print(f"{'run':22s} {'n':>3s} {'num':>7s} {'word':>7s} {'both':>7s}")
byrun = defaultdict(list)
for r in rows:
    byrun[r["run"]].append(r)
signs = defaultdict(lambda: [0, 0])
for run in sorted(byrun):
    rr = byrun[run]
    vals = {v: pooled(rr, v)[0] for v in VARIANTS}
    for v in VARIANTS:
        signs[v][vals[v] < 0] += 1
    print(f"{run:22s} {len(rr):3d} {vals['num']:+6.2f}% {vals['word']:+6.2f}% {vals['both']:+6.2f}%")
print("\nRuns where restoring REDUCES tokens / runs where it increases:")
for v in VARIANTS:
    print(f"  {v:5s}: {signs[v][1]} reduce, {signs[v][0]} increase")
per_note = {v: [pct(r[f'tok_{v}'], r['tok']) for r in rows] for v in VARIANTS}
for v in VARIANTS:
    xs = per_note[v]
    print(f"  per-note {v:5s}: median {statistics.median(xs):+.2f}%, range {min(xs):+.1f}% to {max(xs):+.1f}%, "
          f"{sum(x < 0 for x in xs)}/{len(xs)} notes reduce")
