"""Summarize all runs: progress, tokens, and missing-space scores by kind of text."""
import json
import sys
from pathlib import Path

args = [a for a in sys.argv[1:] if not a.startswith("-")]
root = Path(args[0] if args else "/home/user/runs")
verbose = "-v" in sys.argv
for d in sorted(p for p in root.iterdir() if (p / "events.jsonl").exists()):
    turns = [json.loads(l) for l in open(d / "events.jsonl") if '"kind": "turn"' in l]
    ev = [json.loads(l) for l in open(d / "events.jsonl")]
    writes = [json.loads(l) for l in open(d / "writes.jsonl")] if (d / "writes.jsonl").exists() else []
    last = turns[-1] if turns else {}
    tot = last.get("totals", {})
    ncomp = sum(e["kind"] == "compaction_done" for e in ev)
    errs = sum(e["kind"] == "api_error" for e in ev)
    ended = any(e["kind"] == "end" for e in ev)
    print(f"== {d.name}: turns={len(turns)} ctx={last.get('context', 0)/1e3:.0f}k in={tot.get('input', 0)/1e6:.1f}M "
          f"cached={tot.get('cached', 0)/1e6:.1f}M out={tot.get('output', 0)/1e3:.0f}k compactions={ncomp} "
          f"api_errors={errs}{' ENDED' if ended else ''} last={last.get('t', '')[11:16]}")
    by = {}
    for w in writes:
        b = by.setdefault(w["kind"], [0, 0, 0, 0, 0])
        b[0] += 1; b[1] += w["words"]; b[2] += w["missing"]; b[3] += w["numbers"]; b[4] += w["glued_numbers"]
    for k, (n, words, miss, nums, glued) in by.items():
        print(f"   {k:18s} n={n:3d} words={words:6d} miss/100w={100*miss/max(words,1):5.1f} glued_nums={100*glued/max(nums,1):4.0f}%")
    notes = [w for w in writes if w["kind"] == "notes_to_self" and w["words"] >= 20]
    if notes:
        print("   notes trajectory (time ctx score glued%):",
              " ".join(f"{w['t'][11:16]}/{w['context']//1000}k/{w['score']:.0f}/{0 if w['glued_frac'] is None else 100*w['glued_frac']:.0f}" for w in notes[-12:]))
        if verbose:
            w = notes[-1]
            print("   last note excerpt:", w["text"][:600].replace("\n", " | "))
