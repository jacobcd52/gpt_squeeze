"""Emit one line per notable event across runs: squeezed notes, compactions, API errors, run ends."""
import json
import time
from pathlib import Path

root = Path("/home/user/runs")
pos = {}
best = {}
while True:
    for f in list(root.glob("*/writes.jsonl")) + list(root.glob("*/events.jsonl")):
        p = pos.get(f, 0)
        with open(f) as fh:
            fh.seek(p)
            for line in fh:
                if not line.endswith("\n"):
                    break
                p += len(line.encode())
                r = json.loads(line)
                run = f.parent.name
                if f.name == "writes.jsonl" and r["kind"] == "notes_to_self" and r["words"] >= 20:
                    g = r["glued_frac"] or 0
                    b = best.setdefault(run, [5.0, 0.3])
                    if r["score"] >= b[0] + 5 or g >= b[1] + 0.15:
                        b[0] = max(b[0], r["score"]); b[1] = max(b[1], g)
                        print(f"SQUEEZE? {run} {r['t'][11:16]} ctx={r['context']//1000}k score={r['score']:.0f} glued={100*g:.0f}% :: {r['text'][:160]!r}", flush=True)
                elif f.name == "events.jsonl" and r["kind"] in ("compaction_done", "end"):
                    print(f"{r['kind'].upper()} {run} {r['t'][11:16]}", flush=True)
                elif f.name == "events.jsonl" and r["kind"] == "api_error" and r.get("attempt", 0) >= 2:
                    print(f"API_ERROR {run} attempt={r['attempt']} {r['error'][:200]}", flush=True)
        pos[f] = p
    time.sleep(20)
