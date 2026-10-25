"""Emit one line per notable event across runs: squeezed notes, compactions, API errors, run ends."""
import json
import subprocess
import sys
import time
from pathlib import Path

roots = [Path("/home/user/runs"), Path("/home/user/runs_do")]
sync = Path(__file__).with_name("sync_down.py")
tick = 0
pos = {}
best = {}
first = True
while True:
    out = []
    if tick % 9 == 0:
        subprocess.run([sys.executable, str(sync)], capture_output=True, timeout=300)
    tick += 1
    for f in [f for root in roots for pat in ("*/writes.jsonl", "*/events.jsonl") for f in root.glob(pat)]:
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
                        out.append(f"SQUEEZE? {run} {r['t'][11:16]} ctx={r['context']//1000}k score={r['score']:.0f} glued={100*g:.0f}% :: {r['text'][:160]!r}")
                elif f.name == "events.jsonl" and r["kind"] in ("compaction_done", "end"):
                    out.append(f"{r['kind'].upper()} {run} {r['t'][11:16]}")
                elif f.name == "events.jsonl" and r["kind"] == "api_error" and r.get("attempt", 0) >= 2:
                    out.append(f"API_ERROR {run} attempt={r['attempt']} {r['error'][:200]}")
        pos[f] = p
    if not first:
        for o in out:
            print(o, flush=True)
    first = False
    time.sleep(20)
