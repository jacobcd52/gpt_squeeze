"""On a remote box: periodically upload run logs to OpenAI Files (purpose=batch, which can be downloaded back).

One JSONL line {"tag", "t", "tgz_b64"}; older uploads with the same filename are deleted.
"""
import base64
import io
import json
import sys
import tarfile
import time
from pathlib import Path

from openai import OpenAI

tag, root = sys.argv[1], Path(sys.argv[2])
fname = f"squeeze_sync_{tag}.jsonl"
c = OpenAI()
while True:
    try:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for run in sorted(root.iterdir()):
                for p in [run / "events.jsonl", run / "writes.jsonl", run / "agent.out", *run.glob("work/scratch/*.md")]:
                    if p.is_file():
                        tar.add(p, arcname=str(p.relative_to(root)))
        line = json.dumps({"tag": tag, "t": time.time(), "tgz_b64": base64.b64encode(buf.getvalue()).decode()}) + "\n"
        new = c.files.create(file=(fname, io.BytesIO(line.encode())), purpose="batch")
        for f in c.files.list(purpose="batch"):
            if f.filename == fname and f.id != new.id:
                c.files.delete(f.id)
    except Exception as e:  # noqa: BLE001
        print("sync error", repr(e)[:300], flush=True)
    time.sleep(180)
