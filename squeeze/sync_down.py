"""Pull the newest synced run logs from OpenAI Files into <dest>/ (default /home/user/runs_do)."""
import base64
import io
import json
import sys
import tarfile
from pathlib import Path

from openai import OpenAI

dest = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/user/runs_do")
dest.mkdir(parents=True, exist_ok=True)
c = OpenAI()
newest = {}
for f in c.files.list(purpose="batch"):
    if f.filename.startswith("squeeze_sync_") and (f.filename not in newest or f.created_at > newest[f.filename].created_at):
        newest[f.filename] = f
for f in newest.values():
    rec = json.loads(c.files.content(f.id).read())
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(rec["tgz_b64"])), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
    print(f"pulled {f.filename} ({f.bytes} bytes)")
