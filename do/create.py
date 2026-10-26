"""Create a DigitalOcean droplet that runs a batch of squeeze runs and syncs logs back via OpenAI Files.

usage: python do/create.py <tag> <size> name:task:ncores [...]
Requires DIGITALOCEAN_ACCESS_TOKEN and OPENAI_API_KEY in the environment.
"""
import base64
import io
import json
import os
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.digitalocean.com/v2"


def do(method, path, body=None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {os.environ['DIGITALOCEAN_ACCESS_TOKEN']}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        sys.exit(f"DO API {e.code}: {e.read().decode()[:500]}")


tag, size = sys.argv[1], sys.argv[2]
agent_args = os.environ.get("AGENT_ARGS", "")
runs = sys.argv[3:]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for p in list((ROOT / "squeeze").glob("*.py")) + list((ROOT / "tasks").glob("*.md")):
        tar.add(p, arcname=str(p.relative_to(ROOT)))
user_data = ((ROOT / "do/bootstrap.sh.tmpl").read_text()
             .replace("__CODE_B64__", base64.b64encode(buf.getvalue()).decode())
             .replace("__OPENAI_KEY__", os.environ["OPENAI_API_KEY"])
             .replace("__TAG__", tag)
             .replace("__AGENT_ARGS__", agent_args)
             .replace("__RUNS__", " ".join(runs)))
assert len(user_data) < 64000, len(user_data)
keys = [k["id"] for k in do("GET", "/account/keys")["ssh_keys"]]
d = do("POST", "/droplets", {"name": f"squeeze-{tag}", "region": os.environ.get("DO_REGION", "nyc1"), "size": size, "image": "ubuntu-24-04-x64",
                             "ssh_keys": keys, "user_data": user_data, "tags": ["squeeze"]})
print(d["droplet"]["id"], d["droplet"]["name"], "ssh keys:", len(keys))
