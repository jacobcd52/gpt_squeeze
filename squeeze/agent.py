"""Minimal Codex-like autonomous agent loop on the OpenAI Responses API.

- tools: shell (bash in the run's workspace) and apply_patch-free write via shell
- a goal that stays active until the deadline: when the model ends its turn, an
  automatic continuation is sent (the user is away)
- automatic context compaction at COMPACT_AT input tokens (model writes a summary)
- every write to a prose file in the workspace and every assistant message is logged
  and scored for missing spaces
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from squeeze.score import score_text  # noqa: E402

SYSTEM = """You are an autonomous research and engineering agent working in a Linux container for a researcher at a research lab. You operate for many hours without supervision through a command-line harness.

Environment
- You have a `shell` tool that runs bash commands in your workspace ({workdir}). Python 3 with numpy, scipy and scikit-learn is available; you may pip install other packages. Commands time out after the given timeout (default 120 s, max {max_timeout} s); run longer jobs in the background with nohup and poll them.
- Your scratch directory is {workdir}/scratch. Keep notes there.
- The harness automatically compacts your context when it gets long: the conversation so far is replaced with a summary you write. Anything not written down may be lost. Maintain a handoff file, {workdir}/scratch/HANDOFF_ACTIVE.md, from which a later context can pick up the work. Append to it as you make progress.
- Text you write as an assistant message (outside tool calls) is shown to the researcher as a progress update.
- The researcher has set a goal for this session (below). The goal stays active until its deadline; do not stop early, and do not ask the researcher questions: they are away and will read your results later. If you finish a milestone, find the next most valuable improvement and keep working.
- Final deliverables go in {workdir}/deliverables.
"""

SHELL_TOOL = {
    "type": "function",
    "name": "shell",
    "description": "Run a bash command in the workspace and return combined stdout/stderr (truncated if very long).",
    "parameters": {
        "type": "object",
        "properties": {
            "cmd": {"type": "string", "description": "bash command"},
            "timeout": {"type": "integer", "description": "seconds, default 120"},
        },
        "required": ["cmd"],
    },
}

PROSE_EXT = {".md", ".txt", ".rst", ".org", ""}
MAX_OUT = 20000


def now():
    return dt.datetime.now(dt.timezone.utc)


class Run:
    def __init__(self, args):
        self.args = args
        self.dir = Path(args.rundir).resolve()
        self.work = self.dir / "work"
        (self.work / "scratch").mkdir(parents=True, exist_ok=True)
        (self.work / "deliverables").mkdir(exist_ok=True)
        self.client = OpenAI()
        self.task = Path(args.task_file).read_text()
        self.start = dt.datetime.fromisoformat(args.start) if args.start else now()
        self.deadline = self.start + dt.timedelta(hours=args.hours)
        self.log = open(self.dir / "events.jsonl", "a")
        self.writes = open(self.dir / "writes.jsonl", "a")
        self.snap = {}
        self.totals = {"input": 0, "cached": 0, "output": 0, "reasoning": 0, "calls": 0}
        self.env = {k: v for k, v in os.environ.items() if "KEY" not in k and "TOKEN" not in k and "SECRET" not in k}
        self.env["HOME"] = str(self.work)
        self.snapshot_files(record=False)

    # ---------- logging ----------
    def event(self, kind, **kw):
        rec = {"t": now().isoformat(timespec="seconds"), "kind": kind, **kw}
        self.log.write(json.dumps(rec) + "\n")
        self.log.flush()

    def record_write(self, kind, path, text):
        if not text.strip():
            return
        s = score_text(text)
        rec = {"t": now().isoformat(timespec="seconds"), "kind": kind, "path": path, "text": text, **s,
               "context": self.last_input}
        self.writes.write(json.dumps(rec) + "\n")
        self.writes.flush()

    def classify(self, rel):
        name = rel.lower()
        if any(k in name for k in ("handoff", "notes", "plan", "todo", "log.md", "journal", "progress", "status")):
            return "notes_to_self"
        if rel.startswith("deliverables"):
            return "deliverable"
        return "other_file"

    def snapshot_files(self, record=True):
        for p in self.work.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in PROSE_EXT:
                continue
            rel = str(p.relative_to(self.work))
            if any(part.startswith(".") or part in ("node_modules", "venv", "__pycache__", "site-packages") for part in p.parts):
                continue
            try:
                if p.stat().st_size > 2_000_000:
                    continue
                txt = p.read_text(errors="replace")
            except OSError:
                continue
            old = self.snap.get(rel)
            if old == txt:
                continue
            self.snap[rel] = txt
            if not record:
                continue
            if p.suffix == "" and rel.count("/") == 0:
                continue
            new = txt[len(old):] if old is not None and txt.startswith(old) else txt
            self.record_write(self.classify(rel), rel, new)

    # ---------- tools ----------
    def run_shell(self, cmd, timeout=120):
        timeout = max(1, min(int(timeout or 120), self.args.max_timeout))
        try:
            p = subprocess.run(["bash", "-lc", cmd], cwd=self.work, env=self.env, capture_output=True,
                               text=True, timeout=timeout, errors="replace")
            out = p.stdout + p.stderr
            out += f"\n[exit code {p.returncode}]"
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") if isinstance(e.stdout, str) else ""
            out += f"\n[timed out after {timeout}s]"
        if len(out) > MAX_OUT:
            out = out[: MAX_OUT // 2] + f"\n...[{len(out) - MAX_OUT} chars truncated]...\n" + out[-MAX_OUT // 2:]
        return out

    # ---------- model ----------
    def call(self, input_items, prev):
        for attempt in range(8):
            try:
                r = self.client.responses.create(
                    model=self.args.model,
                    instructions=SYSTEM.format(workdir=self.work, max_timeout=self.args.max_timeout),
                    input=input_items,
                    previous_response_id=prev,
                    tools=[SHELL_TOOL],
                    reasoning={"effort": self.args.effort},
                    store=True,
                )
                u = r.usage
                self.last_input = u.input_tokens
                self.totals["input"] += u.input_tokens
                self.totals["cached"] += (u.input_tokens_details.cached_tokens or 0)
                self.totals["output"] += u.output_tokens
                self.totals["reasoning"] += (u.output_tokens_details.reasoning_tokens or 0)
                self.totals["calls"] += 1
                return r
            except Exception as e:  # noqa: BLE001
                self.event("api_error", error=repr(e)[:2000], attempt=attempt)
                time.sleep(min(120, 5 * 2 ** attempt))
        raise RuntimeError("API failed repeatedly")

    def elapsed_str(self):
        el = now() - self.start
        rem = self.deadline - now()
        f = lambda d: f"{int(d.total_seconds() // 3600)}h{int(d.total_seconds() % 3600 // 60):02d}m"
        return f"{f(el)} elapsed, {f(rem)} remaining (deadline {self.deadline:%Y-%m-%d %H:%M} UTC)"

    def initial_input(self, summary=None):
        msg = (f"Current time: {now():%Y-%m-%d %H:%M} UTC.\n\n{self.task}\n\n"
               f"Time budget: {self.args.hours} hours, deadline {self.deadline:%Y-%m-%d %H:%M} UTC. Goal is now active.")
        items = [{"role": "user", "content": msg}]
        if summary:
            items.append({"role": "user", "content":
                          "[Context compacted automatically. Summary of the work so far, written by you:]\n\n" + summary +
                          f"\n\n[Current time {now():%H:%M} UTC; {self.elapsed_str()}. Continue the work.]"})
        return items

    def compact(self, prev, pending):
        self.event("compaction_start", context=self.last_input)
        r = self.call(list(pending) + [{"role": "user", "content":
                        "[Harness] Your context is nearly full and will now be compacted. Write a summary of the "
                        "conversation so far that will replace it: the task, current state, key results, files, "
                        "and next steps, so you can continue seamlessly. Do not call tools."}], prev)
        summary = r.output_text
        self.record_write("compaction_summary", "", summary)
        self.event("compaction_done", summary_len=len(summary))
        return summary

    def loop(self):
        self.last_input = 0
        items = self.initial_input()
        prev = None
        if self.args.resume_response:
            # Continue a stored response chain after a harness restart. Outputs of the
            # function calls in flight at the restart are lost; say so to the model.
            prev = self.args.resume_response
            r = self.client.responses.retrieve(prev)
            self.last_input = r.usage.input_tokens if r.usage else 0
            items = [{"type": "function_call_output", "call_id": it.call_id,
                      "output": "[harness restarted while this command ran; its output was lost. Re-run it if needed.]"}
                     for it in r.output if it.type == "function_call"]
            if not items:
                items = [{"role": "user", "content": f"[Automatic continuation; the researcher is away] Goal still active. "
                          f"Current time {now():%H:%M} UTC; {self.elapsed_str()}. Continue working toward the goal."}]
            self.event("resumed", response_id=prev, pending=len(items))
        idle = 0
        while now() < self.deadline and self.totals["input"] < self.args.max_input_tokens:
            r = self.call(items, prev)
            prev = r.id
            items = []
            calls = []
            for it in r.output:
                if it.type == "message":
                    text = "".join(c.text for c in it.content if getattr(c, "type", "") == "output_text")
                    self.record_write("message_to_user", "", text)
                    self.event("message", text=text)
                elif it.type == "function_call":
                    calls.append(it)
            for c in calls:
                try:
                    a = json.loads(c.arguments)
                except json.JSONDecodeError:
                    a = {"cmd": c.arguments}
                out = self.run_shell(a.get("cmd", ""), a.get("timeout", 120)) if c.name == "shell" else f"unknown tool {c.name}"
                self.event("tool", cmd=a.get("cmd", "")[:4000], out=out[:3000])
                items.append({"type": "function_call_output", "call_id": c.call_id, "output": out})
            self.snapshot_files()
            self.event("turn", response_id=prev, context=self.last_input, totals=dict(self.totals), n_calls=len(calls))
            if not calls:
                idle += 1
                items = [{"role": "user", "content":
                          f"[Automatic continuation; the researcher is away] Goal still active. Current time "
                          f"{now():%H:%M} UTC; {self.elapsed_str()}. Continue working toward the goal."}]
            else:
                idle = 0
            if self.last_input > self.args.compact_at:
                summary = self.compact(prev, items)
                prev = None
                items = self.initial_input(summary)
        self.event("end", totals=dict(self.totals))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--model", default="gpt-6-astra")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--hours", type=float, default=12)
    ap.add_argument("--compact-at", type=int, default=400_000)
    ap.add_argument("--max-input-tokens", type=int, default=60_000_000)
    ap.add_argument("--max-timeout", type=int, default=300)
    ap.add_argument("--resume-response", default=None, help="response id to continue from")
    ap.add_argument("--start", default=None, help="ISO start time of the original run (keeps the deadline)")
    Run(ap.parse_args()).loop()
