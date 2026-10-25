"""How well can a model estimate the length of its own chain of thought?

For each prompt: ask for a one-word answer, read the true reasoning-token count
from the API usage, then ask in a follow-up turn (with the reasoning kept in
context) how many tokens the reasoning was. Writes JSONL to results/<model>.jsonl.

Usage: python run.py {openai,anthropic} [--model M] [--n N]
"""
import argparse, json, os, re, concurrent.futures as cf
from prompts import PROMPTS

SUFFIX = " Answer with a single word or number only."
FOLLOWUP = ("How many tokens were in your reasoning (chain of thought) for your previous answer? "
            "Give your best estimate as a single integer only.")


def parse_int(s):
    m = re.search(r"\d[\d,]*", s or "")
    return int(m.group().replace(",", "")) if m else None


def run_openai(model, prompt, effort=None, hide_cot=False):
    import openai
    c = openai.OpenAI()
    kw = dict(model=model, reasoning={"effort": effort}) if effort else dict(model=model)
    r1 = c.responses.create(input=prompt + SUFFIX, **kw)
    if hide_cot:  # control: replay only the visible answer, dropping the reasoning item
        hist = [{"role": "user", "content": prompt + SUFFIX},
                {"role": "assistant", "content": r1.output_text},
                {"role": "user", "content": FOLLOWUP}]
        r2 = c.responses.create(input=hist, **kw)
    else:
        r2 = c.responses.create(previous_response_id=r1.id, input=FOLLOWUP, **kw)
    return dict(answer=r1.output_text, actual=r1.usage.output_tokens_details.reasoning_tokens,
                total_output=r1.usage.output_tokens, claim_text=r2.output_text, stop=r2.status)


def run_anthropic(model, prompt, effort=None, hide_cot=False):
    import anthropic
    c = anthropic.Anthropic()
    kw = dict(model=model, max_tokens=32000, thinking={"type": "adaptive"})
    if effort:
        kw["output_config"] = {"effort": effort}
    msgs = [{"role": "user", "content": prompt + SUFFIX}]
    m1 = c.messages.create(messages=msgs, **kw)
    msgs += [{"role": "assistant", "content": m1.content}, {"role": "user", "content": FOLLOWUP}]
    m2 = c.messages.create(messages=msgs, **kw)
    text = lambda m: "".join(b.text for b in m.content if b.type == "text")
    return dict(answer=text(m1), actual=m1.usage.output_tokens_details.thinking_tokens,
                total_output=m1.usage.output_tokens, claim_text=text(m2), stop=m2.stop_reason)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("provider", choices=["openai", "anthropic"])
    ap.add_argument("--model")
    ap.add_argument("--n", type=int, default=len(PROMPTS))
    ap.add_argument("--effort")
    ap.add_argument("--hide-cot", action="store_true", help="control: drop reasoning from follow-up context")
    a = ap.parse_args()
    model = a.model or {"openai": "gpt-6-astra", "anthropic": "claude-opus-5-5"}[a.provider]
    fn = run_openai if a.provider == "openai" else run_anthropic
    tag = model + (f"_{a.effort}" if a.effort else "") + ("_hidecot" if a.hide_cot else "")
    os.makedirs("results", exist_ok=True)
    with cf.ThreadPoolExecutor(8) as ex, open(f"results/{tag}.jsonl", "w") as f:
        futs = {ex.submit(fn, model, p, a.effort, a.hide_cot): p for p in PROMPTS[:a.n]}
        for fut in cf.as_completed(futs):
            try:
                row = dict(prompt=futs[fut], **fut.result())
                row["claimed"] = parse_int(row["claim_text"])
            except Exception as e:
                row = dict(prompt=futs[fut], error=repr(e))
            print(json.dumps(row)[:200], flush=True)
            f.write(json.dumps(row) + "\n")
