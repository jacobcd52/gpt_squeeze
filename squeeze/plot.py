"""Small multiples per run: number-glue rate and word-fusion rate of each handoff write,
against handoff-write index and against context length. Writes PNGs to the given directory."""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63", "#e6e5e0"

runs = {}
for f in sorted(glob.glob("/home/user/runs*/*/writes.jsonl")):
    rows = []
    for line in open(f):
        w = json.loads(line)
        if "handoff" not in w["path"].lower() or w["words"] < 20:
            continue
        words_only = w["words"] - w["glued_numbers"]
        rows.append({
            "ctx": w["context"] / 1000,
            "num": 100 * w["glued_numbers"] / w["numbers"] if w["numbers"] >= 5 else None,
            "word": 100 * (w["missing"] - w["glued_numbers"]) / max(words_only, 1),
        })
    if rows:
        runs[Path(f).parent.name] = rows
order = sorted(runs, key=lambda r: -max(x["word"] for x in runs[r]))


def figure(xkey, xlabel, fname):
    n = len(order)
    cols = 5
    nrows = (n + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(3.2 * cols, 2.5 * nrows), sharey=True, squeeze=False)
    for ax, run in zip(axes.flat, order):
        rows = runs[run]
        xs = list(range(1, len(rows) + 1)) if xkey == "idx" else [r["ctx"] for r in rows]
        nx = [x for x, r in zip(xs, rows) if r["num"] is not None]
        ny = [r["num"] for r in rows if r["num"] is not None]
        ax.plot(nx, ny, color=BLUE, lw=2, marker="o", ms=4, label="numbers glued to a word (% of numbers)")
        ax.plot(xs, [r["word"] for r in rows], color=ORANGE, lw=2, marker="o", ms=4,
                label="fused words (missing spaces per 100 words, numbers excluded)")
        ax.set_title(run.replace("astra_", "local ").replace("do1_", "droplet "), fontsize=10, color=INK, loc="left")
        ax.set_ylim(-3, 103)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        if xkey == "idx":
            ax.xaxis.get_major_locator().set_params(integer=True)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("%", color=MUTED)
    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.01, 1.0))
    fig.suptitle("Squeezing in each Astra run's handoff file, per write", x=0.01, y=1.04, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / fname, dpi=130, bbox_inches="tight", facecolor="white")


figure("idx", "handoff write #", "squeeze_vs_writes.png")
figure("ctx", "context length (thousand tokens)", "squeeze_vs_context.png")
print("ok", len(order), "runs")
