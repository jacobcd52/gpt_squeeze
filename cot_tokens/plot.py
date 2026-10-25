"""Actual vs claimed reasoning tokens. Left: reasoning kept in context. Right: control with it removed."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2a78d6", "#eb6834"]  # categorical slots 1, 2
MODELS = [m for m in ["gpt-6-astra", "claude-opus-5-5"] if os.path.exists(f"results/{m}.jsonl")]


def load(path):
    rows = [r for r in map(json.loads, open(path)) if r.get("claimed") is not None]
    return np.array([r["actual"] for r in rows]), np.array([r["claimed"] for r in rows])


fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
for ax, suffix, title in [(axes[0], "", "Reasoning visible in follow-up"),
                          (axes[1], "_hidecot", "Control: reasoning removed from context")]:
    for model in MODELS:
        for (eff, label), color in zip([("", "default effort"), ("_high", "high effort")], COLORS):
            path = f"results/{model}{eff}{suffix}.jsonl"
            if not os.path.exists(path):
                continue
            x, y = load(path)
            ax.scatter(x + 1, y + 1, s=36, color=color, alpha=0.8, edgecolor="white", linewidth=1,
                       label=f"{model}, {label} (n={len(x)})")
    ax.plot([1, 2e4], [1, 2e4], color="#888", ls="--", lw=1, label="y = x")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Actual reasoning tokens + 1 (API usage)")
    ax.grid(True, which="major", color="#e5e5e5", lw=0.6); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Claimed reasoning tokens + 1")
axes[0].legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("results/actual_vs_claimed.png", dpi=150)
print("saved results/actual_vs_claimed.png")
