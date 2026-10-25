import json, sys, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(6, 6))
for path in sorted(glob.glob("results/*.jsonl")):
    rows = [r for r in map(json.loads, open(path)) if r.get("claimed") and r.get("actual")]
    if not rows:
        continue
    x = np.array([r["actual"] for r in rows]); y = np.array([r["claimed"] for r in rows])
    r = np.corrcoef(np.log(x), np.log(y))[0, 1]
    name = path.split("/")[-1][:-6]
    ax.scatter(x, y, alpha=0.7, label=f"{name} (n={len(x)}, log-r={r:.2f}, median ratio={np.median(y/x):.2f})")
lo, hi = ax.get_xlim()[0], max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.set_xscale("log"); ax.set_yscale("log")
ax.plot([1, 1e5], [1, 1e5], "k--", lw=1, label="y = x")
ax.set_xlabel("Actual reasoning tokens (API usage)"); ax.set_ylabel("Model's claimed reasoning tokens")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("results/actual_vs_claimed.png", dpi=150)
