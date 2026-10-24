"""Render standalone figures from the archived training and post-training results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs"
HISTORY = json.loads((ROOT / "docs/experiments/20261024/history.json").read_text())
SUMMARY = json.loads((ROOT / "docs/experiments/20261024/attack_summary.json").read_text())
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 180})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


epochs = [row["epoch"] for row in HISTORY]
fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True, layout="constrained")
axes[0].plot(epochs, [row["train_loss"] for row in HISTORY], "o-", label="Training MSE", ms=3)
axes[0].plot(epochs, [row["val_loss"] for row in HISTORY], "s-", label="Validation MSE", ms=3)
axes[0].set(yscale="log", ylabel="MSE", title="Two-round AES distinguisher: full training")
axes[0].legend(loc="upper left")
axes[0].annotate("Epoch 14: validation failure", xy=(14, HISTORY[13]["val_loss"]),
                 xytext=(9, 0.07), arrowprops={"arrowstyle": "->"})
for key, label in [("val_acc", "Accuracy"), ("tpr", "TPR"), ("tnr", "TNR")]:
    axes[1].plot(epochs, [row[key] * 100 for row in HISTORY], "o-", label=label, ms=3)
axes[1].set(ylabel="Validation rate (%)", ylim=(45, 103))
axes[1].legend(loc="lower left", ncol=3)
axes[2].plot(epochs, [row["lr"] for row in HISTORY], "o-", color="#7c3aed", ms=3)
axes[2].set(xlabel="Epoch", ylabel="Learning rate", xticks=epochs)
for ax in axes:
    ax.axvline(1, color="#166534", ls=":", alpha=0.8)
    ax.grid(alpha=0.2)
fig.suptitle("Best checkpoint selected at epoch 1 by validation MSE", fontsize=11)
save(fig, "training_curves")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
x = np.array([row["pairs"] for row in SUMMARY])
for field, interval, label, shift, color in [
    ("top1_rate", "top1_wilson_95", "Argmax correct", -0.06, "#2563eb"),
    ("unique_rate", "unique_wilson_95", "Unique true rank 1", 0.06, "#059669"),
]:
    y = np.array([row[field] for row in SUMMARY])
    bounds = np.array([row[interval] for row in SUMMARY])
    axes[0].errorbar(x + shift, 100 * y,
                     yerr=100 * np.maximum(0, np.stack((y - bounds[:, 0], bounds[:, 1] - y))),
                     fmt="o-", capsize=4, label=label, color=color)
axes[0].set(xlabel="Chosen-plaintext pairs", ylabel="Observed success (%)",
            xticks=x, ylim=(-3, 105), title="16-bit subkey recovery, 1,000 paired trials")
axes[0].legend(loc="lower right", fontsize=9)
axes[0].grid(alpha=0.2)
milliseconds = [1000 * row["median_scoring_seconds"] for row in SUMMARY]
bars = axes[1].bar(x, milliseconds, color="#475569", width=0.5)
axes[1].bar_label(bars, fmt="%.1f ms", padding=3)
axes[1].set(xlabel="Chosen-plaintext pairs", ylabel="Median candidate scoring time (ms)",
            xticks=x, ylim=(0, max(milliseconds) * 1.22), title="RTX 5060 Laptop; all 65,536 candidates")
axes[1].grid(axis="y", alpha=0.2)
fig.suptitle("Same keys and nested plaintext pairs; 95% Wilson intervals; fixed trained model", fontsize=10)
save(fig, "key_recovery_pairs")
print("Saved PNG and SVG: training_curves, key_recovery_pairs")
