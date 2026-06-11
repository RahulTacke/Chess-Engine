"""
Plot AutoResearch experiment results for both the 200K and 500K phases.
Produces a two-panel figure saved as Training/experiment_results.png.
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load CSV ──────────────────────────────────────────────────────────────────

def load_csv(filename):
    rows = []
    with open(os.path.join(SCRIPT_DIR, filename), newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "exp":       row["exp"],
                "desc":      row["description"],
                "val_loss":  float(row["val_loss"]),
                "kept":      row["kept"].strip().lower() == "yes",
            })
    return rows


# ── Best-so-far step line ─────────────────────────────────────────────────────

def best_so_far(losses):
    best, result = float("inf"), []
    for v in losses:
        best = min(best, v)
        result.append(best)
    return result


# ── Single-phase plot ─────────────────────────────────────────────────────────

def plot_phase(ax, rows, title):
    xs     = list(range(len(rows)))
    losses = [r["val_loss"] for r in rows]
    kept   = [r["kept"]     for r in rows]
    labels = [r["exp"]      for r in rows]
    bsf    = best_so_far(losses)

    # Scatter: kept = filled blue, rejected = hollow red
    for x, loss, k in zip(xs, losses, kept):
        if k:
            ax.scatter(x, loss, color="#2196F3", s=60, zorder=3)
        else:
            ax.scatter(x, loss, color="#EF5350", s=60,
                       facecolors="none", edgecolors="#EF5350",
                       linewidths=1.5, zorder=3)

    # Best-so-far step line
    ax.step(xs, bsf, where="post", color="#FF9800",
            linewidth=2, zorder=2, label="best so far")

    # Baseline and final best reference lines
    baseline = losses[0]
    final    = min(losses)
    ax.axhline(baseline, color="gray",   linewidth=1, linestyle="--", alpha=0.6)
    ax.axhline(final,    color="#4CAF50", linewidth=1, linestyle="--", alpha=0.8)
    ax.text(len(rows) - 0.5, baseline + 0.0003,
            f"baseline {baseline:.4f}", ha="right", va="bottom",
            fontsize=7, color="gray")
    ax.text(len(rows) - 0.5, final - 0.0005,
            f"best {final:.4f}", ha="right", va="top",
            fontsize=7, color="#4CAF50")

    # X-axis labels (experiment IDs)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Validation Loss (MSE)", fontsize=9)
    ax.set_xlabel("Experiment", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Improvement annotation
    improvement = (baseline - final) / baseline * 100
    ax.text(0.02, 0.04,
            f"improvement: {improvement:.1f}%  ({baseline:.4f} → {final:.4f})",
            transform=ax.transAxes, fontsize=8, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.8))


# ── Main ──────────────────────────────────────────────────────────────────────

phase1 = load_csv("original_exp_set_results.csv")
phase2 = load_csv("500k_fixed_seed_results.csv")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("AutoResearch — Chess CNN Evaluation Network",
             fontsize=13, fontweight="bold", y=0.98)

plot_phase(ax1, phase1, "Phase 1: 200K random subset (10 epochs each)")
plot_phase(ax2, phase2, "Phase 2: 500K fixed seed (10 epochs each)")

# Shared legend
kept_patch     = mpatches.Patch(color="#2196F3", label="kept (new best)")
rejected_patch = mpatches.Patch(facecolor="none",
                                edgecolor="#EF5350", label="rejected")
bsf_line       = plt.Line2D([0], [0], color="#FF9800",
                             linewidth=2, label="best so far")
fig.legend(handles=[kept_patch, rejected_patch, bsf_line],
           loc="upper right", fontsize=9, framealpha=0.9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(SCRIPT_DIR, "experiment_results.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
