#!/usr/bin/env python3
"""
Figure 2B — Annotation probe AUROC by category (horizontal boxplot).

Binary classification heads from clinvar-evo2-probe-v1, grouped by broad
annotation category (excluding Pathogenicity), sorted by median AUROC.

Input:  data/panels/fig2b.csv
Output: figures/figure2/panels/fig2b.{png,pdf}
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from theme.mayo_theme import apply_theme, save_figure

PANELS = ROOT / "artifacts"
OUT_STEM = ROOT / "figures" / "figure2" / "panels" / "fig2b"

apply_theme()

# Annotation category colors (from probe analysis notebook)
GROUP_COLORS = {
    "Regulatory":           "#4C72B0",
    "InterPro":             "#DD8452",
    "Pfam":                 "#CCBB44",
    "Protein & Structure":  "#55A868",
    "Substitution & Motif": "#C44E52",
    "Clinical":             "#8172B3",
    "Sequence & Region":    "#937860",
    "Splicing":             "#DA8BC3",
    "Expression":           "#8C8C8C",
    "Conservation":         "#CCB974",
}


def plot(ax):
    """Horizontal boxplot of binary head AUROC by annotation category."""
    df = pl.read_csv(PANELS / "fig2b.csv")

    # Group by broad_group, sort by median AUROC ascending
    group_stats = (
        df.group_by("broad_group")
        .agg(pl.col("auroc").median().alias("med"))
        .sort("med")
    )
    group_order = group_stats["broad_group"].to_list()

    data_by_group, labels, counts, color_list = [], [], [], []
    for g in group_order:
        vals = df.filter(pl.col("broad_group") == g)["auroc"].drop_nulls().to_numpy()
        if len(vals) > 0:
            data_by_group.append(vals)
            labels.append(g)
            counts.append(len(vals))
            color_list.append(GROUP_COLORS.get(g, "#888888"))

    positions = list(range(len(labels)))
    bp = ax.boxplot(
        data_by_group, positions=positions, vert=False, widths=0.55,
        patch_artist=True, showfliers=False,
        boxprops=dict(linewidth=0.8),
        medianprops=dict(color="black", linewidth=1.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, color in zip(bp["boxes"], color_list):
        patch.set_facecolor(color + "55")
        patch.set_edgecolor(color)

    # Jittered scatter overlay
    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(data_by_group, color_list)):
        jitter = rng.uniform(-0.22, 0.22, size=len(vals))
        ax.scatter(vals, i + jitter, s=10, alpha=0.45, color=color,
                   edgecolors="none", zorder=3)

    ax.set_yticks(positions)
    ax.set_yticklabels([f"{l}  (n={c})" for l, c in zip(labels, counts)])
    ax.set_xlabel("AUROC")
    ax.axvline(0.5, color="grey", linestyle="--", alpha=0.3)
    ax.set_xlim(0.48, 1.02)
    ax.grid(axis="x", alpha=0.2)


def main():
    fig, ax = plt.subplots(figsize=(8, 5))
    plot(ax)
    fig.tight_layout()
    save_figure(fig, OUT_STEM)
    print(f"Saved: {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
